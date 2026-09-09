"""Few-shot and zero-shot evaluation for SACF-SINR."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import models
import utils
from sacf_model import (
    checkpoint_parameters,
    checkpoint_state,
    is_sacf_checkpoint,
)


DEFAULT_CONTEXT_SIZES = (0, 1, 2, 5, 10, 20, 50)
DEFAULT_MODES = ("locs", "text", "image", "text+image")


def average_precision_columns(scores, labels):
    """Compute average precision independently for each matrix column."""
    order = torch.argsort(scores, dim=0, descending=True)
    sorted_labels = torch.gather(labels, 0, order)
    true_positives = torch.cumsum(sorted_labels, dim=0, dtype=torch.float32)
    false_positives = torch.cumsum(1 - sorted_labels, dim=0, dtype=torch.float32)
    positives = labels.sum(dim=0, dtype=torch.float32)
    recall = true_positives / positives.clamp_min(1)
    precision = true_positives / (true_positives + false_positives)
    recall_edges = torch.cat(
        (
            torch.zeros(1, labels.shape[1], device=labels.device),
            recall,
            torch.ones(1, labels.shape[1], device=labels.device),
        ),
        dim=0,
    )
    recall_steps = (recall_edges[1:] - recall_edges[:-1])[:-1]
    return (recall_steps * precision).sum(dim=0)


def average_precision_columns_safe(scores, labels, minimum_columns=16):
    """Retry average precision in column blocks after a CUDA OOM."""
    try:
        return average_precision_columns(scores, labels)
    except torch.cuda.OutOfMemoryError:
        if scores.device.type != "cuda" or scores.shape[1] <= minimum_columns:
            raise
        torch.cuda.empty_cache()
        midpoint = scores.shape[1] // 2
        return torch.cat(
            (
                average_precision_columns_safe(
                    scores[:, :midpoint], labels[:, :midpoint], minimum_columns
                ),
                average_precision_columns_safe(
                    scores[:, midpoint:], labels[:, midpoint:], minimum_columns
                ),
            )
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Trained SACF-SINR checkpoint.")
    parser.add_argument("--out", default=None, help="Output JSON path.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--metadata-cache", default=None)
    parser.add_argument("--image-path", default="data/image_features.pt")
    parser.add_argument("--text-path", default="data/eval/gpt_data.pt")
    parser.add_argument("--snt-path", default="data/eval/snt/snt_res_5.npy")
    parser.add_argument("--iucn-path", default="data/eval/iucn/iucn_res_5.json")
    parser.add_argument("--occurrence-path", default="data/eval/positive_eval_data.npz")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--ks", type=int, nargs="+", default=list(DEFAULT_CONTEXT_SIZES)
    )
    parser.add_argument(
        "--datasets", choices=("snt", "iucn"), nargs="+", default=["snt", "iucn"]
    )
    parser.add_argument(
        "--modes", choices=DEFAULT_MODES, nargs="+", default=list(DEFAULT_MODES)
    )
    parser.add_argument("--score-chunk", type=int, default=256)
    parser.add_argument(
        "--image-index",
        type=int,
        default=None,
        help="Use this zero-based row after sorting each species' images by filename.",
    )
    parser.add_argument(
        "--save-per-species",
        action="store_true",
        help="Include the AP of every evaluated species in the output JSON.",
    )
    return parser


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --device cpu.")
    return device


def load_model(checkpoint_path, metadata_cache, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_params = dict(checkpoint["params"])
    if not is_sacf_checkpoint(source_params):
        raise ValueError("The checkpoint is not a SACF-SINR model.")
    params = checkpoint_parameters(source_params)
    params["input_dim"] = checkpoint["state_dict"]["pos_enc.feats.0.weight"].shape[1]
    params["mm_init_ckpt"] = ""
    if metadata_cache is not None:
        params["metadata_cache_path"] = metadata_cache
    model = models.get_model(params)
    state = checkpoint_state(checkpoint["state_dict"], source_params)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), params


def load_eval_images(path, image_index):
    if image_index is None:
        return utils.load_first_image_embeddings(path), None
    if image_index < 0:
        raise SystemExit("--image-index must be non-negative.")
    selected = {}
    metadata = {}
    for taxon, entries in utils.load_image_embedding_rows(path).items():
        if image_index >= len(entries):
            continue
        filename, feature = entries[image_index]
        selected[taxon] = F.normalize(feature.float(), dim=-1)
        metadata[str(taxon)] = {
            "image_index": image_index,
            "filename": filename,
            "available_images": len(entries),
        }
    return selected, metadata


def encode_locations(model, encoder, coordinates, device, chunk_size=65536):
    base = encoder.encode(coordinates)
    output = None
    with torch.no_grad():
        for start in range(0, len(base), chunk_size):
            stop = min(start + chunk_size, len(base))
            block = model.encode_location(base[start:stop].to(device))
            if output is None:
                output = torch.empty(
                    (len(base),) + tuple(block.shape[1:]),
                    dtype=block.dtype,
                    device=device,
                )
            output[start:stop] = block
    return output


def build_metadata_tokens(model, taxa, text_features, image_features, device):
    tokens = {}
    text_taxa = [taxon for taxon in taxa if taxon in text_features]
    image_taxa = [taxon for taxon in taxa if taxon in image_features]
    with torch.no_grad():
        if text_taxa:
            values = torch.stack([text_features[taxon] for taxon in text_taxa]).to(
                device
            )
            for taxon, token in zip(text_taxa, model.encode_text(values)):
                tokens.setdefault(taxon, {})["text"] = token
        if image_taxa:
            values = torch.stack([image_features[taxon] for taxon in image_taxa]).to(
                device
            )
            for taxon, token in zip(image_taxa, model.encode_image(values)):
                tokens.setdefault(taxon, {})["image"] = token
    return tokens


def sample_location_tokens(
    model, encoder, taxa, observations, context_size, seed, device, normalise=True
):
    generator = np.random.RandomState(seed * 1000 + context_size)
    kept = []
    samples = []
    for taxon in taxa:
        if context_size == 0:
            points = np.zeros((1, 2), dtype=np.float32)
        else:
            available = observations.get(taxon)
            if available is None or len(available) == 0:
                continue
            indices = generator.choice(
                len(available), size=min(context_size, len(available)), replace=False
            )
            points = available[indices]
        kept.append(taxon)
        samples.append(points)
    if not kept:
        return None

    width = context_size if context_size > 0 else 1
    coordinates = np.zeros((len(kept), width, 2), dtype=np.float32)
    valid = np.zeros((len(kept), width), dtype=np.bool_)
    for row, points in enumerate(samples):
        coordinates[row, : len(points)] = points
        valid[row, : len(points)] = True
    features = encoder.encode(
        torch.from_numpy(coordinates).to(device).reshape(-1, 2), normalize=normalise
    ).reshape(len(kept), width, -1)
    with torch.no_grad():
        location_tokens = model.encode_location(features)
    valid_tokens = torch.from_numpy(valid).to(device)
    if context_size == 0:
        valid_tokens.zero_()
    return kept, location_tokens, valid_tokens


def select_metadata(rows, mode, metadata_tokens, device, dimension=256):
    kept = []
    row_indices = []
    values = []
    for row_index, taxon in enumerate(rows):
        text = (
            metadata_tokens.get(taxon, {}).get("text")
            if mode in {"text", "text+image"}
            else None
        )
        image = (
            metadata_tokens.get(taxon, {}).get("image")
            if mode in {"image", "text+image"}
            else None
        )
        if mode != "locs" and text is None and image is None:
            continue
        kept.append(taxon)
        row_indices.append(row_index)
        values.append((text, image))
    if not kept:
        return None

    tokens = torch.zeros(len(kept), 2, dimension, device=device)
    valid = torch.zeros(len(kept), 2, dtype=torch.bool, device=device)
    for row, pair in enumerate(values):
        for column, token in enumerate(pair):
            if token is not None:
                tokens[row, column] = token
                valid[row, column] = True
    return kept, row_indices, tokens, valid


def score_fields(location_embeddings, fields, location_chunk=65536):
    if fields.ndim != 3:
        raise ValueError(
            "Expected range fields with shape [species, fields, features]."
        )
    scores = torch.empty(
        len(location_embeddings), len(fields), device=location_embeddings.device
    )
    for location_start in range(0, len(location_embeddings), location_chunk):
        location_stop = min(location_start + location_chunk, len(location_embeddings))
        locations = location_embeddings[location_start:location_stop].float()
        reduced = None
        for field_index in range(fields.shape[1]):
            term = locations @ fields[:, field_index].float().T
            reduced = term if reduced is None else torch.logaddexp(reduced, term)
        scores[location_start:location_stop] = reduced
    return scores


def prepare_snt_evaluation(data, taxa, device):
    generator = np.random.default_rng(7499)
    by_taxon = {}
    maximum_length = 0
    taxon_row = {taxon: row for row, taxon in enumerate(taxa)}
    for taxon in taxa:
        row = taxon_row[taxon]
        labels = (np.asarray(data["labels_per_species"][row]) > 0).astype(np.float32)
        indices = np.asarray(data["loc_indices_per_species"][row])
        validation_size = int(np.floor(len(indices) * 0.5))
        selected = generator.permutation(len(indices))[validation_size:]
        indices, labels = indices[selected], labels[selected]
        if labels.sum() == 0:
            continue
        by_taxon[taxon] = (indices, labels)
        maximum_length = max(maximum_length, len(indices))

    def compute(scores, rows, chunk_size):
        evaluated = [taxon for taxon in rows if taxon in by_taxon]
        score_column = {taxon: column for column, taxon in enumerate(rows)}
        results = []
        sequence = torch.arange(maximum_length, device=device).unsqueeze(1)
        for start in range(0, len(evaluated), chunk_size):
            subset = evaluated[start : start + chunk_size]
            index_matrix = np.zeros((maximum_length, len(subset)), dtype=np.int64)
            label_matrix = np.zeros((maximum_length, len(subset)), dtype=np.float32)
            lengths = np.zeros(len(subset), dtype=np.int64)
            for column, taxon in enumerate(subset):
                indices, labels = by_taxon[taxon]
                lengths[column] = len(indices)
                index_matrix[: len(indices), column] = indices
                label_matrix[: len(labels), column] = labels
            columns = torch.tensor(
                [score_column[taxon] for taxon in subset], device=device
            )
            gathered = scores[:, columns].gather(
                0, torch.from_numpy(index_matrix).to(device)
            )
            padding = sequence >= torch.from_numpy(lengths).to(device).unsqueeze(0)
            gathered.masked_fill_(padding, float("-inf"))
            labels = torch.from_numpy(label_matrix).to(device)
            results.extend(average_precision_columns(gathered, labels).cpu().tolist())
        return evaluated, results

    return compute


def load_occurrences(path, required_taxa):
    data = np.load(path, allow_pickle=True)
    locations = data["locs"]
    labels = data["labels"]
    taxa = [int(value) for value in data["class_to_taxa"]]
    required = set(required_taxa)
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels, minlength=len(taxa))
    starts = np.cumsum(counts) - counts
    return {
        taxon: locations[order[starts[row] : starts[row] + counts[row]]].astype(
            np.float32
        )
        for row, taxon in enumerate(taxa)
        if taxon in required and counts[row] > 0
    }


def evaluate(args):
    device = resolve_device(args.device)
    model, params = load_model(args.ckpt, args.metadata_cache, device)
    snt_data = np.load(args.snt_path, allow_pickle=True).item()
    snt_taxa = [int(taxon) for taxon in snt_data["taxa"]]
    with open(args.iucn_path, encoding="utf-8") as stream:
        iucn_data = json.load(stream)
    presence = {
        int(taxon): np.asarray(indices)
        for taxon, indices in iucn_data["taxa_presence"].items()
        if len(indices) > 0
    }
    iucn_taxa = list(presence)
    all_taxa = sorted(set(snt_taxa) | set(iucn_taxa))

    text_data = utils.load_keyed_embeddings(args.text_path)
    text_features = {
        taxon: values["range"]
        for taxon, values in text_data.items()
        if "range" in values
    }
    image_features, image_metadata = load_eval_images(args.image_path, args.image_index)
    metadata_tokens = build_metadata_tokens(
        model, all_taxa, text_features, image_features, device
    )

    coordinate_encoder = utils.CoordEncoder(params.get("input_enc", "sin_cos"))
    snt_location_embeddings = encode_locations(
        model, coordinate_encoder, snt_data["obs_locs"], device
    )
    iucn_locations = np.asarray(iucn_data["locs"], dtype=np.float32)
    iucn_location_embeddings = encode_locations(
        model, coordinate_encoder, iucn_locations, device
    )
    snt_average_precision = prepare_snt_evaluation(snt_data, snt_taxa, device)

    observations = load_occurrences(args.occurrence_path, all_taxa)
    observations_by_dataset = {
        "snt": {
            taxon: observations[taxon] for taxon in snt_taxa if taxon in observations
        },
        "iucn": {
            taxon: observations[taxon] for taxon in iucn_taxa if taxon in observations
        },
    }
    taxa_by_dataset = {"snt": snt_taxa, "iucn": iucn_taxa}
    units = [
        (context_size, mode)
        for context_size in args.ks
        for mode in args.modes
        if not (context_size == 0 and mode == "locs")
    ]
    aggregate = {
        (context_size, mode): {dataset: [] for dataset in args.datasets}
        for context_size, mode in units
    }
    per_seed = {}
    per_species = {}
    progress = tqdm(total=len(args.seeds) * len(units), desc="evaluation")

    for seed in args.seeds:
        location_cache = {}
        for context_size, mode in units:
            seed_result = {}
            for dataset_name in args.datasets:
                cache_key = (dataset_name, context_size)
                if cache_key not in location_cache:
                    location_cache[cache_key] = sample_location_tokens(
                        model,
                        coordinate_encoder,
                        taxa_by_dataset[dataset_name],
                        observations_by_dataset[dataset_name],
                        context_size,
                        seed,
                        device,
                        normalise=params.get("ctx_pool_norm", True),
                    )
                location_part = location_cache[cache_key]
                if location_part is None:
                    continue
                rows, location_tokens, location_valid = location_part
                metadata_part = select_metadata(
                    rows, mode, metadata_tokens, device, params["num_filts"]
                )
                if metadata_part is None:
                    continue
                kept, row_indices, content_tokens, content_valid = metadata_part
                selected = torch.tensor(row_indices, device=device)
                with torch.no_grad():
                    fields = model.species_emb.context_tf(
                        location_tokens[selected],
                        location_valid[selected],
                        content_tokens,
                        content_valid,
                    )

                if dataset_name == "snt":
                    values = []
                    evaluated_taxa = []
                    for start in range(0, len(kept), args.score_chunk):
                        subset = kept[start : start + args.score_chunk]
                        scores = score_fields(
                            snt_location_embeddings,
                            fields[start : start + len(subset)],
                        )
                        taxa_result, ap_values = snt_average_precision(
                            scores, subset, args.score_chunk
                        )
                        evaluated_taxa.extend(taxa_result)
                        values.extend(ap_values)
                else:
                    values = []
                    evaluated_taxa = []
                    for start in range(0, len(kept), args.score_chunk):
                        subset = kept[start : start + args.score_chunk]
                        scores = score_fields(
                            iucn_location_embeddings,
                            fields[start : start + len(subset)],
                        )
                        for offset in range(0, len(subset), args.score_chunk):
                            taxa_subset = subset[offset : offset + args.score_chunk]
                            labels = torch.zeros(
                                len(iucn_locations),
                                len(taxa_subset),
                                dtype=torch.float16,
                                device=device,
                            )
                            for column, taxon in enumerate(taxa_subset):
                                labels[
                                    torch.from_numpy(presence[taxon]).to(device), column
                                ] = 1
                            ap_values = (
                                average_precision_columns_safe(
                                    scores[:, offset : offset + len(taxa_subset)],
                                    labels,
                                )
                                .cpu()
                                .tolist()
                            )
                            evaluated_taxa.extend(taxa_subset)
                            values.extend(ap_values)

                aggregate[(context_size, mode)][dataset_name].extend(values)
                seed_result[dataset_name] = float(np.mean(values)) if values else None
                if args.save_per_species:
                    per_species.setdefault(str(seed), {}).setdefault(
                        f"{mode}@K{context_size}", {}
                    ).setdefault(dataset_name, {}).update(
                        {
                            str(taxon): float(value)
                            for taxon, value in zip(evaluated_taxa, values)
                        }
                    )
            per_seed.setdefault(str(seed), {})[f"{mode}@K{context_size}"] = seed_result
            progress.update()
    progress.close()

    results = {}
    for context_size, mode in units:
        key = f"{mode}@K{context_size}"
        results[key] = {
            dataset: (
                round(float(np.mean(aggregate[(context_size, mode)][dataset])), 6)
                if aggregate[(context_size, mode)][dataset]
                else None
            )
            for dataset in args.datasets
        }
        values = " | ".join(
            f"{dataset.upper()} {results[key][dataset]:.4f}"
            if results[key][dataset] is not None
            else f"{dataset.upper()} --"
            for dataset in args.datasets
        )
        print(f"{key:18s} {values}")

    results["_per_seed"] = per_seed
    if args.save_per_species:
        results["_per_species"] = per_species
    if image_metadata is not None:
        results["_image_selection"] = {
            "policy": "filename-sorted per-species row",
            "zero_based_index": args.image_index,
            "species": image_metadata,
        }
    return results


def main():
    args = build_parser().parse_args()
    results = evaluate(args)
    output_path = (
        Path(args.out)
        if args.out
        else Path(args.ckpt).with_name("fewshot_results.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
