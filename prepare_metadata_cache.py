"""Build the compact metadata cache used to train SACF-SINR."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def indexed_rows(feature_file, *, sort_labels=False):
    data = torch.load(feature_file, map_location="cpu", weights_only=False)
    required = {"taxon_id", "keys", "data"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{feature_file} is missing fields: {sorted(missing)}")
    taxa = data["taxon_id"].tolist()
    rows = {}
    labels = {}
    for row, key in enumerate(data["keys"]):
        taxon_index = int(key[0])
        taxon = int(taxa[taxon_index])
        rows.setdefault(taxon, []).append(row)
        labels[row] = str(key[1])
    if sort_labels:
        for taxon_rows in rows.values():
            taxon_rows.sort(key=labels.__getitem__)
    return data["data"], rows


def group_features(features, rows_by_taxon, taxa, *, normalise=False):
    starts = []
    counts = []
    ordered_rows = []
    offset = 0
    for taxon in taxa:
        rows = rows_by_taxon.get(taxon, [])
        starts.append(offset)
        counts.append(len(rows))
        ordered_rows.extend(rows)
        offset += len(rows)
    if ordered_rows:
        grouped = features.index_select(0, torch.tensor(ordered_rows))
        if normalise:
            grouped = F.normalize(grouped.float(), dim=-1)
        grouped = grouped.half()
    else:
        grouped = features.new_empty((0, features.shape[1]), dtype=torch.float16)
    return grouped, torch.tensor(starts), torch.tensor(counts)


def build_cache(text_path, image_path, output_path):
    text_features, text_rows = indexed_rows(text_path, sort_labels=True)
    image_features, image_rows = indexed_rows(image_path)
    taxa = sorted(set(text_rows) | set(image_rows))
    grouped_text, text_starts, text_counts = group_features(
        text_features, text_rows, taxa
    )
    grouped_images, image_starts, image_counts = group_features(
        image_features, image_rows, taxa, normalise=True
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "taxon_id": torch.tensor(taxa),
            "text_features": grouped_text,
            "text_start": text_starts,
            "text_count": text_counts,
            "image_features": grouped_images,
            "image_start": image_starts,
            "image_count": image_counts,
        },
        output,
    )
    print(
        f"Metadata cache saved to {output} "
        f"({len(taxa):,} species, {len(grouped_text):,} text features, "
        f"{len(grouped_images):,} image features)."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="data/train/wiki_data.pt")
    parser.add_argument("--images", default="data/image_features.pt")
    parser.add_argument("--out", default="data/train/metadata_cache.pt")
    args = parser.parse_args()
    build_cache(args.text, args.images, args.out)


if __name__ == "__main__":
    main()
