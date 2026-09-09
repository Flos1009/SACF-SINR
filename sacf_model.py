"""Model components for SACF-SINR."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from context_transformer import MultimodalContextTransformer
from models import SimpleFCNet


MODEL_ID = "SACF-SINR"
EXPERIMENT_PARAMETER_KEYS = frozenset(
    {"mafq_n_queries", "tri_image_path", "mm_cache_path"}
)


class ImageToTextAdapter(nn.Module):
    """Map EVA-02 image features to the GritLM feature space."""

    def __init__(self, in_dim=1024, hidden_dim=2048, out_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, values):
        return F.normalize(self.net(values), dim=-1)


class TextEncoder(nn.Sequential):
    """Convert GritLM features into 256-dimensional context tokens."""

    def __init__(self, in_dim=4096, out_dim=256, hidden_dim=512):
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )


class SpeciesMetadataStore(nn.Module):
    """Provide one randomly sampled text and image feature per species."""

    def __init__(self, class_to_taxa, cache_path, token_dim=256):
        super().__init__()
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        row_of = {int(taxon): row for row, taxon in enumerate(data["taxon_id"])}
        rows = torch.tensor(
            [row_of.get(int(taxon), -1) for taxon in class_to_taxa],
            dtype=torch.int64,
        )
        safe_rows = rows.clamp(min=0)

        if "text_features" in data:
            text_features = data["text_features"]
            text_start = data["text_start"]
            text_count = data["text_count"]
            image_features = data["image_features"]
            image_start = data["image_start"]
            image_count = data["image_count"]
        else:
            required = {
                "pool",
                "pool_start",
                "pool_count",
                "pool_is_img",
                "img_raw",
                "img_raw_start",
                "img_raw_count",
            }
            missing = required.difference(data)
            if missing:
                raise RuntimeError(
                    f"The metadata cache is missing fields: {sorted(missing)}"
                )
            text_features = data["pool"]
            text_start = data["pool_start"]
            is_text = data["pool_is_img"] == 0
            owner = torch.repeat_interleave(
                torch.arange(len(data["pool_count"])), data["pool_count"]
            )
            text_count = torch.bincount(
                owner[is_text], minlength=len(data["pool_count"])
            )
            image_features = data["img_raw"]
            image_start = data["img_raw_start"]
            image_count = data["img_raw_count"]

        self.register_buffer("text_features", text_features, persistent=False)
        self.register_buffer("text_start", text_start[safe_rows], persistent=False)
        self.register_buffer(
            "txt_count", text_count[safe_rows] * (rows >= 0), persistent=False
        )
        self.register_buffer("image_features", image_features, persistent=False)
        self.register_buffer("image_start", image_start[safe_rows], persistent=False)
        self.register_buffer(
            "image_count",
            image_count[safe_rows] * (rows >= 0),
            persistent=False,
        )

        self.fallback = nn.Embedding(len(class_to_taxa), token_dim)
        nn.init.normal_(self.fallback.weight, mean=0.0, std=0.01)
        self.context_tf = None

    def sample(self, species, modality):
        if modality == "text":
            count = self.txt_count[species]
            start = self.text_start[species]
            source = self.text_features
        elif modality == "image":
            count = self.image_count[species]
            start = self.image_start[species]
            source = self.image_features
        else:
            raise ValueError(f"Unknown metadata modality: {modality}")

        offset = (
            (
                torch.rand(species.shape, device=species.device)
                * count.clamp(min=1).float()
            )
            .floor()
            .long()
        )
        index = (start + offset).clamp(max=max(0, source.shape[0] - 1))
        return source[index].float(), count > 0


def checkpoint_state(state_dict, params):
    """Prepare experiment checkpoint weights for the released model."""
    excluded_prefixes = ("geo_trunk.", "species_enc.")
    image_path = params.get("image_path")
    if image_path is None:
        image_path = (
            "adapter"
            if params.get("tri_image_path", "online") == "online"
            else "independent"
        )
    excluded_keys = [
        key
        for key in state_dict
        if key.startswith(excluded_prefixes)
        or key == "species_params"
        or (image_path == "adapter" and key.startswith("image_stem."))
    ]
    uses_aligner_name = any(key.startswith("image_aligner.") for key in state_dict)
    if not excluded_keys and not uses_aligner_name:
        return state_dict
    has_shared_encoder = any(key.startswith("geo_trunk.") for key in excluded_keys)
    if (
        has_shared_encoder
        and params.get("paper_shared_representation_enabled") is not False
    ):
        raise RuntimeError(
            "This checkpoint uses a shared representation encoder and is "
            "incompatible with the released model."
        )
    return type(state_dict)(
        (
            (
                key.replace("image_aligner.", "image_adapter.", 1)
                if key.startswith("image_aligner.")
                else key
            ),
            value,
        )
        for key, value in state_dict.items()
        if key not in excluded_keys
    )


def checkpoint_parameters(params):
    """Map experiment checkpoint metadata to released configuration fields."""
    values = dict(params)
    if EXPERIMENT_PARAMETER_KEYS.issubset(values):
        values["model"] = MODEL_ID
    values.setdefault(
        "image_path",
        "adapter"
        if values.get("tri_image_path", "online") == "online"
        else "independent",
    )
    values.setdefault(
        "metadata_cache_path",
        values.get("mm_cache_path", "data/train/metadata_cache.pt"),
    )
    values.setdefault("text_dim", values.get("tri_text_dim", 4096))
    values.setdefault("image_dim", values.get("tri_image_dim", 1024))
    values.setdefault("num_queries", values.get("mafq_n_queries", 3))
    values.setdefault("context_prior", values.get("mafq_anchor_enabled", True))
    values.setdefault(
        "mediated_image_attention",
        values.get("mafq_mediated_image_attention", True),
    )
    return values


def is_sacf_checkpoint(params):
    """Return whether checkpoint parameters describe a SACF-SINR model."""
    return params.get("model") == MODEL_ID or EXPERIMENT_PARAMETER_KEYS.issubset(params)


class SACFSINRModel(nn.Module):
    """Few-shot species distribution model described in the manuscript."""

    sacf_sinr_model = True

    def __init__(
        self,
        num_inputs,
        num_filts,
        pos_enc_depth,
        class_to_taxa,
        metadata_cache_path,
        mm_init_ckpt="",
        image_path="adapter",
        text_dim=4096,
        image_dim=1024,
        num_queries=3,
        context_prior=True,
        mediated_image_attention=True,
    ):
        super().__init__()
        if image_path not in {"adapter", "independent"}:
            raise ValueError("image_path must be 'adapter' or 'independent'")

        self.pos_enc = SimpleFCNet(
            num_inputs,
            num_filts,
            num_filts,
            depth=pos_enc_depth,
            dropout_p=0.1,
        )
        self.pos_enc.class_emb = nn.Identity()

        self.species_emb = SpeciesMetadataStore(
            class_to_taxa, metadata_cache_path, token_dim=num_filts
        )
        self.text_stem = TextEncoder(text_dim, num_filts)
        self.image_path = image_path
        if image_path == "adapter":
            self.image_adapter = ImageToTextAdapter(in_dim=image_dim, out_dim=text_dim)
        else:
            self.image_stem = nn.Sequential(
                nn.Linear(image_dim, 512),
                nn.GELU(),
                nn.Linear(512, num_filts),
            )

        self.species_emb.context_tf = MultimodalContextTransformer(
            dim=num_filts,
            layers=4,
            heads=2,
            ff=512,
            dropout=0.1,
            n_queries=num_queries,
            context_prior=context_prior,
            mediated_image_attention=mediated_image_attention,
        )

        self.text_dim = int(text_dim)
        self.image_dim = int(image_dim)
        if mm_init_ckpt:
            self._load_location_encoder(mm_init_ckpt)

    def _load_location_encoder(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source = dict(checkpoint["state_dict"])
        if "pos_enc.feats.0.weight" not in source and "feats.0.weight" in source:
            source = {
                (f"pos_enc.{key}" if key.startswith("feats.") else key): value
                for key, value in source.items()
            }
        expected = {
            key: value
            for key, value in self.state_dict().items()
            if key.startswith("pos_enc.feats.")
        }
        compatible = {
            key: source[key]
            for key, value in expected.items()
            if key in source and source[key].shape == value.shape
        }
        missing = sorted(set(expected).difference(compatible))
        if missing:
            raise RuntimeError(
                f"The warm-start checkpoint is missing location encoder tensors: {missing}"
            )
        self.load_state_dict(compatible, strict=False)

    @staticmethod
    def _normalise_token(values):
        return F.normalize(values.float(), dim=-1) * 5.0

    def encode_location(self, features):
        return self.pos_enc(features)

    def encode_text(self, values):
        return self._normalise_token(self.text_stem(values.float()))

    def _readonly_text_encoder(self, values):
        output = values
        for layer in self.text_stem:
            if isinstance(layer, nn.Linear):
                bias = layer.bias.detach() if layer.bias is not None else None
                output = F.linear(output, layer.weight.detach(), bias)
            else:
                output = layer(output)
        return output

    def encode_image(self, values):
        if self.image_path == "independent":
            encoded = self.image_stem(values.float())
        else:
            adapted = self.image_adapter(F.normalize(values.float(), dim=-1))
            encoded = self._readonly_text_encoder(adapted)
        return self._normalise_token(encoded)

    def build_content_tokens(self, species):
        text_values, has_text = self.species_emb.sample(species, "text")
        image_values, has_image = self.species_emb.sample(species, "image")
        text_tokens = self.encode_text(text_values)
        image_tokens = self.encode_image(image_values)

        no_metadata = ~(has_text | has_image)
        fallback = self._normalise_token(self.species_emb.fallback(species))
        text_tokens = torch.where(no_metadata.unsqueeze(1), fallback, text_tokens)
        has_text = has_text | no_metadata
        return torch.stack((text_tokens, image_tokens), dim=1), has_text, has_image
