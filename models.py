"""Location encoders and model construction for SACF-SINR."""

import torch
import torch.nn as nn


class ResLayer(nn.Module):
    def __init__(self, width, dropout=0.5):
        super().__init__()
        self.l_size = width
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.w1 = nn.Linear(width, width)
        self.w2 = nn.Linear(width, width)

    def forward(self, values):
        residual = self.w1(values)
        residual = self.nonlin1(residual)
        residual = self.dropout1(residual)
        residual = self.w2(residual)
        residual = self.nonlin2(residual)
        return values + residual


class ResidualFCNet(nn.Module):
    """SINR location encoder used for warm-start pretraining."""

    def __init__(self, num_inputs, num_classes, num_filts, depth=4, dropout_p=0.5):
        super().__init__()
        self.class_emb = nn.Linear(num_filts, num_classes, bias=False)
        layers = [nn.Linear(num_inputs, num_filts), nn.ReLU(inplace=True)]
        layers.extend(ResLayer(num_filts, dropout_p) for _ in range(depth))
        self.feats = nn.Sequential(*layers)

    def forward(self, values, return_feats=False):
        location_embedding = self.feats(values)
        if return_feats:
            return location_embedding
        return torch.sigmoid(self.class_emb(location_embedding))


class SimpleFCNet(ResidualFCNet):
    """Location encoder variant that returns unnormalised embeddings."""

    def forward(self, values, return_feats=True):
        if not return_feats:
            raise ValueError("SimpleFCNet returns location embeddings only")
        return self.class_emb(self.feats(values))


def get_model(params):
    """Construct the SINR pretraining model or SACF-SINR."""
    from sacf_model import checkpoint_parameters

    params = checkpoint_parameters(params)
    model_id = params["model"]

    if model_id == "ResidualFCNet":
        return ResidualFCNet(
            params["input_dim"],
            params["num_classes"],
            params["num_filts"],
            params["depth"],
        )
    if model_id == "SACF-SINR":
        from sacf_model import SACFSINRModel

        return SACFSINRModel(
            params["input_dim"],
            params["num_filts"],
            params["pos_enc_depth"],
            class_to_taxa=params["class_to_taxa"],
            metadata_cache_path=params["metadata_cache_path"],
            mm_init_ckpt=params.get("mm_init_ckpt", ""),
            image_path=params["image_path"],
            text_dim=params["text_dim"],
            image_dim=params["image_dim"],
            num_queries=params["num_queries"],
            context_prior=params["context_prior"],
            mediated_image_attention=params["mediated_image_attention"],
        )
    raise ValueError(f"Unsupported model: {params['model']}")
