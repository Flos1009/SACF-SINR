"""Multimodal context fusion used by SACF-SINR."""

import torch
import torch.nn as nn


class MultimodalContextTransformer(nn.Module):
    """Generate species range fields from multimodal context tokens.

    The pooled context prior conditions the learnable field queries before
    self-attention. Image-mediated attention blocks direct query-to-image
    attention when text or observation locations are also available.
    """

    def __init__(
        self,
        dim=256,
        layers=4,
        heads=2,
        ff=512,
        dropout=0.1,
        n_queries=3,
        context_prior=True,
        mediated_image_attention=True,
    ):
        super().__init__()
        self.n_queries = int(n_queries)
        if self.n_queries < 1:
            raise ValueError("n_queries must be positive")

        self.attention_heads = int(heads)
        self.context_prior = bool(context_prior)
        self.mediated_image_attention = bool(mediated_image_attention)

        self.cls = nn.Parameter(torch.randn(self.n_queries, dim) * 0.02)
        self.reg = nn.Parameter(torch.randn(dim) * 0.02)
        self.type_emb = nn.Embedding(5, dim)
        with torch.no_grad():
            self.type_emb.weight *= 0.02
            nn.init.xavier_uniform_(self.cls)
            nn.init.xavier_uniform_(self.reg.unsqueeze(0))
            nn.init.xavier_uniform_(self.type_emb.weight)

        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="relu",
        )
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.in_norm = nn.LayerNorm(dim)
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(dim, dim),
        )
        self.anchor_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.anchor_proj = nn.Linear(dim, self.n_queries * dim, bias=False)
        nn.init.xavier_uniform_(self.anchor_proj.weight, gain=1e-3)

    @staticmethod
    def _masked_mean(tokens, valid):
        weight = valid.to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * weight).sum(1) / weight.sum(1).clamp(min=1.0)

    @staticmethod
    def _mediated_image_attention_mask(
        key_padding_mask, n_queries, evidence_groups, heads=None
    ):
        if key_padding_mask.ndim != 2:
            raise ValueError("key_padding_mask must have shape [batch, sequence]")
        batch, sequence = key_padding_mask.shape
        evidence_start = int(n_queries) + 1
        if sequence != evidence_start + len(evidence_groups):
            raise ValueError("evidence_groups does not match sequence length")

        mask = torch.zeros(
            batch,
            sequence,
            sequence,
            dtype=torch.bool,
            device=key_padding_mask.device,
        )
        image_columns = [
            index for index, group in enumerate(evidence_groups) if group == "image"
        ]
        if image_columns:
            valid = ~key_padding_mask[:, evidence_start:]
            other_valid = torch.zeros(
                batch, dtype=torch.bool, device=key_padding_mask.device
            )
            for index, group in enumerate(evidence_groups):
                if group != "image":
                    other_valid |= valid[:, index]
            columns = torch.tensor(
                [evidence_start + index for index in image_columns],
                dtype=torch.long,
                device=key_padding_mask.device,
            )
            mask[other_valid, :n_queries, columns] = True

        if heads is not None:
            mask = mask.repeat_interleave(int(heads), dim=0)
        return mask

    def forward(self, location_tokens, location_valid, content_tokens, content_valid):
        batch, _, dim = location_tokens.shape
        device = location_tokens.device
        location_valid = location_valid.bool()
        content_valid = content_valid.bool()
        if content_tokens.shape[:2] != content_valid.shape:
            raise ValueError("content token and validity shapes do not match")
        if location_tokens.shape[:2] != location_valid.shape:
            raise ValueError("location token and validity shapes do not match")

        if self.context_prior:
            summary_tokens = torch.cat((content_tokens, location_tokens), dim=1)
            summary_valid = torch.cat((content_valid, location_valid), dim=1)
            normalised = self.anchor_norm(summary_tokens)
            pooled = self._masked_mean(normalised, summary_valid).detach()
            prior = self.anchor_proj(pooled).view(batch, self.n_queries, dim)
        else:
            prior = torch.zeros(
                batch,
                self.n_queries,
                dim,
                dtype=location_tokens.dtype,
                device=device,
            )

        queries = self.cls.unsqueeze(0).expand(batch, -1, -1) + prior
        register = self.reg.view(1, 1, -1).expand(batch, 1, -1)
        metadata = torch.cat((queries, register), dim=1)
        metadata_types = torch.cat(
            (
                self.type_emb.weight[0].expand(self.n_queries, -1),
                self.type_emb.weight[3].unsqueeze(0),
            ),
            dim=0,
        )
        metadata = metadata + metadata_types.unsqueeze(0)

        content_type_ids = torch.tensor(
            [1, 4][: content_tokens.shape[1]], device=device
        )
        content_tokens = content_tokens + self.type_emb.weight[content_type_ids]
        location_tokens = location_tokens + self.type_emb.weight[2]

        sequence = torch.cat((metadata, content_tokens, location_tokens), dim=1)
        padding = torch.cat(
            (
                torch.zeros(
                    batch,
                    self.n_queries + 1,
                    dtype=torch.bool,
                    device=device,
                ),
                ~content_valid,
                ~location_valid,
            ),
            dim=1,
        )
        evidence_groups = ["text", "image"][: content_tokens.shape[1]] + [
            "location"
        ] * location_tokens.shape[1]

        attention_mask = None
        if self.mediated_image_attention:
            attention_mask = self._mediated_image_attention_mask(
                padding,
                self.n_queries,
                evidence_groups,
                heads=self.attention_heads,
            )
        output = self.tf(
            self.in_norm(sequence),
            mask=attention_mask,
            src_key_padding_mask=padding,
        )
        fields = self.decoder(output[:, : self.n_queries])
        return fields
