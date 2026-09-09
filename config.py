"""Configuration objects for SACF-SINR and SINR pretraining."""

from copy import deepcopy


DEFAULTS = {
    "device": "cuda",
    "save_base": "./experiments/",
    "experiment_name": "SACF-SINR",
    "train_seed": 2026,
    "bg_type": "spherical",
    "species_set": "all_minus_eval",
    "hard_cap_seed": 9472,
    "hard_cap_num_per_class": 100,
    "obs_file": "geo_prior_train.csv",
    "taxa_file": "geo_prior_train_meta.json",
    "num_workers": 8,
    "model": "SACF-SINR",
    "num_filts": 256,
    "input_enc": "sin_cos",
    "depth": 4,
    "pos_enc_depth": 4,
    "metadata_cache_path": "data/train/metadata_cache.pt",
    "mm_init_ckpt": "experiments/SINR-sc-20ep-noeval/model.pt",
    "image_path": "adapter",
    "text_dim": 4096,
    "image_dim": 1024,
    "num_queries": 3,
    "context_prior": True,
    "mediated_image_attention": True,
    "batch_size": 2048,
    "lr": 5e-4,
    "num_epochs": 40,
    "lr_decay": 0.98,
    "amp": True,
    "grad_clip": 1.0,
    "weight_decay": 0.0,
    "save_every_epochs": 0,
    "optimizer": "adamwr",
    "adamwr_t0": 10,
    "adamwr_t_mult": 1,
    "adamwr_eta_min": 5e-5,
    "resume_ckpt": "",
    "loss": "sacf_context",
    "pos_weight": 2048,
    "ctx_k": 20,
    "ctx_loc_token_drop": 0.3,
    "ctx_pool_norm": True,
}


class _Section:
    def __init__(self, params, keys):
        object.__setattr__(self, "_params", params)
        object.__setattr__(self, "_keys", frozenset(keys))

    def __getattr__(self, name):
        if name not in self._keys:
            raise AttributeError(name)
        return self._params[name]

    def __setattr__(self, name, value):
        if name not in self._keys:
            raise AttributeError(f"Unknown configuration field: {name}")
        self._params[name] = value


class ExperimentConfig:
    """Structured view over the flat dictionary stored in checkpoints."""

    def __init__(self):
        self._params = deepcopy(DEFAULTS)
        self.base = _Section(
            self._params,
            {
                "device",
                "save_base",
                "experiment_name",
                "train_seed",
                "bg_type",
            },
        )
        self.data = _Section(
            self._params,
            {
                "species_set",
                "hard_cap_seed",
                "hard_cap_num_per_class",
                "obs_file",
                "taxa_file",
                "num_workers",
            },
        )
        self.model = _Section(
            self._params,
            {
                "model",
                "num_filts",
                "input_enc",
                "depth",
                "pos_enc_depth",
                "metadata_cache_path",
                "mm_init_ckpt",
                "image_path",
                "text_dim",
                "image_dim",
                "num_queries",
                "context_prior",
                "mediated_image_attention",
            },
        )
        self.optim = _Section(
            self._params,
            {
                "batch_size",
                "lr",
                "num_epochs",
                "lr_decay",
                "amp",
                "grad_clip",
                "weight_decay",
                "save_every_epochs",
                "optimizer",
                "adamwr_t0",
                "adamwr_t_mult",
                "adamwr_eta_min",
            },
        )
        self.loss = _Section(
            self._params,
            {"loss", "pos_weight", "ctx_k", "ctx_loc_token_drop"},
        )

    def print_summary(self, output_path=None):
        print("=" * 72)
        print("SACF-SINR training")
        print(f"  experiment : {self.base.experiment_name}")
        print(f"  device     : {self.base.device}")
        if output_path:
            print(f"  output     : {output_path}")
        print(
            f"  model      : {self.model.model} | input={self.model.input_enc} | "
            f"width={self.model.num_filts} | depth={self.model.depth}"
        )
        print(
            f"  data       : {self.data.species_set} | "
            f"cap={self.data.hard_cap_num_per_class} | workers={self.data.num_workers}"
        )
        print(
            f"  training   : epochs={self.optim.num_epochs} | "
            f"batch={self.optim.batch_size} | lr={self.optim.lr:.2e}"
        )
        print(
            "  fusion     : "
            f"image={self.model.image_path} | "
            f"context_prior={self.model.context_prior} | "
            f"image_mediation={self.model.mediated_image_attention} | "
            f"queries={self.model.num_queries}"
        )
        warm_start = self.model.mm_init_ckpt
        if self.model.model == "ResidualFCNet":
            warm_start = "not used"
        print(f'  warm start : {warm_start if warm_start else "off"}')
        print("=" * 72)

    def to_flat_dict(self):
        return deepcopy(self._params)
