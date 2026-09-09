"""Pretrain the SINR location encoder used to initialise SACF-SINR."""

import torch

import train
from config import ExperimentConfig

cfg = ExperimentConfig()
cfg.base.experiment_name = "SINR-sc-20ep-noeval"
cfg.base.device = "cuda" if torch.cuda.is_available() else "cpu"
cfg.base.bg_type = "spherical"

cfg.model.model = "ResidualFCNet"
cfg.model.input_enc = "sin_cos"
cfg.model.num_filts = 256
cfg.model.depth = 4

cfg.data.species_set = "all_minus_eval"
cfg.data.hard_cap_num_per_class = 1000
cfg.data.num_workers = 4

cfg.optim.num_epochs = 20
cfg.optim.batch_size = 2048
cfg.optim.lr = 0.0005
cfg.optim.lr_decay = 0.98
cfg.optim.amp = True
cfg.optim.optimizer = "adam"
cfg.optim.save_every_epochs = 0

cfg.loss.loss = "an_full"
cfg.loss.pos_weight = 2048

if __name__ == "__main__":
    train.run_pipeline(cfg)
