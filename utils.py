import torch
import numpy as np
import math


class CoordEncoder:
    def __init__(self, input_enc="sin_cos"):
        if input_enc != "sin_cos":
            raise ValueError("SACF-SINR uses sine-cosine coordinate encoding.")
        self.input_enc = input_enc

    def encode(self, locs, normalize=True):
        if isinstance(locs, np.ndarray):
            locs = torch.from_numpy(locs).float()
        if normalize:
            locs = locs.clone()
            locs = normalize_coords(locs)
        return encode_loc(locs)


def normalize_coords(locs):
    locs[:, 0] /= 180.0
    locs[:, 1] /= 90.0

    return locs


def encode_loc(loc_ip, concat_dim=1):
    if isinstance(loc_ip, np.ndarray):
        loc_ip = torch.from_numpy(loc_ip).float()
    feats = torch.cat(
        (torch.sin(math.pi * loc_ip), torch.cos(math.pi * loc_ip)), concat_dim
    )
    return feats


def rand_samples(batch_size, device):
    rand_loc = torch.rand(batch_size, 2).to(device)
    theta1 = 2.0 * math.pi * rand_loc[:, 0]
    theta2 = torch.acos(2.0 * rand_loc[:, 1] - 1.0)
    lat = 1.0 - 2.0 * theta2 / math.pi
    lon = theta1 / math.pi - 1.0
    return torch.stack((lon, lat), dim=1)


def sample_bg_locations(batch_size, params):
    if params.get("bg_type") != "spherical":
        raise ValueError("Only spherical background sampling is supported.")
    return rand_samples(batch_size, params["device"])


def load_keyed_embeddings(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    ids = d["taxon_id"].tolist()
    out = {}
    for p, (idx, k) in enumerate(d["keys"]):
        out.setdefault(int(ids[idx]), {})[k] = d["data"][p].float()
    return out


def load_image_embedding_rows(path="data/image_features.pt"):
    d = torch.load(path, map_location="cpu", weights_only=False)
    ids = d["taxon_id"].tolist()
    rows = {}
    for p, (idx, fname) in enumerate(d["keys"]):
        rows.setdefault(int(ids[idx]), []).append((str(fname), int(p)))
    out = {}
    for taxon, entries in rows.items():
        entries.sort(key=lambda item: item[0])
        out[taxon] = [(fname, d["data"][row]) for fname, row in entries]
    return out


def load_first_image_embeddings(path="data/image_features.pt"):
    rows = load_image_embedding_rows(path)
    return {
        taxon: torch.nn.functional.normalize(entries[0][1].float(), dim=-1)
        for taxon, entries in rows.items()
    }


def tqdm_disabled():
    import sys
    import os

    if os.environ.get("FORCE_TQDM", "0") == "1":
        return False
    return not sys.stderr.isatty()
