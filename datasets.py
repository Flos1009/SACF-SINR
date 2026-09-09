import os
import numpy as np
import json
import pandas as pd
import torch
import utils


class LocationDataset(torch.utils.data.Dataset):
    def __init__(self, locs, labels, class_to_taxa, input_enc):
        self.input_enc = input_enc
        self.enc = utils.CoordEncoder(input_enc)

        self.locs = locs
        self.loc_feats = self.enc.encode(self.locs)
        self.labels = labels
        self.class_to_taxa = class_to_taxa

        self.num_classes = len(np.unique(labels))
        self.input_dim = self.loc_feats.shape[1]

    def __len__(self):
        return self.loc_feats.shape[0]

    def __getitem__(self, index):
        loc_feat = self.loc_feats[index, :]
        loc = self.locs[index, :]
        class_id = self.labels[index]
        return loc_feat, loc, class_id


def load_inat_data(ip_file, taxa_of_interest=None):
    print(f"  [data] loading {ip_file}", flush=True)
    data = pd.read_csv(ip_file)

    num_obs = data.shape[0]
    data = data[
        (
            (data["latitude"] <= 90)
            & (data["latitude"] >= -90)
            & (data["longitude"] <= 180)
            & (data["longitude"] >= -180)
        )
    ]
    if (num_obs - data.shape[0]) > 0:
        print(
            f"  [data] removed {num_obs - data.shape[0]:,} invalid-location rows",
            flush=True,
        )

    if "accuracy" in data.columns:
        data.drop(["accuracy"], axis=1, inplace=True)

    if "positional_accuracy" in data.columns:
        data.drop(["positional_accuracy"], axis=1, inplace=True)

    if "geoprivacy" in data.columns:
        data.drop(["geoprivacy"], axis=1, inplace=True)

    if "observed_on" in data.columns:
        data.rename(columns={"observed_on": "date"}, inplace=True)

    num_obs_orig = data.shape[0]
    data = data.dropna()
    size_diff = num_obs_orig - data.shape[0]
    if size_diff > 0:
        print(
            f"  [data] removed {size_diff:,} rows with NaN values "
            f"out of {num_obs_orig:,}",
            flush=True,
        )

    if taxa_of_interest is not None:
        num_obs_orig = data.shape[0]
        data = data[data["taxon_id"].isin(taxa_of_interest)]
        print(
            f"  [data] removed {num_obs_orig - data.shape[0]:,} rows from "
            "excluded taxa",
            flush=True,
        )

    print(
        f'  [data] unique classes: ' f'{np.unique(data["taxon_id"].values).shape[0]:,}',
        flush=True,
    )

    locs = np.vstack((data["longitude"].values, data["latitude"].values)).T.astype(
        np.float32
    )
    taxa = data["taxon_id"].values.astype(np.int64)

    return locs, taxa


def get_taxa_of_interest(taxa_file):
    D = np.load(
        os.path.join("data", "eval", "snt", "snt_res_5.npy"), allow_pickle=True
    ).item()
    eval_taxa = set(int(t) for t in D["taxa"])
    with open(os.path.join("data", "eval", "iucn", "iucn_res_5.json")) as f:
        eval_taxa |= set(int(t) for t in json.load(f)["taxa_presence"])
    with open(os.path.join("data", "train", taxa_file), "r") as f:
        metadata = json.load(f)
    taxa = [
        int(row["taxon_id"])
        for row in metadata
        if int(row["taxon_id"]) not in eval_taxa
    ]
    print(
        f"  [data] all_minus_eval: {len(taxa):,} species "
        f"(excluded {len(eval_taxa):,} eval taxa)",
        flush=True,
    )
    return taxa


def get_idx_subsample_observations(labels, hard_cap=-1, hard_cap_seed=123):
    if hard_cap == -1:
        return np.arange(len(labels))
    print(f"  [data] subsampling up to {hard_cap} rows/class", flush=True)
    class_counts = {id: 0 for id in np.unique(labels)}
    ss_rng = np.random.default_rng(hard_cap_seed)
    idx_rand = ss_rng.permutation(len(labels))
    idx_ss = []
    for i in idx_rand:
        class_id = labels[i]
        if class_counts[class_id] < hard_cap:
            idx_ss.append(i)
            class_counts[class_id] += 1
    idx_ss = np.sort(idx_ss)
    print(f"  [data] final training rows: {len(idx_ss):,}", flush=True)
    return idx_ss


def get_train_data(params):
    with open("paths.json", "r") as f:
        paths = json.load(f)
    data_dir = paths["train"]
    obs_file = os.path.join(data_dir, params["obs_file"])
    if params["species_set"] != "all_minus_eval":
        raise ValueError(
            "The production protocol requires species_set='all_minus_eval'."
        )
    taxa_of_interest = get_taxa_of_interest(params["taxa_file"])

    locs, labels = load_inat_data(obs_file, taxa_of_interest)
    unique_taxa, class_ids = np.unique(labels, return_inverse=True)
    class_to_taxa = unique_taxa.tolist()

    idx_ss = get_idx_subsample_observations(
        class_ids, params["hard_cap_num_per_class"], params["hard_cap_seed"]
    )

    locs = torch.from_numpy(np.array(locs)[idx_ss])

    labels = torch.from_numpy(np.array(class_ids)[idx_ss])

    ds = LocationDataset(locs, labels, class_to_taxa, params["input_enc"])

    return ds
