# External data

Data are intentionally not included in this code release. Download the source releases below and arrange the required files according to the complete layout shown here.

- Image features supplied with FS-SINR: [Google Drive](https://drive.google.com/file/d/1qb_frR3TWD-o4xexMtpbvwOenBssR238/view)
- Training text features supplied with LE-SINR: [Google Drive](https://drive.google.com/file/d/1xv47k6dqex0z33NDQrVgLUTtz6_vjr6i/view)
- Evaluation text features supplied with LE-SINR: [Google Drive](https://drive.google.com/file/d/1EaMJwfcpmDzGiPZm3464tYBtmfsrWUg0/view)
- Occurrence records and evaluation data supplied with SINR: [CaltechDATA](https://data.caltech.edu/records/b0wyb-tat89/files/data.zip)

The default commands expect this directory structure:

```text
data/
├── README.md
├── image_features.pt
├── train/
│   ├── geo_prior_train.csv
│   ├── geo_prior_train_meta.json
│   ├── wiki_data.pt
│   └── metadata_cache.pt          # generated locally
└── eval/
    ├── gpt_data.pt
    ├── positive_eval_data.npz
    ├── iucn/
    │   └── iucn_res_5.json
    └── snt/
        └── snt_res_5.npy
```

Place the training text release at `data/train/wiki_data.pt`, the evaluation text release at `data/eval/gpt_data.pt`, and the image release at `data/image_features.pt`. Extract the required occurrence and benchmark files from the SINR archive into the paths shown above. Files with other names in the source archives are not used by this implementation.

Finally, create the compact training cache from the downloaded text and image features:

```bash
python prepare_metadata_cache.py
```

This command writes `data/train/metadata_cache.pt`. It preserves the training-time text-section ordering, L2-normalises each raw image feature, and stores the features by taxon for efficient sampling. Downloaded data, generated caches, checkpoints, and outputs should not be committed to the code repository.
