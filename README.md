# SACF-SINR

Code accompanying the manuscript **SACF-SINR: A framework for few-shot species distribution estimation based on semantic adaptation and context-guided fusion**.

SACF-SINR extends spatial implicit neural representations (SINR) with task-oriented image-to-text adaptation and controlled multimodal context fusion. The released implementation matches the main experimental model and does not include a post-encoding shared representation encoder. It includes:

- a SINR location encoder for spatial context;
- a trainable text encoder for species metadata;
- an image-to-text adapter followed by a read-only text-encoder path;
- a multimodal context Transformer with pooled context conditioning;
- multiple field queries for distribution readout; and
- image-mediated attention that limits direct image-to-query information flow when other context is available.

## Repository contents

The Python modules implement data loading, model components, training objectives, training, and few-shot evaluation.

The main files are:

- `sacf_model.py`: modality encoders and the SACF-SINR model;
- `context_transformer.py`: context-prior conditioning, image-mediated attention, and field decoding;
- `losses.py`: training objectives;
- `prepare_metadata_cache.py`: creation of the training metadata cache from the released text and image features;
- `run_sinr_pretrain.py`: SINR location-encoder pretraining entry point;
- `run_sacf_sinr.py`: training entry point;
- `evaluation.py`: IUCN and S&T evaluation implementation; and
- `evaluate_sacf_sinr.py`: public evaluation entry point.

No datasets, feature caches, model checkpoints, experiment outputs, or paper figures are included in this repository. Downloaded data should be placed under `data/` as described below. The training-data root can be changed in `paths.json`.

## Installation

Python 3.10 or newer is recommended. Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Data and expected layout

The experiments require the occurrence records, species text features, image features, and evaluation benchmarks described in the manuscript. See [`data/README.md`](data/README.md) for the external download links and the expected file layout. The training-data root is defined in [`paths.json`](paths.json).

After arranging the downloaded files, build the compact training metadata cache:

```bash
python prepare_metadata_cache.py
```

## Training and evaluation

Run commands from the repository root. First pretrain the SINR location encoder used to initialise SACF-SINR:

```bash
python run_sinr_pretrain.py
```

The checkpoint is written to `experiments/SINR-sc-20ep-noeval/model.pt`. Then train SACF-SINR:

```bash
python run_sacf_sinr.py --device cuda
```

To run a few-shot evaluation from a trained checkpoint:

```bash
python evaluate_sacf_sinr.py --ckpt path/to/model.pt --device cuda
```

The command-line defaults reproduce the main configuration used in the manuscript. A different SINR checkpoint can be supplied with `--mm-init-ckpt`; `--no-warm-start` disables this initialisation. The component ablations can be launched explicitly:

```bash
# Independent image encoder
python run_sacf_sinr.py --image-encoder independent

# One field query
python run_sacf_sinr.py --queries 1

# No pooled context prior
python run_sacf_sinr.py --no-context-prior

# No image-mediated attention
python run_sacf_sinr.py --no-image-mediated-attention
```

To save AP values for individual species, add `--save-per-species` to the evaluation command. To evaluate a particular candidate image, add `--image-index N`, where `N` is the zero-based row after sorting each species' images by filename. Species with fewer than `N + 1` images are excluded from that run.

## Reproducibility

The main configuration uses three learnable field queries, four Transformer layers, two attention heads, model dimension 256, feed-forward dimension 512, AdamW with cosine warm restarts, and the image-mediated attention mask described in the manuscript. Random seeds and output locations are controlled by the command-line launchers.

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE).
