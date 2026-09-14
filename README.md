# SACF-SINR

Implementation accompanying the manuscript **SACF-SINR: A framework for few-shot species distribution estimation based on semantic adaptation and context-guided fusion**.

## Overview

SACF-SINR estimates the geographic distribution of an unseen species from a small set of occurrence locations and any available species metadata. The model combines spatial observations, text and images through task-oriented semantic adaptation and context-guided fusion.

This repository contains the model configuration used in the main experiments. Encoded modality tokens are passed directly to the Multimodal Context Transformer, without a post-encoding shared representation encoder. The main components are:

- a SINR location encoder for spatial observations;
- a trainable text encoder for species metadata;
- an image-to-text adapter with a read-only path through the text encoder;
- pooled-context conditioning of the field queries;
- multiple field queries for species range representation; and
- image-mediated attention for controlling image-to-query information flow.

## Repository structure

- `sacf_model.py`: modality encoders and the SACF-SINR model.
- `context_transformer.py`: context-prior conditioning, image-mediated attention and field decoding.
- `models.py`: the SINR location encoder and model construction.
- `losses.py`: the SINR pretraining and SACF-SINR training objectives.
- `datasets.py`: occurrence-data loading and training-set construction.
- `prepare_metadata_cache.py`: preparation of the training metadata cache.
- `run_sinr_pretrain.py`: SINR location-encoder pretraining.
- `run_sacf_sinr.py`: SACF-SINR training and ablation entry point.
- `evaluate_sacf_sinr.py`: evaluation entry point.
- `evaluation.py`: implementation of the IUCN and S&T evaluation protocols.

## Installation

Python 3.10 or later is recommended. Install the required packages from the repository root:

```bash
pip install -r requirements.txt
```

## Data preparation

Data, pretrained features and model checkpoints are not included in this repository. Download links and the complete directory structure are provided in [`data/README.md`](data/README.md).

After placing the downloaded files in the specified directories, build the metadata cache used during training:

```bash
python prepare_metadata_cache.py
```

The default training-data directory is defined in [`paths.json`](paths.json).

## Training

Run all commands from the repository root. To reproduce the main training protocol, first pretrain the SINR location encoder:

```bash
python run_sinr_pretrain.py
```

The pretrained checkpoint is saved as `experiments/SINR-sc-20ep-noeval/model.pt`. SACF-SINR can then be trained with:

```bash
python run_sacf_sinr.py --device cuda
```

Use `--mm-init-ckpt` to provide a different SINR checkpoint. Use `--no-warm-start` to train without SINR initialisation.

## Evaluation

Evaluate a trained checkpoint with:

```bash
python evaluate_sacf_sinr.py --ckpt path/to/model.pt --device cuda
```

By default, the script evaluates the configured zero-shot and few-shot settings on both IUCN and S&T. Results are written to `fewshot_results.json` beside the checkpoint unless another path is supplied with `--out`.

To include the AP of each evaluated species in the output file, run:

```bash
python evaluate_sacf_sinr.py --ckpt path/to/model.pt --save-per-species
```

To evaluate a specific candidate image for each species, provide its zero-based index. For example, the following command selects index 0 after sorting each species' images by filename:

```bash
python evaluate_sacf_sinr.py --ckpt path/to/model.pt --image-index 0
```

When a larger index is requested, species without an image at that index are excluded from that evaluation run.

## Component ablations

The component ablations described in the manuscript can be launched as follows:

```bash
# Replace the image-to-text adaptation path with an independent image encoder
python run_sacf_sinr.py --image-encoder independent

# Use a single field query
python run_sacf_sinr.py --queries 1

# Remove the pooled context prior
python run_sacf_sinr.py --no-context-prior

# Remove image-mediated attention
python run_sacf_sinr.py --no-image-mediated-attention
```

## Reproducibility

The default configuration uses three field queries, four Transformer layers, two attention heads, a model dimension of 256 and a feed-forward dimension of 512. Training uses AdamW with cosine warm restarts. Random seeds, output directories and component-ablation options can be set through the training command.

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22744053.svg)](https://doi.org/10.5281/zenodo.22744053)
