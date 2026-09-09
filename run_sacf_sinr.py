"""Train SACF-SINR and optionally run the few-shot evaluation."""

import argparse
import gc
import os
import subprocess
import sys

import torch

from config import ExperimentConfig


DEFAULT_WARM_START = "experiments/SINR-sc-20ep-noeval/model.pt"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=None, help="Experiment directory name.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--cap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume-ckpt", default="")
    parser.add_argument("--eval-chunk", type=int, default=256)
    parser.add_argument("--mm-init-ckpt", default=DEFAULT_WARM_START)
    parser.add_argument("--metadata-cache", default="data/train/metadata_cache.pt")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument(
        "--image-encoder",
        choices=("adapter", "independent"),
        default="adapter",
        help="Use the proposed adapter or the independent-image-encoder ablation.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        choices=range(1, 6),
        default=3,
        metavar="{1,2,3,4,5}",
        help="Number of field queries and output range fields.",
    )
    parser.add_argument(
        "--no-context-prior",
        action="store_true",
        help="Disable pooled-context conditioning of the field queries.",
    )
    parser.add_argument(
        "--no-image-mediated-attention",
        action="store_true",
        help="Allow field queries to attend directly to image tokens.",
    )
    return parser


def make_config(args):
    if args.name:
        experiment_name = args.name
    else:
        suffixes = []
        if args.image_encoder == "independent":
            suffixes.append("independent-image")
        if args.queries != 3:
            suffixes.append(f"q{args.queries}")
        if args.no_context_prior:
            suffixes.append("no-context-prior")
        if args.no_image_mediated_attention:
            suffixes.append("no-image-mediation")
        experiment_name = "SACF-SINR"
        if suffixes:
            experiment_name += "-" + "-".join(suffixes)

    config = ExperimentConfig()
    config._params.update(
        {
            "experiment_name": experiment_name,
            "model": "SACF-SINR",
            "loss": "sacf_context",
            "num_epochs": args.epochs,
            "hard_cap_num_per_class": args.cap,
            "train_seed": args.seed,
            "device": args.device,
            "save_every_epochs": args.save_every,
            "grad_clip": 1.0,
            "resume_ckpt": args.resume_ckpt,
            "mm_init_ckpt": "" if args.no_warm_start else args.mm_init_ckpt,
            "metadata_cache_path": args.metadata_cache,
            "image_path": args.image_encoder,
            "num_queries": args.queries,
            "context_prior": not args.no_context_prior,
            "mediated_image_attention": not args.no_image_mediated_attention,
            "text_dim": 4096,
            "image_dim": 1024,
        }
    )
    return config


def main():
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --device cpu.")

    config = make_config(args)
    output = os.path.abspath(
        os.path.join(config.base.save_base, config.base.experiment_name)
    )
    checkpoint = os.path.join(output, "model.pt")
    if args.resume_ckpt and os.path.abspath(args.resume_ckpt) == checkpoint:
        raise SystemExit("The resume checkpoint must differ from the output path.")

    if args.skip_train:
        if not os.path.exists(checkpoint):
            raise SystemExit(f"Checkpoint not found: {checkpoint}")
    else:
        import train

        train.run_pipeline(config)
        gc.collect()
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.eval:
        output_json = os.path.join(output, "fewshot_results.json")
        command = [
            sys.executable,
            "evaluate_sacf_sinr.py",
            "--ckpt",
            checkpoint,
            "--out",
            output_json,
            "--seeds",
            "0",
            "--score-chunk",
            str(args.eval_chunk),
            "--device",
            args.device,
            "--metadata-cache",
            args.metadata_cache,
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
