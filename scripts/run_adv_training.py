import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure local src/ modules can be imported when running from the repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

import adversarial_training
import model_utils
import run_experiments


def parse_args():
    parser = argparse.ArgumentParser(description="Run adversarial training with configurable inputs.")

    # Paths
    parser.add_argument("--model_path", type=str, default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/finetune/Llama-3.2-1B-Instruct-tofu", help="Base model path or Hugging Face repo id.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Optional override for model chat template selection.")
    parser.add_argument(
        "--logging_path", type=str, default="./results", help="Directory for trainer logs and outputs."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/data/xiangtao/projects/crossdefense/code/defense/safety/Continuous-AdvTrain/weights/Llama-3.2-1B-Instruct-tofu-AdvTrain",
        help="Directory to store checkpoints. Defaults to logging_path if unset.",
    )
    parser.add_argument(
        "--experiments_path", type=str, default="./experiments", help="Directory for experiment tracking files."
    )
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default=None,
        help="Local path or repo id of a fine-tuned Hugging Face model to load before training.",
    )
    parser.add_argument(
        "--load_checkpoint",
        action="store_true",
        help="Whether to load the provided fine-tuned checkpoint before training.",
    )

    # Dataset
    parser.add_argument("--data_path", type=str, default="./data/", help="Root directory for datasets.")
    parser.add_argument(
        "--utility_data",
        type=str,
        default="ultrachat",
        help="Optional utility dataset name; set to None to disable utility data.",
    )
    parser.add_argument(
        "--probabilities",
        nargs=2,
        type=float,
        default=[0.5, 0.5],
        metavar=("TRAIN_PROB", "UTILITY_PROB"),
        help="Sampling probabilities when mixing adversarial and utility datasets.",
    )
    parser.add_argument(
        "--stopping_strategy",
        type=str,
        default="first_exhausted",
        choices=["first_exhausted", "all_exhausted"],
        help="Stopping strategy when interleaving datasets.",
    )
    parser.add_argument(
        "--diverse_safe_answers",
        action="store_true",
        help="Use the diverse safe answer set for adversarial behaviors.",
    )
    parser.add_argument(
        "--restricted_trainingset_size",
        type=int,
        default=None,
        help="Optional cap on the adversarial training set size.",
    )

    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=-1)

    # Trainer hyperparameters
    parser.add_argument("--trainer_type", type=str, choices=["ul", "dpo"], default="ul")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--padding_side", type=str, choices=["left", "right"], default="left")
    parser.add_argument("--do_online_dpo", action="store_true", help="Enable online DPO training.")
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--dpo_weight", type=float, default=1.0)

    # PEFT/BnB toggles
    parser.add_argument("--disable_peft", action="store_true", help="Disable LoRA PEFT configuration.")
    parser.add_argument("--disable_bnb", action="store_true", help="Disable 4-bit loading (BnB config).")

    # SFT Trainer options
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--packing", action="store_true", help="Enable packing in the SFT trainer.")

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = run_experiments.GlobalConfig()

    # Path configuration
    cfg.path.model_path = args.model_path
    cfg.path.model_name = args.model_name
    cfg.path.logging_path = args.logging_path
    cfg.path.checkpoint_path = args.checkpoint_path or args.logging_path
    cfg.path.experiments_path = args.experiments_path
    cfg.path.load_checkpoint_path = args.load_checkpoint_path
    cfg.path.load_checkpoint = args.load_checkpoint and args.load_checkpoint_path is not None

    # Dataset configuration
    cfg.dataset.data_path = args.data_path
    cfg.dataset.utility_data = None if args.utility_data == "None" else args.utility_data
    cfg.dataset.probabilities = args.probabilities
    cfg.dataset.stopping_strategy = args.stopping_strategy
    cfg.dataset.diverse_safe_answers = args.diverse_safe_answers
    cfg.dataset.restricted_trainingset_size = args.restricted_trainingset_size

    # Training configuration
    cfg.training.num_train_epochs = args.num_train_epochs
    cfg.training.per_device_train_batch_size = args.per_device_train_batch_size
    cfg.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    cfg.training.learning_rate = args.learning_rate
    cfg.training.weight_decay = args.weight_decay
    cfg.training.max_grad_norm = args.max_grad_norm
    cfg.training.logging_steps = args.logging_steps
    cfg.training.save_steps = args.save_steps
    cfg.training.max_steps = args.max_steps

    # Trainer hyperparameters
    cfg.trainer_hparams.trainer_type = args.trainer_type
    cfg.trainer_hparams.dtype = args.dtype
    cfg.trainer_hparams.padding_side = args.padding_side
    cfg.trainer_hparams.do_online_dpo = args.do_online_dpo
    cfg.trainer_hparams.dpo_beta = args.dpo_beta
    cfg.trainer_hparams.dpo_weight = args.dpo_weight

    # SFT Trainer config
    cfg.sfttrainer.max_seq_length = args.max_seq_length
    cfg.sfttrainer.packing = args.packing

    # Optional feature toggles
    cfg.peft = None if args.disable_peft else cfg.peft
    cfg.bnb = None if args.disable_bnb else cfg.bnb

    os.makedirs(cfg.path.logging_path, exist_ok=True)
    os.makedirs(cfg.path.checkpoint_path, exist_ok=True)

    model_name = cfg.path.model_name or model_utils.get_model_name(cfg.path.model_path)

    adversarial_training.adversarial_training_loop(
        model_name,
        asdict(cfg.path),
        asdict(cfg.adversarial),
        asdict(cfg.dataset),
        asdict(cfg.training),
        asdict(cfg.peft) if cfg.peft is not None else None,
        asdict(cfg.bnb) if cfg.bnb is not None else None,
        asdict(cfg.sfttrainer),
        asdict(cfg.trainer_hparams),
    )


if __name__ == "__main__":
    main()
