from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pmtl_disco.pipeline import run_experiment
from pmtl_disco.utils import load_yaml_config, merge_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PMTL-DisCo reproduction experiment.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=["politifact", "gossipcop", "fang"])
    parser.add_argument("--few-shot", type=int, dest="few_shot")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=None)
    parser.add_argument("--max-len", type=int, dest="max_len", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--K", type=float, default=None)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--alpha1", type=float, default=None)
    parser.add_argument("--alpha2", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-reasoning", action="store_true")
    return parser.parse_args()


def build_runtime_config(base_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    override: Dict[str, Any] = {
        "experiment": {},
        "data": {},
        "train": {},
        "inference": {},
        "output": {},
    }

    if args.dataset is not None:
        override["experiment"]["dataset"] = args.dataset
    if args.few_shot is not None:
        override["experiment"]["few_shot"] = args.few_shot
    if args.iters is not None:
        override["experiment"]["iterations"] = args.iters
    if args.seed is not None:
        override["train"]["seed"] = args.seed
    if args.epochs is not None:
        override["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        override["train"]["batch_size"] = args.batch_size
    if args.max_len is not None:
        override["train"]["max_length"] = args.max_len
    if args.lr is not None:
        override["train"]["lr"] = args.lr
    if args.alpha1 is not None:
        override["train"]["alpha1"] = args.alpha1
    if args.alpha2 is not None:
        override["train"]["alpha2"] = args.alpha2
    if args.gamma is not None:
        override["train"]["gamma"] = args.gamma

    if args.K is not None:
        override["inference"]["k_threshold"] = args.K
    if args.mu is not None:
        override["inference"]["mu"] = args.mu
    if args.no_reasoning:
        override["inference"]["use_reasoning"] = False

    if args.data_dir is not None:
        override["data"]["data_dir"] = args.data_dir
    if args.output_dir is not None:
        override["output"]["root_dir"] = args.output_dir

    return merge_config(base_cfg, override)


def main() -> None:
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    config = load_yaml_config(cfg_path)
    config = build_runtime_config(config, args)

    set_seed(int(config["train"]["seed"]))

    summary = run_experiment(
        dataset=config["experiment"]["dataset"],
        n_samples=int(config["experiment"]["few_shot"]),
        iterations=int(config["experiment"]["iterations"]),
        config=config,
    )

    output_root = PROJECT_ROOT / config["output"]["root_dir"]
    out_dir = output_root / summary.dataset / f"fewshot_{summary.n_samples}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"summary_seed{config['train']['seed']}_iter{summary.iterations}.json"
    out_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    print(f"Saved summary to: {out_path}")


if __name__ == "__main__":
    main()

