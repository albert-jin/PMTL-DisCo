from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from pmtl_disco.training import PMTLDisCoTrainer, TrainerConfig


@dataclass
class ExperimentSummary:
    dataset: str
    n_samples: int
    iterations: int
    accuracy_mean: float
    precision_mean: float
    recall_mean: float
    f1_mean: float
    accuracy_all: List[float]


def _pick_callable(module: Any, names: Sequence[str]) -> Callable[..., Any]:
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    available = [x for x in dir(module) if not x.startswith("_")]
    raise AttributeError(
        f"None of expected callables {list(names)} exists in {module.__name__}. "
        f"Available: {available}"
    )


def _build_model(model_cfg: Dict[str, Any]):
    from pmtl_disco import modeling

    model_cls = None
    for candidate in ["PMTLDisCoModel", "PromptMultiTaskModel", "DisCoModel"]:
        obj = getattr(modeling, candidate, None)
        if obj is not None:
            model_cls = obj
            break
    if model_cls is None:
        raise AttributeError(
            "No model class found in pmtl_disco.modeling (expected PMTLDisCoModel)."
        )
    return model_cls(**model_cfg)


def run_experiment(
    dataset: str,
    n_samples: int,
    iterations: int,
    config: Dict[str, Any],
) -> ExperimentSummary:
    import pmtl_disco.data as data_io

    split_loader = _pick_callable(
        data_io,
        [
            "load_fewshot_split",
            "load_fewshot_splits",
            "get_fewshot_splits",
            "get_splits_fewshot",
        ],
    )
    graph_loader = _pick_callable(
        data_io,
        [
            "load_train_conf_and_adj",
            "load_graph_data",
            "load_or_build_graph_data",
            "load_adjacency_and_train_conf",
        ],
    )

    try:
        split = split_loader(dataset=dataset, n_samples=n_samples, data_dir=config["data"]["data_dir"])
    except TypeError:
        split = split_loader(obj=dataset, n_samples=n_samples, data_dir=config["data"]["data_dir"])
    if hasattr(split, "x_train"):
        train_texts, test_texts, train_labels, test_labels = (
            split.x_train,
            split.x_test,
            split.y_train,
            split.y_test,
        )
    else:
        train_texts, test_texts, train_labels, test_labels = split

    try:
        graph_pack = graph_loader(
            dataset=dataset,
            n_samples=n_samples,
            user_threshold=int(config["data"]["user_threshold"]),
            data_dir=config["data"]["data_dir"],
            prefer_preprocessed=bool(config["data"].get("prefer_preprocessed_graph", True)),
        )
    except TypeError:
        graph_pack = graph_loader(
            dataset=dataset,
            n_samples=n_samples,
            user_threshold=int(config["data"]["user_threshold"]),
            data_dir=config["data"]["data_dir"],
        )

    if isinstance(graph_pack, dict):
        adjacency = graph_pack.get("adjacency") or graph_pack.get("adj")
    elif hasattr(graph_pack, "adj"):
        adjacency = graph_pack.adj
    else:
        # expected tuple: (train_conf, adjacency, ...)
        adjacency = graph_pack[1]
    if adjacency is None:
        raise ValueError("Adjacency matrix is missing from loaded graph data.")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["plm_name"])
    trainer_cfg = TrainerConfig(
        lr=float(config["train"]["lr"]),
        batch_size=int(config["train"]["batch_size"]),
        max_length=int(config["train"]["max_length"]),
        epochs=int(config["train"]["epochs"]),
        alpha1=float(config["train"]["alpha1"]),
        alpha2=float(config["train"]["alpha2"]),
        gamma=float(config["train"]["gamma"]),
        k_threshold=float(config["inference"]["k_threshold"]),
        mu=float(config["inference"]["mu"]),
        prompt_template=str(config["model"]["prompt_template"]),
        use_reasoning=bool(config["inference"]["use_reasoning"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics: List[Tuple[float, float, float, float]] = []
    for it in range(iterations):
        model_cfg = {
            "model_name_or_path": config["model"]["plm_name"],
            "tokenizer": tokenizer,
            "prompt_template": config["model"]["prompt_template"],
            "real_label_words": config["model"].get("real_label_words"),
            "fake_label_words": config["model"].get("fake_label_words"),
            "verbalizer_hidden_dim": int(config["model"]["verbalizer_hidden_dim"]),
            "verbalizer_dropout": float(config["model"]["verbalizer_dropout"]),
            "alpha1": float(config["train"]["alpha1"]),
            "alpha2": float(config["train"]["alpha2"]),
            "gamma": float(config["train"]["gamma"]),
        }
        model = _build_model(model_cfg)
        trainer = PMTLDisCoTrainer(
            model=model,
            tokenizer=tokenizer,
            config=trainer_cfg,
            device=device,
        )
        result = trainer.train_one_iteration(
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=test_texts,
            test_labels=test_labels,
            adjacency=adjacency,
            seed=int(config["train"]["seed"]) + it,
        )
        metrics.append((result.accuracy, result.precision, result.recall, result.f1))

    arr = np.asarray(metrics, dtype=np.float64)
    return ExperimentSummary(
        dataset=dataset,
        n_samples=n_samples,
        iterations=iterations,
        accuracy_mean=float(arr[:, 0].mean()),
        precision_mean=float(arr[:, 1].mean()),
        recall_mean=float(arr[:, 2].mean()),
        f1_mean=float(arr[:, 3].mean()),
        accuracy_all=arr[:, 0].tolist(),
    )
