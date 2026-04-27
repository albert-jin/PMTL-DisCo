from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .io import read_csv, resolve_data_root


@dataclass(frozen=True)
class FewShotSplit:
    x_train: list[str]
    x_test: list[str]
    y_train: list[int]
    y_test: list[int]


def _coerce_labels(labels: Sequence[object], source: Path) -> list[int]:
    series = pd.Series(labels)
    try:
        return series.astype(int).tolist()
    except ValueError as exc:
        raise ValueError(f"Non-integer labels found in {source}") from exc


def _read_fewshot_csv(path: Path) -> tuple[list[str], list[int]]:
    # Prompt-and-Align split files are two-column CSVs: label,text without header.
    frame = read_csv(path, header=None, names=["label", "text"], keep_default_na=False)
    texts = frame["text"].fillna("").astype(str).tolist()
    labels = _coerce_labels(frame["label"].tolist(), path)
    return texts, labels


def load_fewshot_split(
    dataset: str,
    n_samples: int,
    data_dir: str | Path = "data",
) -> FewShotSplit:
    """Load few-shot train/test texts and labels from Prompt-and-Align CSVs."""
    data_root = resolve_data_root(data_dir)
    train_path = data_root / f"{dataset}_train_{n_samples}.csv"
    test_path = data_root / f"{dataset}_test.csv"

    x_train, y_train = _read_fewshot_csv(train_path)
    x_test, y_test = _read_fewshot_csv(test_path)

    return FewShotSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )


def get_splits_fewshot(
    obj: str,
    n_samples: int,
    data_dir: str | Path = "data",
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Compatibility wrapper matching Prompt-and-Align's original API."""
    split = load_fewshot_split(dataset=obj, n_samples=n_samples, data_dir=data_dir)
    return split.x_train, split.x_test, split.y_train, split.y_test

