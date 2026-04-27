from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import pandas as pd


def resolve_data_root(data_dir: str | Path = "data") -> Path:
    """Resolve the dataset root path in a reproducible way."""
    root = Path(data_dir)
    if root.is_absolute():
        return root

    cwd_candidate = Path.cwd() / root
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = Path(__file__).resolve().parents[2] / root
    return repo_candidate


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file with utf-8 fallback and a clear error."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    try:
        return pd.read_csv(csv_path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1", **kwargs)


def load_pickle(path: str | Path) -> Any:
    """Load a pickle object from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pickle file does not exist: {p}")
    with p.open("rb") as fh:
        return pickle.load(fh)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save a pickle object to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump(obj, fh)

