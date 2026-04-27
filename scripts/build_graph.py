from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _read_news_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    if "news_id" not in df.columns:
        raise ValueError(f"Missing `news_id` column in: {path}")
    if "label" not in df.columns:
        raise ValueError(f"Missing `label` column in: {path}")
    return df


def _read_social_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    required_cols = {"sid", "uid"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in: {path}")
    return df


def _build_train_conf(news_df: pd.DataFrame, n_samples: int) -> np.ndarray:
    labels = news_df.iloc[:n_samples]["label"].astype(int).tolist()
    conf = np.zeros((n_samples, 2), dtype=np.float32)
    for idx, label in enumerate(labels):
        if label not in (0, 1):
            raise ValueError(f"Unexpected label {label} at train row {idx}. Expected 0/1.")
        conf[idx, label] = 1.0
    return conf


def _normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
    rowsum = np.array(adj.sum(1)).flatten()
    d_row = np.power(rowsum, -0.5, where=rowsum != 0)
    d_row[np.isinf(d_row)] = 0.0
    d_row[np.isnan(d_row)] = 0.0
    d_row_mat = sp.diags(d_row)
    return d_row_mat.dot(adj).dot(d_row_mat)


def _build_adjacency(news_ids: list[str], social_df: pd.DataFrame, threshold_user: int) -> sp.csr_matrix:
    social_df = social_df[social_df["sid"].isin(news_ids)].copy()
    social_df["sid"] = social_df["sid"].astype(str)
    social_df["uid"] = social_df["uid"].astype(str)

    user_counts = Counter(social_df["uid"].tolist())
    active_users = {u for u, c in user_counts.items() if c >= threshold_user}

    news_to_idx = {sid: idx for idx, sid in enumerate(news_ids)}
    active_user_to_idx = {uid: idx for idx, uid in enumerate(sorted(active_users))}

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    active_news = set()
    for _, row in social_df.iterrows():
        uid = row["uid"]
        sid = row["sid"]
        if uid not in active_user_to_idx:
            continue
        rows.append(active_user_to_idx[uid])
        cols.append(news_to_idx[sid])
        data.append(1.0)
        active_news.add(sid)

    cold_news = [sid for sid in news_ids if sid not in active_news]
    offset = len(active_user_to_idx)
    for i, sid in enumerate(cold_news):
        rows.append(offset + i)
        cols.append(news_to_idx[sid])
        data.append(1.0)

    num_rows = len(active_user_to_idx) + len(cold_news)
    num_cols = len(news_ids)
    if num_rows == 0 or num_cols == 0:
        raise ValueError("Empty graph encountered during construction.")

    engagement = sp.csr_matrix((data, (rows, cols)), shape=(num_rows, num_cols), dtype=np.float32)
    adjacency = engagement.T.dot(engagement).astype(np.float32)
    adjacency = _normalize_adj(adjacency).tocsr()
    return adjacency


def build_graph_for_setting(
    dataset: str,
    n_samples: int,
    threshold_user: int,
    data_root: Path,
    output_dir: Path,
    social_template: str,
    news_template: str,
) -> Path:
    social_path = data_root / social_template.format(dataset=dataset, few_shot=n_samples)
    news_path = data_root / news_template.format(dataset=dataset, few_shot=n_samples)
    if not social_path.exists():
        raise FileNotFoundError(f"Social context file not found: {social_path}")
    if not news_path.exists():
        raise FileNotFoundError(f"News raw file not found: {news_path}")

    news_df = _read_news_df(news_path)
    social_df = _read_social_df(social_path)

    news_ids = news_df["news_id"].astype(str).tolist()
    train_conf = _build_train_conf(news_df, n_samples)
    adj = _build_adjacency(news_ids, social_df, threshold_user)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_nn_relations_{n_samples}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump([train_conf, adj], f)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PMTL-DisCo news graph cache from raw social context.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["politifact", "gossipcop", "fang"],
        choices=["politifact", "gossipcop", "fang"],
        help="One or more datasets.",
    )
    parser.add_argument(
        "--few-shot",
        nargs="+",
        type=int,
        default=[16, 32, 128],
        help="One or more few-shot settings.",
    )
    parser.add_argument("--threshold-user", type=int, default=5, help="Active-user threshold t_u.")
    parser.add_argument("--data-root", default="data", help="Data root directory.")
    parser.add_argument("--social-template", default="social_context_raw/{dataset}_socialcontext_train{few_shot}.csv")
    parser.add_argument("--news-template", default="news_articles_raw/{dataset}_full_train{few_shot}.csv")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for pkl cache. Default: data/adjs/user_t{threshold-user}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = _resolve_path(args.data_root)
    output_dir = (
        _resolve_path(args.output_dir)
        if args.output_dir is not None
        else _resolve_path(f"data/adjs/user_t{args.threshold_user}")
    )

    for dataset in args.datasets:
        for n_samples in args.few_shot:
            out_path = build_graph_for_setting(
                dataset=dataset,
                n_samples=n_samples,
                threshold_user=args.threshold_user,
                data_root=data_root,
                output_dir=output_dir,
                social_template=args.social_template,
                news_template=args.news_template,
            )
            print(f"[OK] {dataset} few-shot={n_samples}: {out_path}")


if __name__ == "__main__":
    main()
