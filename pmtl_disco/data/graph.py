from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .io import load_pickle, read_csv, resolve_data_root


NodeRef = int | str


@dataclass(slots=True)
class NewsGraph:
    """Graph container with neighbor retrieval for inference-time augmentation."""

    train_conf: np.ndarray
    adj: sp.csr_matrix
    node_ids: list[str]
    node_to_index: dict[str, int]

    def _resolve_index(self, node: NodeRef) -> int:
        if isinstance(node, int):
            if node < 0 or node >= len(self.node_ids):
                raise IndexError(f"Node index out of range: {node}")
            return node
        try:
            return self.node_to_index[node]
        except KeyError as exc:
            raise KeyError(f"Unknown node id: {node}") from exc

    def get_node_index(self, node_id: str) -> int:
        return self._resolve_index(node_id)

    def get_node_id(self, index: int) -> str:
        return self.node_ids[self._resolve_index(index)]

    def get_adjacency(
        self,
        node: NodeRef,
        return_ids: bool = True,
        include_self: bool = False,
    ) -> dict[NodeRef, float]:
        """Return non-zero normalized adjacency weights for one node."""
        idx = self._resolve_index(node)
        row = self.adj.getrow(idx)
        neighbors: dict[NodeRef, float] = {}

        for nbr_idx, weight in zip(row.indices.tolist(), row.data.tolist()):
            if not include_self and nbr_idx == idx:
                continue
            key: NodeRef = self.node_ids[nbr_idx] if return_ids else nbr_idx
            neighbors[key] = float(weight)

        return neighbors

    def first_hop_neighbors(
        self,
        node: NodeRef,
        return_ids: bool = True,
        include_self: bool = False,
    ) -> list[NodeRef]:
        idx = self._resolve_index(node)
        neighbors = set(self.adj.getrow(idx).indices.tolist())
        if not include_self and idx in neighbors:
            neighbors.remove(idx)
        sorted_neighbors = sorted(neighbors)
        if return_ids:
            return [self.node_ids[i] for i in sorted_neighbors]
        return sorted_neighbors

    def second_hop_neighbors(
        self,
        node: NodeRef,
        return_ids: bool = True,
        include_self: bool = False,
        include_first_hop: bool = False,
    ) -> list[NodeRef]:
        idx = self._resolve_index(node)
        first_hop = set(self.adj.getrow(idx).indices.tolist())
        first_hop.discard(idx)

        second_hop: set[int] = set()
        for nbr in first_hop:
            second_hop.update(self.adj.getrow(nbr).indices.tolist())

        if not include_first_hop:
            second_hop.difference_update(first_hop)
        if not include_self:
            second_hop.discard(idx)

        sorted_neighbors = sorted(second_hop)
        if return_ids:
            return [self.node_ids[i] for i in sorted_neighbors]
        return sorted_neighbors


def _raw_news_path(data_root: Path, dataset: str, n_samples: int) -> Path:
    return data_root / "news_articles_raw" / f"{dataset}_full_train{n_samples}.csv"


def _raw_social_path(data_root: Path, dataset: str, n_samples: int) -> Path:
    return data_root / "social_context_raw" / f"{dataset}_socialcontext_train{n_samples}.csv"


def _preprocessed_adj_path(
    data_root: Path,
    dataset: str,
    n_samples: int,
    user_threshold: int,
) -> Path:
    return (
        data_root
        / "adjs"
        / f"user_t{user_threshold}"
        / f"{dataset}_nn_relations_{n_samples}.pkl"
    )


def _normalize_adj(adj: sp.spmatrix) -> sp.csr_matrix:
    adj_csr = adj.tocsr().astype(np.float32)
    degrees = np.asarray(adj_csr.sum(axis=1)).reshape(-1)
    with np.errstate(divide="ignore"):
        deg_inv_sqrt = np.power(degrees, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    d_inv_sqrt = sp.diags(deg_inv_sqrt, dtype=np.float32)
    normalized = d_inv_sqrt @ adj_csr @ d_inv_sqrt
    return normalized.tocsr()


def _build_train_conf(labels: Iterable[object], n_samples: int) -> np.ndarray:
    labels_series = pd.Series(list(labels)[:n_samples])
    if len(labels_series) != n_samples:
        raise ValueError(
            f"Requested {n_samples} training samples, but only {len(labels_series)} labels found."
        )

    try:
        y = labels_series.astype(int).to_numpy()
    except ValueError as exc:
        raise ValueError("Training labels must be integer-like.") from exc

    unique_labels = sorted(set(y.tolist()))
    if len(unique_labels) > 2:
        raise ValueError(f"Expected binary labels, got {unique_labels}")

    if set(unique_labels).issubset({0, 1}):
        y01 = y
    else:
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y01 = np.array([label_to_idx[label] for label in y], dtype=np.int64)

    train_conf = np.zeros((n_samples, 2), dtype=np.float32)
    train_conf[np.arange(n_samples), y01] = 1.0
    return train_conf


def _load_node_ids_from_raw(
    dataset: str,
    n_samples: int,
    data_root: Path,
) -> list[str]:
    news_path = _raw_news_path(data_root, dataset, n_samples)
    news_df = read_csv(news_path)
    if "news_id" not in news_df.columns:
        raise ValueError(f"'news_id' column missing in {news_path}")
    return news_df["news_id"].astype(str).tolist()


def build_graph_from_raw(
    dataset: str,
    n_samples: int,
    user_threshold: int = 5,
    data_dir: str | Path = "data",
    ensure_isolated_self_loop: bool = True,
) -> NewsGraph:
    """
    Build user-news matrix B from raw CSVs, then A=B^T B and D^{-1/2} A D^{-1/2}.
    """
    data_root = resolve_data_root(data_dir)
    news_path = _raw_news_path(data_root, dataset, n_samples)
    social_path = _raw_social_path(data_root, dataset, n_samples)

    news_df = read_csv(news_path)
    social_df = read_csv(social_path)

    required_news_cols = {"news_id", "label"}
    required_social_cols = {"sid", "uid"}
    if not required_news_cols.issubset(news_df.columns):
        raise ValueError(f"{news_path} must contain columns: {required_news_cols}")
    if not required_social_cols.issubset(social_df.columns):
        raise ValueError(f"{social_path} must contain columns: {required_social_cols}")

    node_ids = news_df["news_id"].astype(str).tolist()
    node_to_index = {nid: idx for idx, nid in enumerate(node_ids)}
    news_id_set = set(node_to_index.keys())

    train_conf = _build_train_conf(news_df["label"], n_samples=n_samples)

    social_view = social_df.copy()
    social_view["sid"] = social_view["sid"].astype(str)
    social_view["uid"] = social_view["uid"].astype(str)
    social_view = social_view[social_view["sid"].isin(news_id_set)]

    if social_view.empty:
        adj = sp.eye(len(node_ids), dtype=np.float32, format="csr")
        return NewsGraph(
            train_conf=train_conf,
            adj=adj,
            node_ids=node_ids,
            node_to_index=node_to_index,
        )

    uid_counts = social_view["uid"].value_counts()
    active_uids = uid_counts[uid_counts >= user_threshold].index.tolist()
    active_df = social_view[social_view["uid"].isin(active_uids)]

    user_to_row = {uid: idx for idx, uid in enumerate(active_uids)}
    row_idx: list[int] = []
    col_idx: list[int] = []

    if not active_df.empty:
        row_idx = [user_to_row[uid] for uid in active_df["uid"].tolist()]
        col_idx = [node_to_index[sid] for sid in active_df["sid"].tolist()]

    active_news_ids = set(active_df["sid"].tolist())
    if ensure_isolated_self_loop:
        missing_news_ids = [nid for nid in node_ids if nid not in active_news_ids]
    else:
        missing_news_ids = []

    n_active_users = len(active_uids)
    for offset, news_id in enumerate(missing_news_ids):
        row_idx.append(n_active_users + offset)
        col_idx.append(node_to_index[news_id])

    total_users = n_active_users + len(missing_news_ids)
    if total_users == 0:
        b_matrix = sp.csr_matrix((0, len(node_ids)), dtype=np.float32)
    else:
        values = np.ones(len(row_idx), dtype=np.float32)
        b_matrix = sp.csr_matrix(
            (values, (np.asarray(row_idx), np.asarray(col_idx))),
            shape=(total_users, len(node_ids)),
            dtype=np.float32,
        )

    adj = b_matrix.transpose().dot(b_matrix)
    adj = _normalize_adj(adj)

    return NewsGraph(
        train_conf=train_conf,
        adj=adj,
        node_ids=node_ids,
        node_to_index=node_to_index,
    )


def load_preprocessed_graph(
    dataset: str,
    n_samples: int,
    user_threshold: int = 5,
    data_dir: str | Path = "data",
) -> NewsGraph:
    """Load train_conf + adjacency from preprocessed pickle."""
    data_root = resolve_data_root(data_dir)
    pkl_path = _preprocessed_adj_path(
        data_root=data_root,
        dataset=dataset,
        n_samples=n_samples,
        user_threshold=user_threshold,
    )
    payload = load_pickle(pkl_path)
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        raise ValueError(f"Expected [train_conf, adj] in {pkl_path}")

    train_conf_raw, adj_raw = payload
    train_conf = np.asarray(train_conf_raw, dtype=np.float32)

    if sp.issparse(adj_raw):
        adj = adj_raw.tocsr().astype(np.float32)
    else:
        adj = sp.csr_matrix(np.asarray(adj_raw, dtype=np.float32))

    try:
        node_ids = _load_node_ids_from_raw(
            dataset=dataset,
            n_samples=n_samples,
            data_root=data_root,
        )
    except (FileNotFoundError, ValueError):
        node_ids = [f"node_{idx}" for idx in range(adj.shape[0])]

    if len(node_ids) < adj.shape[0]:
        extra = [f"node_{idx}" for idx in range(len(node_ids), adj.shape[0])]
        node_ids = node_ids + extra
    elif len(node_ids) > adj.shape[0]:
        node_ids = node_ids[: adj.shape[0]]

    node_to_index = {nid: idx for idx, nid in enumerate(node_ids)}
    return NewsGraph(
        train_conf=train_conf,
        adj=adj,
        node_ids=node_ids,
        node_to_index=node_to_index,
    )


def load_graph(
    dataset: str,
    n_samples: int,
    user_threshold: int = 5,
    data_dir: str | Path = "data",
    prefer_preprocessed: bool = True,
) -> NewsGraph:
    """
    Load graph from preprocessed pickle when available; otherwise build from raw.
    """
    if prefer_preprocessed:
        try:
            return load_preprocessed_graph(
                dataset=dataset,
                n_samples=n_samples,
                user_threshold=user_threshold,
                data_dir=data_dir,
            )
        except (FileNotFoundError, pickle.UnpicklingError, ValueError, EOFError):
            pass

    return build_graph_from_raw(
        dataset=dataset,
        n_samples=n_samples,
        user_threshold=user_threshold,
        data_dir=data_dir,
    )


def load_train_conf_and_adj(
    dataset: str,
    n_samples: int,
    user_threshold: int = 5,
    data_dir: str | Path = "data",
    prefer_preprocessed: bool = True,
) -> tuple[np.ndarray, sp.csr_matrix]:
    """Direct helper for main flow: returns train_conf and normalized adjacency."""
    graph = load_graph(
        dataset=dataset,
        n_samples=n_samples,
        user_threshold=user_threshold,
        data_dir=data_dir,
        prefer_preprocessed=prefer_preprocessed,
    )
    return graph.train_conf, graph.adj


def get_first_hop_neighbors(
    graph: NewsGraph,
    node: NodeRef,
    return_ids: bool = True,
) -> list[NodeRef]:
    return graph.first_hop_neighbors(node=node, return_ids=return_ids)


def get_second_hop_neighbors(
    graph: NewsGraph,
    node: NodeRef,
    return_ids: bool = True,
) -> list[NodeRef]:
    return graph.second_hop_neighbors(node=node, return_ids=return_ids)
