from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


def _safe_log(value: float, eps: float = 1e-12) -> float:
    return float(np.log(max(value, eps)))


@dataclass
class NeighborGraph:
    """Compact neighborhood structure used by reasoning enhancement."""

    adjacency: sp.csr_matrix
    first_hop: Dict[int, List[int]]
    first_hop_weights: Dict[Tuple[int, int], float]
    second_hop: Dict[int, List[int]]

    @classmethod
    def from_matrix(cls, adjacency: sp.spmatrix) -> "NeighborGraph":
        adj = adjacency.tocsr()
        first_hop: Dict[int, List[int]] = {}
        first_hop_weights: Dict[Tuple[int, int], float] = {}
        second_hop: Dict[int, List[int]] = {}
        n_nodes = adj.shape[0]

        for i in range(n_nodes):
            row = adj.getrow(i)
            neighbors = row.indices.tolist()
            values = row.data.tolist()

            hop1: List[int] = []
            for nb, wt in zip(neighbors, values):
                if nb == i or wt <= 0:
                    continue
                hop1.append(int(nb))
                first_hop_weights[(i, int(nb))] = float(wt)
            first_hop[i] = hop1

        for i in range(n_nodes):
            two_hop = set()
            one_hop = set(first_hop.get(i, []))
            for nb in one_hop:
                for nb2 in first_hop.get(nb, []):
                    if nb2 == i or nb2 in one_hop:
                        continue
                    two_hop.add(nb2)
            second_hop[i] = sorted(two_hop)

        return cls(
            adjacency=adj,
            first_hop=first_hop,
            first_hop_weights=first_hop_weights,
            second_hop=second_hop,
        )

    def edge_weight(self, src: int, dst: int) -> float:
        return self.first_hop_weights.get((src, dst), 0.0)


def _compute_trustworthiness(
    node_id: int,
    graph: NeighborGraph,
    predicted_labels: np.ndarray,
) -> float:
    """Compute trustworthiness beta for a node based on 1/2-hop agreement."""
    neighborhood = list(graph.first_hop.get(node_id, []))
    neighborhood.extend(graph.second_hop.get(node_id, []))
    if not neighborhood:
        return 0.5

    node_label = int(predicted_labels[node_id])
    agree = 0
    for nb in neighborhood:
        if int(predicted_labels[nb]) == node_label:
            agree += 1
    return float(agree / len(neighborhood))


def _gamma_first_or_second_hop(
    target: int,
    neighbor: int,
    graph: NeighborGraph,
) -> float:
    if neighbor in graph.first_hop.get(target, []):
        return graph.edge_weight(target, neighbor)

    # For second-hop node, use max min(edge_1, edge_2) across valid paths.
    one_hops = graph.first_hop.get(target, [])
    candidate = 0.0
    for mid in one_hops:
        if neighbor in graph.first_hop.get(mid, []):
            candidate = max(
                candidate,
                min(graph.edge_weight(target, mid), graph.edge_weight(mid, neighbor)),
            )
    return candidate


def refine_uncertain_predictions(
    probs: np.ndarray,
    graph: NeighborGraph,
    k_threshold: float = 0.35,
    mu: float = 0.65,
) -> np.ndarray:
    """
    Refine uncertain predictions with multi-neighbor reasoning enhancement.

    Args:
        probs: shape [num_nodes, 2], predicted probabilities for [real, fake].
        graph: 1/2-hop neighborhood structure.
        k_threshold: uncertainty threshold K.
        mu: interpolation factor between self and neighborhood.
    """
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError("`probs` must be [num_nodes, 2].")
    if not 0.0 <= mu <= 1.0:
        raise ValueError("`mu` must be in [0, 1].")

    refined = probs.copy()
    labels = probs.argmax(axis=1)

    for node_id in range(probs.shape[0]):
        p_self = probs[node_id]
        if abs(float(p_self[0] - p_self[1])) > k_threshold:
            continue

        neighbors = list(graph.first_hop.get(node_id, []))
        neighbors.extend(graph.second_hop.get(node_id, []))
        if not neighbors:
            continue

        raw_scores: List[float] = []
        valid_neighbors: List[int] = []
        for nb in neighbors:
            beta = _compute_trustworthiness(nb, graph, labels)
            gamma = _gamma_first_or_second_hop(node_id, nb, graph)
            value = gamma * beta
            if value <= 0:
                continue
            raw_scores.append(_safe_log(value))
            valid_neighbors.append(nb)

        if not valid_neighbors:
            continue

        score_arr = np.asarray(raw_scores, dtype=np.float64)
        score_arr = score_arr - score_arr.max()
        weight = np.exp(score_arr)
        weight = weight / (weight.sum() + 1e-12)

        neighbor_probs = probs[np.asarray(valid_neighbors, dtype=np.int64)]
        mixed_neighbor = (neighbor_probs * weight[:, None]).sum(axis=0)
        refined[node_id] = mu * p_self + (1.0 - mu) * mixed_neighbor

    row_sum = refined.sum(axis=1, keepdims=True) + 1e-12
    refined = refined / row_sum
    return refined

