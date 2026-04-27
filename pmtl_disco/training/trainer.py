from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from pmtl_disco.inference import NeighborGraph, refine_uncertain_predictions


class PromptDataset(Dataset):
    """Text dataset with prompt template and mask position."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int,
        prompt_template: str,
    ) -> None:
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.mask_token_id = tokenizer.mask_token_id

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.prompt_template.format(text=self.texts[idx])
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        mask_pos = int((input_ids == self.mask_token_id).nonzero(as_tuple=True)[0][0].item())
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "mask_pos": torch.tensor(mask_pos, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


@dataclass
class TrainerConfig:
    lr: float = 5e-5
    batch_size: int = 8
    max_length: int = 128
    epochs: int = 3
    alpha1: float = 1.0
    alpha2: float = 0.01
    gamma: float = 0.1
    k_threshold: float = 0.35
    mu: float = 0.65
    prompt_template: str = "It is a [MASK] news: {text}"
    use_reasoning: bool = True


@dataclass
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float


class PMTLDisCoTrainer:
    """
    Joint optimization trainer for PMTL-DisCo.

    Expected model forward output keys:
    - class_probs: Tensor[batch, 2]
    - cls_embedding: Tensor[batch, hidden]
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config: TrainerConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

    def _make_loader(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        shuffle: bool,
    ) -> DataLoader:
        ds = PromptDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            prompt_template=self.config.prompt_template,
        )
        return DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "mask_positions": batch["mask_pos"].to(self.device),
        }
        outputs = self.model(**inputs)
        if "cls_embedding" not in outputs and "cls_repr" in outputs:
            outputs["cls_embedding"] = outputs["cls_repr"]
        return outputs

    def _classification_loss(
        self, class_probs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        labels = labels.to(self.device)
        return F.nll_loss(torch.log(class_probs + 1e-12), labels)

    def _sample_negative(
        self,
        anchor: int,
        n_nodes: int,
        forbidden: set[int],
        rng: np.random.Generator,
    ) -> Optional[int]:
        candidates = np.setdiff1d(np.arange(n_nodes), np.array(sorted(forbidden), dtype=np.int64))
        if len(candidates) == 0:
            return None
        return int(rng.choice(candidates))

    def _auxiliary_loss(
        self,
        batch_global_indices: np.ndarray,
        train_texts: Sequence[str],
        test_texts: Sequence[str],
        anchor_embeddings: torch.Tensor,
        graph: NeighborGraph,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        all_texts = list(train_texts) + list(test_texts)
        n_nodes = len(all_texts)

        triples: List[Tuple[int, int, int]] = []
        for local_idx, anchor in enumerate(batch_global_indices.tolist()):
            pos_list = graph.first_hop.get(anchor, [])
            if not pos_list:
                continue
            pos = int(rng.choice(np.array(pos_list, dtype=np.int64)))
            neg = self._sample_negative(anchor, n_nodes, set(pos_list) | {anchor}, rng)
            if neg is None:
                continue
            triples.append((local_idx, pos, neg))

        if not triples:
            return torch.zeros((), device=self.device)

        pn_texts = [all_texts[pos] for _, pos, _ in triples] + [
            all_texts[neg] for _, _, neg in triples
        ]
        pn_prompts = [self.config.prompt_template.format(text=t) for t in pn_texts]
        pn_tokens = self.tokenizer(
            pn_prompts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        pn_input_ids = pn_tokens["input_ids"].to(self.device)
        pn_attention = pn_tokens["attention_mask"].to(self.device)
        pn_mask_pos = (pn_input_ids == self.tokenizer.mask_token_id).float().argmax(dim=1).long()
        pn_outputs = self.model(
            input_ids=pn_input_ids,
            attention_mask=pn_attention,
            mask_positions=pn_mask_pos,
        )
        pn_embeddings = pn_outputs.get("cls_embedding", pn_outputs["cls_repr"])
        half = len(triples)
        pos_embeddings = pn_embeddings[:half]
        neg_embeddings = pn_embeddings[half:]

        anchor_vec = anchor_embeddings[
            torch.tensor([x[0] for x in triples], dtype=torch.long, device=self.device)
        ]
        pos_sim = F.cosine_similarity(anchor_vec, pos_embeddings, dim=-1)
        neg_sim = F.cosine_similarity(anchor_vec, neg_embeddings, dim=-1)
        return torch.relu(self.config.gamma + pos_sim - neg_sim).mean()

    def train_one_iteration(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[int],
        test_texts: Sequence[str],
        test_labels: Sequence[int],
        adjacency: sp.spmatrix,
        seed: int = 0,
    ) -> EvalResult:
        rng = np.random.default_rng(seed)
        train_loader = self._make_loader(train_texts, train_labels, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        total_steps = max(1, len(train_loader) * self.config.epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        neighbor_graph = NeighborGraph.from_matrix(adjacency.tocsr())

        for _ in range(self.config.epochs):
            self.model.train()
            for batch in tqdm(train_loader, desc="train", leave=False):
                outputs = self._forward_batch(batch)
                rv_loss = self._classification_loss(outputs["class_probs"], batch["labels"])

                global_indices = batch["index"].numpy().astype(np.int64)
                dr_loss = self._auxiliary_loss(
                    batch_global_indices=global_indices,
                    train_texts=train_texts,
                    test_texts=test_texts,
                    anchor_embeddings=outputs["cls_embedding"],
                    graph=neighbor_graph,
                    rng=rng,
                )

                total_loss = self.config.alpha1 * rv_loss + self.config.alpha2 * dr_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()

        return self.evaluate(
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=test_texts,
            test_labels=test_labels,
            adjacency=adjacency,
        )

    def predict_probabilities(self, texts: Sequence[str]) -> np.ndarray:
        dummy_labels = [0] * len(texts)
        loader = self._make_loader(texts, dummy_labels, shuffle=False)
        self.model.eval()
        probs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="infer", leave=False):
                outputs = self._forward_batch(batch)
                p = outputs["class_probs"].detach().cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0)

    def evaluate(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[int],
        test_texts: Sequence[str],
        test_labels: Sequence[int],
        adjacency: sp.spmatrix,
    ) -> EvalResult:
        all_texts = list(train_texts) + list(test_texts)
        probs = self.predict_probabilities(all_texts)

        if self.config.use_reasoning:
            graph = NeighborGraph.from_matrix(adjacency.tocsr())
            probs = refine_uncertain_predictions(
                probs=probs,
                graph=graph,
                k_threshold=self.config.k_threshold,
                mu=self.config.mu,
            )

        test_probs = probs[len(train_texts) :]
        pred = test_probs.argmax(axis=1)
        gold = np.asarray(test_labels, dtype=np.int64)
        acc = accuracy_score(gold, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold, pred, average="macro", zero_division=0
        )
        return EvalResult(
            accuracy=float(acc),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
        )
