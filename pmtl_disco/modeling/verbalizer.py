from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

DEFAULT_REAL_LABEL_WORDS = (
    "news",
    "real",
    "true",
    "factual",
    "authentic",
    "genuine",
    "verified",
    "credible",
    "accurate",
    "legit",
)

DEFAULT_FAKE_LABEL_WORDS = (
    "rumor",
    "fake",
    "false",
    "hoax",
    "misleading",
    "fabricated",
    "bogus",
    "deceptive",
    "untrue",
    "phony",
)


class ExpandedLabelVerbalizer(nn.Module):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        real_label_words: Optional[Sequence[str]] = None,
        fake_label_words: Optional[Sequence[str]] = None,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.1,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.eps = eps

        self.real_label_words = tuple(real_label_words or DEFAULT_REAL_LABEL_WORDS)
        self.fake_label_words = tuple(fake_label_words or DEFAULT_FAKE_LABEL_WORDS)

        if len(self.real_label_words) == 0 or len(self.fake_label_words) == 0:
            raise ValueError("Both real_label_words and fake_label_words must be non-empty.")

        real_token_ids = self._resolve_label_word_ids(self.real_label_words)
        fake_token_ids = self._resolve_label_word_ids(self.fake_label_words)

        self.register_buffer("real_token_ids", torch.tensor(real_token_ids, dtype=torch.long), persistent=False)
        self.register_buffer("fake_token_ids", torch.tensor(fake_token_ids, dtype=torch.long), persistent=False)

        self.real_weight_mlp = nn.Sequential(
            nn.Linear(len(real_token_ids), mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, len(real_token_ids)),
        )
        self.fake_weight_mlp = nn.Sequential(
            nn.Linear(len(fake_token_ids), mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, len(fake_token_ids)),
        )

    def _resolve_label_word_ids(self, label_words: Sequence[str]) -> Sequence[int]:
        ids = []
        unk_id = self.tokenizer.unk_token_id

        for word in label_words:
            pieces = self.tokenizer.encode(word, add_special_tokens=False)
            if len(pieces) != 1:
                continue
            token_id = pieces[0]
            if unk_id is not None and token_id == unk_id:
                continue
            ids.append(token_id)
        ids = list(dict.fromkeys(ids))
        if not ids:
            raise ValueError("No valid label word can be mapped to a single vocabulary token.")
        return ids

    def forward(self, mask_logits: Tensor) -> Dict[str, Tensor]:
        if mask_logits.ndim != 2:
            raise ValueError(f"Expected mask_logits shape [batch, vocab], got {tuple(mask_logits.shape)}.")

        token_probs = F.softmax(mask_logits, dim=-1)

        real_word_logits = mask_logits.index_select(dim=1, index=self.real_token_ids)
        fake_word_logits = mask_logits.index_select(dim=1, index=self.fake_token_ids)
        real_word_probs = token_probs.index_select(dim=1, index=self.real_token_ids)
        fake_word_probs = token_probs.index_select(dim=1, index=self.fake_token_ids)

        real_word_weights = F.softmax(self.real_weight_mlp(real_word_logits), dim=-1)
        fake_word_weights = F.softmax(self.fake_weight_mlp(fake_word_logits), dim=-1)

        real_prob = (real_word_weights * real_word_probs).sum(dim=-1)
        fake_prob = (fake_word_weights * fake_word_probs).sum(dim=-1)

        class_probs = torch.stack((real_prob, fake_prob), dim=-1)
        class_probs = class_probs / class_probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        return {
            "class_probs": class_probs,
            "real_prob": real_prob,
            "fake_prob": fake_prob,
            "real_word_probs": real_word_probs,
            "fake_word_probs": fake_word_probs,
            "real_word_weights": real_word_weights,
            "fake_word_weights": fake_word_weights,
            "real_token_ids": self.real_token_ids,
            "fake_token_ids": self.fake_token_ids,
        }
