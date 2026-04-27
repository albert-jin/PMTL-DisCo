from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, BertForMaskedLM, PreTrainedTokenizerBase

from .losses import distribution_ranking_loss, joint_multitask_loss, rumor_verification_loss
from .verbalizer import ExpandedLabelVerbalizer


class PMTLDisCoModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        prompt_template: str = "It is a [MASK] news: {text}",
        real_label_words: Optional[Sequence[str]] = None,
        fake_label_words: Optional[Sequence[str]] = None,
        verbalizer_hidden_dim: int = 128,
        verbalizer_dropout: float = 0.1,
        alpha1: float = 1.0,
        alpha2: float = 0.01,
        gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self.plm = BertForMaskedLM.from_pretrained(model_name_or_path)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name_or_path)
        self.prompt_template = prompt_template

        self.verbalizer = ExpandedLabelVerbalizer(
            tokenizer=self.tokenizer,
            real_label_words=real_label_words,
            fake_label_words=fake_label_words,
            mlp_hidden_dim=verbalizer_hidden_dim,
            dropout=verbalizer_dropout,
        )

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.gamma = gamma

    def build_prompt(self, text: str) -> str:
        return self.prompt_template.format(text=text)

    def build_prompts(self, texts: Sequence[str]) -> Sequence[str]:
        return [self.build_prompt(text) for text in texts]

    def tokenize_prompted_texts(
        self,
        texts: Sequence[str],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, Tensor]:
        prompts = self.build_prompts(texts)
        encoded = self.tokenizer(
            prompts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )
        return dict(encoded)

    def infer_mask_positions(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape [batch, seq], got {tuple(input_ids.shape)}")

        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError("Tokenizer does not define a mask token id.")

        is_mask = input_ids.eq(mask_token_id)
        if not is_mask.any(dim=1).all():
            raise ValueError("Each sample must contain at least one [MASK] token.")

        return is_mask.float().argmax(dim=1).long()

    def encode_cls(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        encoder_outputs = self.plm.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return encoder_outputs.last_hidden_state[:, 0, :]

    def compute_distribution_loss(
        self,
        anchor_input_ids: Tensor,
        positive_input_ids: Tensor,
        negative_input_ids: Tensor,
        anchor_attention_mask: Optional[Tensor] = None,
        positive_attention_mask: Optional[Tensor] = None,
        negative_attention_mask: Optional[Tensor] = None,
        anchor_token_type_ids: Optional[Tensor] = None,
        positive_token_type_ids: Optional[Tensor] = None,
        negative_token_type_ids: Optional[Tensor] = None,
        gamma: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        anchor_cls = self.encode_cls(anchor_input_ids, anchor_attention_mask, anchor_token_type_ids)
        positive_cls = self.encode_cls(positive_input_ids, positive_attention_mask, positive_token_type_ids)
        negative_cls = self.encode_cls(negative_input_ids, negative_attention_mask, negative_token_type_ids)

        loss_dr = distribution_ranking_loss(
            anchor_repr=anchor_cls,
            positive_repr=positive_cls,
            negative_repr=negative_cls,
            gamma=self.gamma if gamma is None else gamma,
        )
        return {
            "loss_dr": loss_dr,
            "anchor_cls_repr": anchor_cls,
            "positive_cls_repr": positive_cls,
            "negative_cls_repr": negative_cls,
        }

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        mask_positions: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        anchor_repr: Optional[Tensor] = None,
        positive_repr: Optional[Tensor] = None,
        negative_repr: Optional[Tensor] = None,
        alpha1: Optional[float] = None,
        alpha2: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> Dict[str, Optional[Tensor]]:
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        logits = plm_outputs.logits
        hidden_state = plm_outputs.hidden_states[-1]
        cls_repr = hidden_state[:, 0, :]

        if mask_positions is None:
            mask_positions = self.infer_mask_positions(input_ids)
        else:
            mask_positions = mask_positions.to(logits.device).long()

        batch_size = logits.size(0)
        if mask_positions.numel() != batch_size:
            raise ValueError(
                f"Expected one mask position per sample, got {mask_positions.numel()} for batch size {batch_size}."
            )

        batch_indices = torch.arange(batch_size, device=logits.device)
        mask_logits = logits[batch_indices, mask_positions]

        verbalizer_outputs = self.verbalizer(mask_logits)
        class_probs = verbalizer_outputs["class_probs"]

        loss_rv: Optional[Tensor] = None
        if labels is not None:
            loss_rv = rumor_verification_loss(class_probs=class_probs, labels=labels)

        loss_dr: Optional[Tensor] = None
        if any(t is not None for t in (anchor_repr, positive_repr, negative_repr)):
            if not all(t is not None for t in (anchor_repr, positive_repr, negative_repr)):
                raise ValueError("anchor_repr, positive_repr, and negative_repr must be provided together.")
            loss_dr = distribution_ranking_loss(
                anchor_repr=anchor_repr,
                positive_repr=positive_repr,
                negative_repr=negative_repr,
                gamma=self.gamma if gamma is None else gamma,
            )

        loss_total: Optional[Tensor] = None
        if loss_rv is not None or loss_dr is not None:
            loss_total = joint_multitask_loss(
                loss_rv=loss_rv,
                loss_dr=loss_dr,
                alpha1=self.alpha1 if alpha1 is None else alpha1,
                alpha2=self.alpha2 if alpha2 is None else alpha2,
            )

        return {
            "class_probs": class_probs,
            "cls_repr": cls_repr,
            "mask_logits": mask_logits,
            "mask_positions": mask_positions,
            "token_logits": logits,
            "sequence_hidden_state": hidden_state,
            "real_word_probs": verbalizer_outputs["real_word_probs"],
            "fake_word_probs": verbalizer_outputs["fake_word_probs"],
            "real_word_weights": verbalizer_outputs["real_word_weights"],
            "fake_word_weights": verbalizer_outputs["fake_word_weights"],
            "loss_rv": loss_rv,
            "loss_dr": loss_dr,
            "loss_total": loss_total,
        }
