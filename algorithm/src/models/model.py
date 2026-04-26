from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        projection_dim: int,
        dropout: float = 0.1,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
        )
        self.normalize = normalize

    def mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        emb = self.proj(pooled)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        projection_dim: int,
        dropout: float = 0.1,
        normalize: bool = True,
        share_tower_weights: bool = True,
    ) -> None:
        super().__init__()

        self.query_encoder = TextEncoder(
            encoder_name=encoder_name,
            projection_dim=projection_dim,
            dropout=dropout,
            normalize=normalize
        )

        if share_tower_weights:
            self.candidate_encoder = self.query_encoder
        else:
            self.candidate_encoder = TextEncoder(
                encoder_name=encoder_name,
                projection_dim=projection_dim,
                dropout=dropout,
                normalize=normalize
            )

    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.query_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    def encode_candidate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.candidate_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )