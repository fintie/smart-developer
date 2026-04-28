from __future__ import annotations
import torch
from torch import nn


class CrossLayer(nn.Module):
    """
    One cross layer from the Deep & Cross Network family.

    Given x0 and xl:
        x_{l+1} = x0 * (W xl + b) + xl

    Here W xl is implemented as a scalar per sample:
        (xl @ w) -> shape [batch, 1]
    then broadcast-multiplied with x0.
    """
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.xavier_uniform_(self.weight.unsqueeze(0))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: [batch_size, input_dim]
            xl: [batch_size, input_dim]
        Returns:
            [batch_size, input_dim]
        """
        interaction = torch.sum(xl * self.weight, dim=1, keepdim=True)  # [B, 1]
        return x0 * interaction + self.bias + xl


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DCNReranker(nn.Module):
    """
    Deep & Cross Network style reranker.

    Input:
        A dense feature vector for each (query, candidate) pair.

    Output:
        A single logit for ranking / binary relevance prediction.
    """

    def __init__(
        self,
        input_dim: int,
        cross_layers: int = 3,
        deep_hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if deep_hidden_dims is None:
            deep_hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.cross_layers = nn.ModuleList(
            [CrossLayer(input_dim=input_dim) for _ in range(cross_layers)]
        )

        self.deep_tower = MLPBlock(
            input_dim=input_dim,
            hidden_dims=deep_hidden_dims,
            dropout=dropout,
        )

        final_dim = input_dim + self.deep_tower.output_dim
        self.output_layer = nn.Linear(final_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            logits: [batch_size]
        """
        x0 = x
        xl = x

        for cross_layer in self.cross_layers:
            xl = cross_layer(x0, xl)

        deep_out = self.deep_tower(x)
        combined = torch.cat([xl, deep_out], dim=1)
        logits = self.output_layer(combined).squeeze(1)
        return logits


def build_dcn_reranker_from_config(cfg: dict) -> DCNReranker:
    model_cfg = cfg["model"]
    return DCNReranker(
        input_dim=int(model_cfg["input_dim"]),
        cross_layers=int(model_cfg["cross_layers"]),
        deep_hidden_dims=list(model_cfg["deep_hidden_dims"]),
        dropout=float(model_cfg["dropout"]),
    )