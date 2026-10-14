"""Shared H&E features, independent task queries, and private task heads."""

import torch
from torch import nn
from torch.nn import functional as F

from biomarkers import TASKS


class QueryDecoupledMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("Positive dimensions required; hidden_dim must be divisible by num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                        nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.task_queries = nn.Parameter(torch.randn(len(TASKS), hidden_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads,
                                                     dropout=dropout, batch_first=True)
        # No query-to-query self-attention: each task reads the shared bag independently.
        self.task_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                          nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
            for _ in TASKS
        ])

    def forward(self, features, padding_mask=None, return_attention=False):
        if features.ndim != 3 or features.shape[1] == 0:
            raise ValueError("features must have shape [batch, nonempty patches, feature_dim]")
        if padding_mask is not None and padding_mask.all(dim=1).any():
            raise ValueError("A bag cannot contain only padding")
        tokens = self.projection(features)
        queries = self.task_queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        attended, attention = self.cross_attention(
            queries, tokens, tokens, key_padding_mask=padding_mask,
            need_weights=return_attention, average_attn_weights=True,
        )
        representations = queries + attended
        logits = torch.cat([head(representations[:, i])
                            for i, head in enumerate(self.task_heads)], dim=1)
        return (logits, attention) if return_attention else logits

    def query_penalty(self):
        normalized = F.normalize(self.task_queries, dim=-1)
        similarity = normalized @ normalized.T
        off_diagonal = ~torch.eye(len(TASKS), dtype=torch.bool, device=similarity.device)
        return similarity[off_diagonal].square().mean()


def masked_bce(logits, labels, pos_weight=None):
    """Average observed losses within each task, then equally across available tasks."""
    observed = labels >= 0
    losses = F.binary_cross_entropy_with_logits(logits, labels.clamp_min(0),
                                                pos_weight=pos_weight, reduction="none")
    counts = observed.sum(dim=0)
    available = counts > 0
    if not available.any():
        return logits.sum() * 0
    per_task = (losses * observed).sum(dim=0) / counts.clamp_min(1)
    return per_task[available].mean()
