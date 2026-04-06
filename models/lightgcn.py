from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd

from models.common import RecommendationCandidate

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional at inspection time
    torch = None
    nn = None
    F = None

try:
    from torch_geometric.nn import LGConv
except ImportError:  # pragma: no cover - optional at inspection time
    LGConv = None


if nn is not None:

    class LightGCN(nn.Module):
        def __init__(self, num_users: int, num_items: int, emb_dim: int = 64, n_layers: int = 3) -> None:
            super().__init__()
            if LGConv is None:
                raise ImportError("torch-geometric is required to instantiate LightGCN.")

            self.num_users = num_users
            self.num_items = num_items
            self.emb_dim = emb_dim
            self.n_layers = n_layers
            self.user_emb = nn.Embedding(num_users, emb_dim)
            self.item_emb = nn.Embedding(num_items, emb_dim)
            self.convs = nn.ModuleList([LGConv() for _ in range(n_layers)])
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.xavier_uniform_(self.user_emb.weight)
            nn.init.xavier_uniform_(self.item_emb.weight)

        def forward(self, edge_index: "torch.Tensor") -> "torch.Tensor":
            x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
            all_embeddings = [x]
            for conv in self.convs:
                x = conv(x, edge_index)
                all_embeddings.append(x)
            return torch.stack(all_embeddings, dim=1).mean(dim=1)

        def split_embeddings(self, edge_index: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            all_embeddings = self.forward(edge_index)
            return torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        def score(self, user_indices: "torch.Tensor", item_indices: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
            user_embeddings, item_embeddings = self.split_embeddings(edge_index)
            user_repr = user_embeddings[user_indices]
            item_repr = item_embeddings[item_indices]
            return (user_repr * item_repr).sum(dim=-1)

else:

    class LightGCN:  # pragma: no cover - defensive fallback for missing torch
        def __init__(self, *_: object, **__: object) -> None:
            raise ImportError("torch and torch-geometric are required to instantiate LightGCN.")


def bpr_loss(
    user_emb: "torch.Tensor",
    pos_emb: "torch.Tensor",
    neg_emb: "torch.Tensor",
    reg: float = 1e-4,
) -> "torch.Tensor":
    if F is None:
        raise ImportError("torch is required to compute BPR loss.")

    pos_score = (user_emb * pos_emb).sum(dim=-1)
    neg_score = (user_emb * neg_emb).sum(dim=-1)
    ranking_loss = -F.logsigmoid(pos_score - neg_score).mean()
    reg_loss = reg * (
        user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)
    ) / user_emb.shape[0]
    return ranking_loss + reg_loss


@dataclass(slots=True)
class LightGCNArtifacts:
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray
    user_mapping: dict[str, int]
    item_mapping: dict[str, int]


class LightGCNRecommender:
    def __init__(
        self,
        *,
        artifacts: LightGCNArtifacts,
        articles_df: pd.DataFrame | None = None,
        interactions_df: pd.DataFrame | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.articles_df = self._normalize_article_frame(articles_df)
        self.interactions_df = self._normalize_interactions(interactions_df)
        self._idx_to_article = {idx: article_id for article_id, idx in artifacts.item_mapping.items()}
        self._purchases_by_user = self._build_purchase_lookup(self.interactions_df)

    @classmethod
    def from_artifacts(
        cls,
        *,
        user_embeddings_path: str | Path,
        item_embeddings_path: str | Path,
        user_mapping_path: str | Path,
        item_mapping_path: str | Path,
        articles_path: str | Path | None = None,
        interactions_path: str | Path | None = None,
    ) -> "LightGCNRecommender":
        user_embeddings = np.load(user_embeddings_path)
        item_embeddings = np.load(item_embeddings_path)

        with open(user_mapping_path, "r", encoding="utf-8") as handle:
            user_mapping = {str(key): int(value) for key, value in json.load(handle).items()}

        with open(item_mapping_path, "r", encoding="utf-8") as handle:
            item_mapping = {str(key): int(value) for key, value in json.load(handle).items()}

        articles_df = pd.read_csv(articles_path, dtype={"article_id": str}) if articles_path else None
        interactions_df = (
            pd.read_csv(
                interactions_path,
                dtype={"customer_id": str, "article_id": str},
                parse_dates=["t_dat"],
            )
            if interactions_path
            else None
        )

        artifacts = LightGCNArtifacts(
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
        )
        return cls(artifacts=artifacts, articles_df=articles_df, interactions_df=interactions_df)

    @staticmethod
    def _normalize_article_frame(articles_df: pd.DataFrame | None) -> pd.DataFrame | None:
        if articles_df is None:
            return None
        frame = articles_df.copy()
        if "article_id" in frame.columns:
            frame["article_id"] = frame["article_id"].astype(str).str.zfill(10)
        return frame

    @staticmethod
    def _normalize_interactions(interactions_df: pd.DataFrame | None) -> pd.DataFrame | None:
        if interactions_df is None:
            return None
        frame = interactions_df.copy()
        frame["customer_id"] = frame["customer_id"].astype(str)
        frame["article_id"] = frame["article_id"].astype(str).str.zfill(10)
        if "t_dat" in frame.columns:
            frame["t_dat"] = pd.to_datetime(frame["t_dat"])
        return frame

    @staticmethod
    def _build_purchase_lookup(interactions_df: pd.DataFrame | None) -> dict[str, set[str]]:
        if interactions_df is None or interactions_df.empty:
            return {}
        purchases = interactions_df.groupby("customer_id")["article_id"].agg(lambda values: set(values.astype(str)))
        return purchases.to_dict()

    def _article_metadata(self, article_id: str) -> dict[str, object]:
        if self.articles_df is None:
            return {}
        matches = self.articles_df[self.articles_df["article_id"] == str(article_id)]
        if matches.empty:
            return {}
        record = matches.iloc[0].to_dict()
        return {key: value for key, value in record.items() if pd.notna(value)}

    def _rank_for_user(self, customer_id: str) -> list[tuple[str, float]]:
        user_idx = self.artifacts.user_mapping.get(str(customer_id))
        if user_idx is None:
            return []

        user_vector = self.artifacts.user_embeddings[user_idx]
        scores = self.artifacts.item_embeddings @ user_vector
        purchased = self._purchases_by_user.get(str(customer_id), set())

        ranked: list[tuple[str, float]] = []
        for item_idx in np.argsort(-scores):
            article_id = self._idx_to_article[int(item_idx)]
            if article_id in purchased:
                continue
            ranked.append((article_id, float(scores[item_idx])))
        return ranked

    def get_candidates(self, customer_id: str, k: int = 100) -> list[RecommendationCandidate]:
        ranked = self._rank_for_user(customer_id)[:k]
        return [
            RecommendationCandidate(
                article_id=article_id,
                score=score,
                source="lightgcn",
                metadata=self._article_metadata(article_id),
                features={"gnn_similarity_score": score},
            )
            for article_id, score in ranked
        ]

    def recommend(self, customer_id: str, k: int = 12) -> list[str]:
        return [candidate.article_id for candidate in self.get_candidates(customer_id, k=k)]

    def customer_history(self, customer_id: str, limit: int = 10) -> list[dict[str, object]]:
        if self.interactions_df is None:
            return []
        history = self.interactions_df[self.interactions_df["customer_id"] == str(customer_id)]
        if history.empty:
            return []
        if "t_dat" in history.columns:
            history = history.sort_values("t_dat", ascending=False)
        history = history.head(limit)
        if self.articles_df is not None:
            history = history.merge(self.articles_df, on="article_id", how="left")
        return history.to_dict(orient="records")


def build_edge_index(
    interactions_df: pd.DataFrame,
    *,
    user_mapping: dict[str, int],
    item_mapping: dict[str, int],
) -> "torch.Tensor":
    if torch is None:
        raise ImportError("torch is required to build edge indices.")

    customer_indices = interactions_df["customer_id"].astype(str).map(user_mapping)
    article_indices = interactions_df["article_id"].astype(str).map(item_mapping)
    valid_mask = customer_indices.notna() & article_indices.notna()

    src = torch.tensor(customer_indices[valid_mask].astype(int).to_numpy(), dtype=torch.long)
    dst = torch.tensor(article_indices[valid_mask].astype(int).to_numpy(), dtype=torch.long)
    dst = dst + len(user_mapping)

    forward_edges = torch.stack([src, dst], dim=0)
    reverse_edges = torch.stack([dst, src], dim=0)
    return torch.cat([forward_edges, reverse_edges], dim=1)


def sample_bpr_triplets(
    interactions_df: pd.DataFrame,
    *,
    user_mapping: dict[str, int],
    item_mapping: dict[str, int],
    num_items: int,
    samples_per_user: int = 1,
    seed: int = 42,
) -> Iterable[tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    positives = interactions_df.groupby("customer_id")["article_id"].agg(lambda values: list(values.astype(str))).to_dict()

    for customer_id, bought_items in positives.items():
        user_idx = user_mapping.get(str(customer_id))
        if user_idx is None:
            continue

        bought_set = {item_mapping[item] for item in bought_items if item in item_mapping}
        if not bought_set:
            continue

        positive_indices = list(bought_set)
        for _ in range(samples_per_user):
            pos_idx = int(rng.choice(positive_indices))
            neg_idx = int(rng.integers(0, num_items))
            while neg_idx in bought_set:
                neg_idx = int(rng.integers(0, num_items))
            yield user_idx, pos_idx, neg_idx
