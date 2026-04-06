from __future__ import annotations

from math import ceil
from pathlib import Path

import pandas as pd


def load_transactions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"customer_id": str, "article_id": str})
    frame["article_id"] = frame["article_id"].astype(str).str.zfill(10)
    frame["customer_id"] = frame["customer_id"].astype(str)
    frame["t_dat"] = pd.to_datetime(frame["t_dat"])
    return frame


def load_articles(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"article_id": str})
    frame["article_id"] = frame["article_id"].astype(str).str.zfill(10)
    return frame


def temporal_split(
    transactions_df: pd.DataFrame,
    *,
    test_weeks: int = 4,
    test_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if transactions_df.empty:
        return transactions_df.copy(), transactions_df.copy()

    min_date = transactions_df["t_dat"].min()
    max_date = transactions_df["t_dat"].max()
    total_days = max(1, (max_date - min_date).days)

    requested_days = test_days if test_days is not None else test_weeks * 7
    if requested_days >= total_days:
        requested_days = max(1, ceil(total_days * 0.2))

    cutoff = max_date - pd.Timedelta(days=requested_days)
    train = transactions_df[transactions_df["t_dat"] <= cutoff].copy()
    test = transactions_df[transactions_df["t_dat"] > cutoff].copy()

    if train.empty and not test.empty:
        train = transactions_df[transactions_df["t_dat"] < max_date].copy()
        test = transactions_df[transactions_df["t_dat"] >= max_date].copy()

    if test.empty and not train.empty:
        test = transactions_df[transactions_df["t_dat"] == max_date].copy()
        train = transactions_df[transactions_df["t_dat"] < max_date].copy()

    return train, test


def interaction_lookup(transactions_df: pd.DataFrame) -> dict[str, list[str]]:
    grouped = transactions_df.groupby("customer_id")["article_id"].agg(lambda values: list(values.astype(str)))
    return grouped.to_dict()


def build_id_mappings(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    customer_mapping = {
        customer_id: idx
        for idx, customer_id in enumerate(sorted(train_df["customer_id"].astype(str).unique()))
    }
    article_mapping = {
        article_id: idx
        for idx, article_id in enumerate(sorted(train_df["article_id"].astype(str).unique()))
    }
    return customer_mapping, article_mapping
