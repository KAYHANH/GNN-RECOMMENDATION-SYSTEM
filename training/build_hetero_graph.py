from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import pandas as pd

from training.data_utils import load_articles, load_transactions

try:
    import torch
    from torch_geometric.data import HeteroData
except ImportError as exc:  # pragma: no cover - optional at inspection time
    raise ImportError("torch and torch-geometric are required to build the heterogeneous graph.") from exc


def _mapping(values: pd.Series) -> dict[str, int]:
    unique_values = sorted(values.astype(str).dropna().unique().tolist())
    return {value: idx for idx, value in enumerate(unique_values)}


def _tensor_from_pairs(source_values: pd.Series, target_values: pd.Series, source_map: dict[str, int], target_map: dict[str, int]) -> torch.Tensor:
    src = source_values.astype(str).map(source_map)
    dst = target_values.astype(str).map(target_map)
    valid_mask = src.notna() & dst.notna()
    return torch.tensor(
        [
            src[valid_mask].astype(int).to_numpy(),
            dst[valid_mask].astype(int).to_numpy(),
        ],
        dtype=torch.long,
    )


def main() -> None:
    parser = ArgumentParser(description="Build the heterogeneous H&M graph with customer/article/attribute nodes.")
    parser.add_argument("--articles", default="data/articles_cleaned.csv")
    parser.add_argument("--customers", default="data/customers_cleaned.csv")
    parser.add_argument("--transactions", default="data/transactions_cleaned.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--embedding-dim", type=int, default=64)
    args = parser.parse_args()

    articles_df = load_articles(args.articles)
    customers_df = pd.read_csv(args.customers, dtype={"customer_id": str})
    transactions_df = load_transactions(args.transactions)

    transactions_df["article_id"] = transactions_df["article_id"].astype(str)
    transactions_df["customer_id"] = transactions_df["customer_id"].astype(str)
    articles_df["article_id"] = articles_df["article_id"].astype(str)
    customers_df["customer_id"] = customers_df["customer_id"].astype(str)

    customer_map = _mapping(customers_df["customer_id"])
    article_map = _mapping(articles_df["article_id"])
    product_group_map = _mapping(articles_df["product_group_name"].fillna("UNKNOWN"))
    colour_group_map = _mapping(articles_df["colour_group_name"].fillna("UNKNOWN"))

    article_attributes = articles_df[["article_id", "product_group_name", "colour_group_name"]].copy()
    article_attributes["product_group_name"] = article_attributes["product_group_name"].fillna("UNKNOWN")
    article_attributes["colour_group_name"] = article_attributes["colour_group_name"].fillna("UNKNOWN")

    data = HeteroData()
    data["customer"].x = torch.randn(len(customer_map), args.embedding_dim)
    data["article"].x = torch.randn(len(article_map), args.embedding_dim)
    data["product_group"].x = torch.randn(len(product_group_map), args.embedding_dim)
    data["colour_group"].x = torch.randn(len(colour_group_map), args.embedding_dim)

    bought = _tensor_from_pairs(transactions_df["customer_id"], transactions_df["article_id"], customer_map, article_map)
    belongs_to = _tensor_from_pairs(article_attributes["article_id"], article_attributes["product_group_name"], article_map, product_group_map)
    has_colour = _tensor_from_pairs(article_attributes["article_id"], article_attributes["colour_group_name"], article_map, colour_group_map)

    data["customer", "bought", "article"].edge_index = bought
    data["article", "rev_bought", "customer"].edge_index = bought.flip(0)
    data["article", "belongs_to", "product_group"].edge_index = belongs_to
    data["product_group", "rev_belongs_to", "article"].edge_index = belongs_to.flip(0)
    data["article", "has_colour", "colour_group"].edge_index = has_colour
    data["colour_group", "rev_has_colour", "article"].edge_index = has_colour.flip(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(data, output_dir / "hetero_graph_data.pt")

    for filename, mapping in [
        ("customer_mapping.json", customer_map),
        ("article_mapping.json", article_map),
        ("product_group_mapping.json", product_group_map),
        ("colour_group_mapping.json", colour_group_map),
    ]:
        with open(output_dir / filename, "w", encoding="utf-8") as handle:
            json.dump(mapping, handle, indent=2)

    print(f"Saved heterogeneous graph to {(output_dir / 'hetero_graph_data.pt').resolve()}")
    print(data)


if __name__ == "__main__":
    main()

