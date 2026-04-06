from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd

from training.data_utils import interaction_lookup, load_transactions, temporal_split


def recall_at_k(predicted: list[str], actual: list[str], k: int) -> float:
    if not actual:
        return 0.0
    predicted_k = set(predicted[:k])
    return len(predicted_k & set(actual)) / min(len(actual), k)


def ndcg_at_k(predicted: list[str], actual: list[str], k: int) -> float:
    if not actual:
        return 0.0
    actual_set = set(actual)
    dcg = sum(1 / np.log2(index + 2) for index, item_id in enumerate(predicted[:k]) if item_id in actual_set)
    idcg = sum(1 / np.log2(index + 2) for index in range(min(len(actual), k)))
    return float(dcg / idcg) if idcg > 0 else 0.0


class PopularityBaseline:
    def __init__(self, train_df: pd.DataFrame) -> None:
        self.rankings = train_df["article_id"].astype(str).value_counts().index.tolist()
        self.purchase_lookup = (
            train_df.groupby("customer_id")["article_id"].agg(lambda values: set(values.astype(str))).to_dict()
        )

    def recommend(self, customer_id: str, k: int = 20) -> list[str]:
        purchased = self.purchase_lookup.get(str(customer_id), set())
        return [article_id for article_id in self.rankings if article_id not in purchased][:k]


def evaluate(
    model: object,
    test_interactions: dict[str, list[str]],
    *,
    k_values: list[int] | tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    max_k = max(k_values)
    results: dict[str, list[float]] = {f"Recall@{k}": [] for k in k_values}
    results.update({f"NDCG@{k}": [] for k in k_values})
    latencies_ms: list[float] = []

    for customer_id, true_items in test_interactions.items():
        started = perf_counter()
        predictions = model.recommend(customer_id, k=max_k)
        latencies_ms.append((perf_counter() - started) * 1000)

        for k in k_values:
            results[f"Recall@{k}"].append(recall_at_k(predictions, true_items, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(predictions, true_items, k))

    metrics = {metric: float(np.mean(values)) if values else 0.0 for metric, values in results.items()}
    metrics["LatencyMs"] = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    metrics["UsersEvaluated"] = float(len(test_interactions))
    return metrics


def compare_models(models: dict[str, object], test_interactions: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for model_name, model in models.items():
        metrics = evaluate(model, test_interactions)
        rows.append({"Model": model_name, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    parser = ArgumentParser(description="Run temporal offline evaluation for the fashion recommender.")
    parser.add_argument("--transactions", required=True, help="Path to transactions_cleaned.csv")
    parser.add_argument("--output", default="artifacts/metrics.json", help="Where to store metrics JSON")
    parser.add_argument("--test-weeks", type=int, default=4, help="Number of weeks to reserve for test")
    parser.add_argument("--test-days", type=int, default=None, help="Optional day-based holdout, useful for short subsets")
    args = parser.parse_args()

    transactions_df = load_transactions(args.transactions)
    train_df, test_df = temporal_split(transactions_df, test_weeks=args.test_weeks, test_days=args.test_days)

    baseline = PopularityBaseline(train_df)
    test_interactions = interaction_lookup(test_df)
    metrics = evaluate(baseline, test_interactions)
    metrics["Model"] = "Popularity baseline"
    metrics["TrainRows"] = float(len(train_df))
    metrics["TestRows"] = float(len(test_df))
    metrics["TrainStart"] = str(train_df["t_dat"].min()) if not train_df.empty else None
    metrics["TrainEnd"] = str(train_df["t_dat"].max()) if not train_df.empty else None
    metrics["TestStart"] = str(test_df["t_dat"].min()) if not test_df.empty else None
    metrics["TestEnd"] = str(test_df["t_dat"].max()) if not test_df.empty else None

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
