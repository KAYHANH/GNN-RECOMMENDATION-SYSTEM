from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from training.prepare_articles import TEXT_COLUMNS, attach_image_columns, create_embedding_text, normalize_article_id


CUSTOMER_FILL_DEFAULTS = {
    "FN": 0,
    "Active": 0,
    "club_member_status": "UNKNOWN",
    "fashion_news_frequency": "UNKNOWN",
    "age": -1,
    "postal_code": "UNKNOWN",
}


def prepare_articles(
    articles_path: str | Path,
    article_ids: set[str],
    images_dir: str | Path | None = None,
) -> pd.DataFrame:
    articles_df = pd.read_csv(articles_path, dtype={"article_id": str})
    normalized_ids = {normalize_article_id(article_id) for article_id in article_ids}
    articles_df["article_id"] = articles_df["article_id"].astype(str).map(normalize_article_id)
    articles_df = articles_df[articles_df["article_id"].astype(str).isin(normalized_ids)].copy()

    for column in TEXT_COLUMNS:
        if column not in articles_df.columns:
            articles_df[column] = ""
        articles_df[column] = articles_df[column].fillna("")

    articles_df["text_for_embedding"] = articles_df.apply(create_embedding_text, axis=1)
    return attach_image_columns(articles_df, images_dir=images_dir)


def prepare_customers(customers_path: str | Path, customer_ids: set[str]) -> pd.DataFrame:
    customers_df = pd.read_csv(customers_path, dtype={"customer_id": str})
    customers_df = customers_df[customers_df["customer_id"].astype(str).isin(customer_ids)].copy()

    for column, default_value in CUSTOMER_FILL_DEFAULTS.items():
        if column not in customers_df.columns:
            customers_df[column] = default_value
        customers_df[column] = customers_df[column].fillna(default_value)

    return customers_df


def prepare_transactions(transactions_path: str | Path, days: int) -> pd.DataFrame:
    transactions_df = pd.read_csv(
        transactions_path,
        dtype={"customer_id": str, "article_id": str},
        parse_dates=["t_dat"],
    )
    transactions_df["article_id"] = transactions_df["article_id"].astype(str).str.zfill(10)
    transactions_df["customer_id"] = transactions_df["customer_id"].astype(str)

    if days <= 0:
        return transactions_df.copy()

    latest_date = transactions_df["t_dat"].max()
    cutoff = latest_date - pd.Timedelta(days=days)
    recent_transactions = transactions_df[transactions_df["t_dat"] > cutoff].copy()
    return recent_transactions


def output_paths(output_dir: Path, output_prefix: str) -> tuple[Path, Path, Path]:
    return (
        output_dir / f"{output_prefix}articles_cleaned.csv",
        output_dir / f"{output_prefix}customers_cleaned.csv",
        output_dir / f"{output_prefix}transactions_cleaned.csv",
    )


def main() -> None:
    parser = ArgumentParser(
        description="Create either the dense 14-day H&M subset or a full cleaned dataset for the fashion recommender."
    )
    parser.add_argument("--articles", required=True, help="Path to raw Kaggle articles.csv")
    parser.add_argument("--customers", required=True, help="Path to raw Kaggle customers.csv")
    parser.add_argument("--transactions", required=True, help="Path to raw Kaggle transactions_train.csv")
    parser.add_argument("--output-dir", default="data", help="Directory for cleaned subset outputs")
    parser.add_argument(
        "--mode",
        choices=["subset", "full"],
        default="subset",
        help="Use 'subset' for the recent dense window or 'full' for the full cleaned dataset.",
    )
    parser.add_argument("--days", type=int, default=14, help="Recent time window in days")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional filename prefix such as 'full_' to keep subset and full outputs side by side.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Optional path to the Kaggle H&M images directory so cleaned outputs include image metadata.",
    )
    args = parser.parse_args()

    requested_days = args.days if args.mode == "subset" else 0
    selected_transactions = prepare_transactions(args.transactions, days=requested_days)
    article_ids = set(selected_transactions["article_id"].astype(str).unique())
    customer_ids = set(selected_transactions["customer_id"].astype(str).unique())

    articles_df = prepare_articles(args.articles, article_ids, images_dir=args.images_dir)
    customers_df = prepare_customers(args.customers, customer_ids)

    filtered_transactions = selected_transactions[
        selected_transactions["article_id"].isin(articles_df["article_id"].astype(str))
        & selected_transactions["customer_id"].isin(customers_df["customer_id"].astype(str))
    ].copy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    articles_output, customers_output, transactions_output = output_paths(output_dir, args.output_prefix)

    articles_df.to_csv(articles_output, index=False)
    customers_df.to_csv(customers_output, index=False)
    filtered_transactions.to_csv(transactions_output, index=False)

    print(f"articles={len(articles_df)} -> {articles_output.resolve()}")
    print(f"customers={len(customers_df)} -> {customers_output.resolve()}")
    print(f"transactions={len(filtered_transactions)} -> {transactions_output.resolve()}")
    print(f"mode={args.mode}")
    print(f"window_days={args.days if args.mode == 'subset' else 'ALL'}")


if __name__ == "__main__":
    main()
