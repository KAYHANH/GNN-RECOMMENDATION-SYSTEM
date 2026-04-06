from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from training.data_utils import load_articles, load_transactions

try:
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - optional at inspection time
    raise ImportError("sentence-transformers and torch are required to fine-tune the encoder.") from exc


TEXT_COLUMNS = [
    "prod_name",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
]


def build_article_texts(articles_df: pd.DataFrame) -> pd.DataFrame:
    frame = articles_df.copy()
    for column in TEXT_COLUMNS:
        frame[column] = frame[column].fillna("") if column in frame.columns else ""

    frame["text_for_embedding"] = frame.apply(
        lambda row: (
            f"Name: {row['prod_name']}. "
            f"Type: {row['product_type_name']} in {row['product_group_name']}. "
            f"Appearance: {row['graphical_appearance_name']} with color {row['colour_group_name']}. "
            f"Category: {row['department_name']}, {row['section_name']}. "
            f"Description: {row['detail_desc']}"
        ),
        axis=1,
    )
    return frame


def build_triplets(articles_df: pd.DataFrame, transactions_df: pd.DataFrame, limit: int = 5000) -> list[InputExample]:
    frame = build_article_texts(articles_df)
    article_lookup = frame.set_index("article_id")["text_for_embedding"].to_dict()
    popularity = transactions_df["article_id"].astype(str).value_counts().index.tolist()

    examples: list[InputExample] = []
    sampled_transactions = transactions_df.head(limit)

    for _, row in sampled_transactions.iterrows():
        positive_id = str(row["article_id"])
        positive_text = article_lookup.get(positive_id)
        if not positive_text:
            continue

        query = f"{row.get('customer_id', 'customer')} prefers {positive_text.split('. ')[0].replace('Name: ', '')}"

        negative_id = next((article_id for article_id in popularity if article_id != positive_id), None)
        negative_text = article_lookup.get(str(negative_id)) if negative_id is not None else None
        if not negative_text:
            continue

        examples.append(InputExample(texts=[query, positive_text, negative_text]))

    return examples


def main() -> None:
    parser = ArgumentParser(description="Fine-tune the sentence encoder on fashion triplets.")
    parser.add_argument("--articles", required=True)
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--output-dir", default="artifacts/fashion-minilm-finetuned")
    args = parser.parse_args()

    articles_df = load_articles(args.articles)
    transactions_df = load_transactions(args.transactions)
    triplets = build_triplets(articles_df, transactions_df)
    if not triplets:
        raise ValueError("No triplets could be generated. Check that articles and transactions overlap.")

    model = SentenceTransformer(args.model_name)
    loader = DataLoader(triplets, shuffle=True, batch_size=args.batch_size)
    triplet_loss = losses.TripletLoss(model)

    model.fit(train_objectives=[(loader, triplet_loss)], epochs=args.epochs, warmup_steps=args.warmup_steps)

    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    print(f"Saved fine-tuned encoder to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

