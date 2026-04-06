from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


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


def create_embedding_text(row: pd.Series) -> str:
    return (
        f"Name: {row['prod_name']}. "
        f"Type: {row['product_type_name']} in {row['product_group_name']}. "
        f"Appearance: {row['graphical_appearance_name']} with color {row['colour_group_name']}. "
        f"Category: {row['department_name']}, {row['section_name']}. "
        f"Description: {row['detail_desc']}"
    )


def main() -> None:
    parser = ArgumentParser(description="Prepare articles.csv into articles_cleaned.csv for training and semantic search.")
    parser.add_argument("--input", required=True, help="Path to raw articles.csv")
    parser.add_argument("--output", default="data/articles_cleaned.csv", help="Path for cleaned output")
    args = parser.parse_args()

    articles_df = pd.read_csv(args.input, dtype={"article_id": str})
    for column in TEXT_COLUMNS:
        if column not in articles_df.columns:
            articles_df[column] = ""
        articles_df[column] = articles_df[column].fillna("")

    articles_df["text_for_embedding"] = articles_df.apply(create_embedding_text, axis=1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    articles_df.to_csv(output_path, index=False)
    print(f"Saved cleaned articles to {output_path.resolve()}")


if __name__ == "__main__":
    main()

