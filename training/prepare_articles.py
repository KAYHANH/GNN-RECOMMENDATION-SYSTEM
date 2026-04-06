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


def normalize_article_id(article_id: str | int) -> str:
    return str(article_id).strip().zfill(10)


def hm_image_relative_path(article_id: str | int) -> str:
    normalized_article_id = normalize_article_id(article_id)
    return f"{normalized_article_id[:3]}/{normalized_article_id}.jpg"


def create_embedding_text(row: pd.Series) -> str:
    return (
        f"Name: {row['prod_name']}. "
        f"Type: {row['product_type_name']} in {row['product_group_name']}. "
        f"Appearance: {row['graphical_appearance_name']} with color {row['colour_group_name']}. "
        f"Category: {row['department_name']}, {row['section_name']}. "
        f"Description: {row['detail_desc']}"
    )


def attach_image_columns(articles_df: pd.DataFrame, images_dir: str | Path | None = None) -> pd.DataFrame:
    frame = articles_df.copy()
    frame["article_id"] = frame["article_id"].astype(str).map(normalize_article_id)
    frame["image_relative_path"] = frame["article_id"].map(hm_image_relative_path)

    if images_dir is None:
        frame["image_available"] = False
        frame["image_local_path"] = ""
        return frame

    base_dir = Path(images_dir)
    image_paths = frame["image_relative_path"].map(lambda relative_path: base_dir / relative_path)
    frame["image_available"] = image_paths.map(Path.exists)
    frame["image_local_path"] = image_paths.map(lambda path: str(path.resolve()) if path.exists() else "")
    return frame


def main() -> None:
    parser = ArgumentParser(description="Prepare articles.csv into articles_cleaned.csv for training and semantic search.")
    parser.add_argument("--input", required=True, help="Path to raw articles.csv")
    parser.add_argument("--output", default="data/articles_cleaned.csv", help="Path for cleaned output")
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Optional path to the Kaggle H&M images directory so cleaned outputs can include image metadata.",
    )
    args = parser.parse_args()

    articles_df = pd.read_csv(args.input, dtype={"article_id": str})
    articles_df["article_id"] = articles_df["article_id"].astype(str).map(normalize_article_id)

    for column in TEXT_COLUMNS:
        if column not in articles_df.columns:
            articles_df[column] = ""
        articles_df[column] = articles_df[column].fillna("")

    articles_df["text_for_embedding"] = articles_df.apply(create_embedding_text, axis=1)
    articles_df = attach_image_columns(articles_df, images_dir=args.images_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    articles_df.to_csv(output_path, index=False)
    print(f"Saved cleaned articles to {output_path.resolve()}")


if __name__ == "__main__":
    main()
