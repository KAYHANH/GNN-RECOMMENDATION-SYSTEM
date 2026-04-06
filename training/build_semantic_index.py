from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import pickle

import pandas as pd

from models.semantic_engine import (
    SEMANTIC_BACKEND_METADATA,
    SEMANTIC_PROJECTOR_ARTIFACT,
    SEMANTIC_VECTORIZER_ARTIFACT,
    SemanticEngine,
)
from training.data_utils import load_articles

try:
    import faiss
except ImportError as exc:  # pragma: no cover - optional at inspection time
    raise ImportError("faiss-cpu is required to build the semantic index.") from exc


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


def ensure_embedding_text(articles_df: pd.DataFrame) -> pd.DataFrame:
    if "text_for_embedding" in articles_df.columns and articles_df["text_for_embedding"].fillna("").astype(str).str.len().gt(0).any():
        return articles_df.copy()
    return build_article_texts(articles_df)


def main() -> None:
    parser = ArgumentParser(description="Build the semantic FAISS index for article discovery.")
    parser.add_argument("--articles", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--backend", default="auto", choices=["auto", "sentence-transformer", "tfidf-svd"])
    args = parser.parse_args()

    articles_df = ensure_embedding_text(load_articles(args.articles))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_ref = args.model_path or args.model_name
    preferred_backend = "sentence-transformer" if args.backend == "auto" else args.backend
    engine = SemanticEngine(model_name=model_ref, backend=preferred_backend)

    if args.backend == "auto":
        try:
            engine.build_index(articles_df, text_column="text_for_embedding")
        except Exception as exc:
            print(f"Sentence-transformer backend unavailable ({exc}). Falling back to tfidf-svd.")
            engine = SemanticEngine(model_name=model_ref, backend="tfidf-svd")
            engine.build_index(articles_df, text_column="text_for_embedding")
    else:
        engine.build_index(articles_df, text_column="text_for_embedding")

    if engine.index is None:
        raise ValueError("Semantic index build failed.")

    faiss.write_index(engine.index, str(output_dir / "semantic_faiss_index.bin"))
    pd.DataFrame({"article_id": engine.article_ids}).to_csv(output_dir / "article_ids.csv", index=False)

    with open(output_dir / SEMANTIC_BACKEND_METADATA, "w", encoding="utf-8") as handle:
        json.dump({"backend": engine.backend, "model_name": model_ref}, handle, indent=2)

    if engine.backend == "tfidf-svd":
        with open(output_dir / SEMANTIC_VECTORIZER_ARTIFACT, "wb") as handle:
            pickle.dump(engine.vectorizer, handle)
        if engine.projector is not None:
            with open(output_dir / SEMANTIC_PROJECTOR_ARTIFACT, "wb") as handle:
                pickle.dump(engine.projector, handle)

    print(f"Saved semantic index artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
