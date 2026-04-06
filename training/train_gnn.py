from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np

from models.lightgcn import LightGCN, bpr_loss, build_edge_index, sample_bpr_triplets
from training.data_utils import build_id_mappings, load_articles, load_transactions, temporal_split

try:
    import mlflow
except ImportError:  # pragma: no cover - optional at inspection time
    mlflow = None

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional at inspection time
    raise ImportError("torch is required to run train_gnn.py") from exc


def main() -> None:
    parser = ArgumentParser(description="Train LightGCN with BPR loss for fashion recommendation.")
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--articles", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-weeks", type=int, default=4)
    parser.add_argument("--test-days", type=int, default=None)
    parser.add_argument("--samples-per-user", type=int, default=2)
    args = parser.parse_args()

    articles_df = load_articles(args.articles)
    transactions_df = load_transactions(args.transactions)
    train_df, _ = temporal_split(transactions_df, test_weeks=args.test_weeks, test_days=args.test_days)
    user_mapping, item_mapping = build_id_mappings(train_df)

    edge_index = build_edge_index(train_df, user_mapping=user_mapping, item_mapping=item_mapping)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LightGCN(
        num_users=len(user_mapping),
        num_items=len(item_mapping),
        emb_dim=args.embedding_dim,
        n_layers=args.n_layers,
    ).to(device)
    edge_index = edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_context = mlflow.start_run(run_name=f"lightgcn-bpr-{args.embedding_dim}d") if mlflow else None

    if mlflow:
        mlflow.log_params(
            {
                "embedding_dim": args.embedding_dim,
                "n_layers": args.n_layers,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "loss": "bpr",
            }
        )

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            batch_count = 0

            for user_idx, pos_idx, neg_idx in sample_bpr_triplets(
                train_df,
                user_mapping=user_mapping,
                item_mapping=item_mapping,
                num_items=len(item_mapping),
                samples_per_user=args.samples_per_user,
            ):
                optimizer.zero_grad()
                user_embeddings, item_embeddings = model.split_embeddings(edge_index)

                user_batch = user_embeddings[torch.tensor([user_idx], dtype=torch.long, device=device)]
                pos_batch = item_embeddings[torch.tensor([pos_idx], dtype=torch.long, device=device)]
                neg_batch = item_embeddings[torch.tensor([neg_idx], dtype=torch.long, device=device)]

                loss = bpr_loss(user_batch, pos_batch, neg_batch, reg=args.weight_decay)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                batch_count += 1

            mean_loss = epoch_loss / max(batch_count, 1)
            print(f"epoch={epoch} loss={mean_loss:.6f}")
            if mlflow:
                mlflow.log_metric("loss", mean_loss, step=epoch)

        model.eval()
        with torch.no_grad():
            user_embeddings, item_embeddings = model.split_embeddings(edge_index)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "user_embeddings.npy", user_embeddings.cpu().numpy())
        np.save(output_dir / "item_embeddings.npy", item_embeddings.cpu().numpy())
        articles_df[["article_id"]].drop_duplicates().to_csv(output_dir / "article_ids.csv", index=False)

        with open(output_dir / "user_mapping.json", "w", encoding="utf-8") as handle:
            json.dump(user_mapping, handle, indent=2)
        with open(output_dir / "item_mapping.json", "w", encoding="utf-8") as handle:
            json.dump(item_mapping, handle, indent=2)

        train_df.to_csv(output_dir / "train_interactions.csv", index=False)
        print(f"Saved artifacts to {output_dir.resolve()}")

        if mlflow:
            mlflow.log_artifact(str(output_dir / "user_mapping.json"))
            mlflow.log_artifact(str(output_dir / "item_mapping.json"))
            mlflow.log_artifact(str(output_dir / "article_ids.csv"))
    finally:
        if run_context is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
