from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import pandas as pd


def main() -> None:
    parser = ArgumentParser(description="Generate a comparison table from metrics JSON files.")
    parser.add_argument("metrics", nargs="+", help="List of metrics JSON files")
    parser.add_argument("--output", default="paper/results_table.md")
    args = parser.parse_args()

    rows = []
    for metrics_path in args.metrics:
        payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        rows.append(
            {
                "Model": payload.get("Model", Path(metrics_path).stem),
                "Recall@10": payload.get("Recall@10", 0.0),
                "NDCG@10": payload.get("NDCG@10", 0.0),
                "Recall@20": payload.get("Recall@20", 0.0),
                "Latency (ms)": payload.get("LatencyMs", 0.0),
            }
        )

    table = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table.to_markdown(index=False), encoding="utf-8")
    print(output_path.resolve())


if __name__ == "__main__":
    main()

