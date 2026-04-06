from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import pandas as pd


def main() -> None:
    parser = ArgumentParser(description="Create an ablation summary from metrics JSON files.")
    parser.add_argument("metrics", nargs="+", help="Metric files such as semantic-only.json and hybrid.json")
    parser.add_argument("--output", default="paper/ablation_study.csv")
    args = parser.parse_args()

    rows = []
    for metrics_path in args.metrics:
        payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        rows.append(payload)

    frame = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(output_path.resolve())


if __name__ == "__main__":
    main()

