from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from .data import load_reviews
from .modeling import (
    RANDOM_STATE,
    build_count_baseline,
    build_preprocessed_count_baseline,
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reviews, dataset_info = load_reviews(
        args.csv_path,
        sample_size=args.sample_size,
        random_state=RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        reviews["review"],
        reviews["sentiment"],
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=reviews["sentiment"],
    )

    rows = []
    reports = {}
    for name, model in {
        "raw_bag_of_words": build_count_baseline(args.classifier),
        "preprocessed_bag_of_words": build_preprocessed_count_baseline(args.classifier),
    }.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rows.append(
            {
                "version": name,
                "classifier": args.classifier,
                "accuracy": accuracy_score(y_test, predictions),
                "f1_macro": f1_score(y_test, predictions, average="macro"),
            }
        )
        reports[name] = classification_report(y_test, predictions, zero_division=0)

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "preprocessing_comparison.csv", index=False)

    delta = (
        results.loc[results["version"] == "preprocessed_bag_of_words", "accuracy"].iloc[0]
        - results.loc[results["version"] == "raw_bag_of_words", "accuracy"].iloc[0]
    )
    conclusion = (
        f"Preprocessing changed accuracy by {delta:+.4f} on this evaluation split."
    )

    (output_dir / "classification_reports.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    (output_dir / "conclusion.txt").write_text(conclusion + "\n", encoding="utf-8")

    print("Part 2 preprocessing comparison complete")
    print(f"Dataset: {dataset_info.csv_path} ({dataset_info.used_rows} rows)")
    print(results.to_string(index=False))
    print(conclusion)
    print(f"Outputs: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 2 preprocessing comparison.")
    parser.add_argument("--csv-path", default="data/sample_reviews.csv")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--classifier",
        choices=("logistic_regression", "naive_bayes"),
        default="logistic_regression",
    )
    parser.add_argument("--output-dir", default="outputs/part2")
    return parser.parse_args()


if __name__ == "__main__":
    main()
