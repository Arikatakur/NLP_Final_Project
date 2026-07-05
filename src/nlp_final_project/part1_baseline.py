from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .data import load_reviews


RANDOM_STATE = 42


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reviews, dataset_info = load_reviews(
        args.csv_path,
        sample_size=args.sample_size,
        random_state=RANDOM_STATE,
    )

    summary = summarize_dataset(reviews)
    summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    save_category_distribution(reviews, output_dir / "category_distribution.png")

    model = build_baseline(args.classifier)
    X_train, X_test, y_train, y_test = train_test_split(
        reviews["review"],
        reviews["sentiment"],
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=reviews["sentiment"],
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "part": 1,
        "dataset_path": str(dataset_info.csv_path),
        "dataset_rows": dataset_info.used_rows,
        "classifier": args.classifier,
        "vectorizer": "Bag-of-Words CountVectorizer",
        "test_size": args.test_size,
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(y_test, predictions, average="macro"),
        "random_state": RANDOM_STATE,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    report = classification_report(y_test, predictions)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    print("Part 1 baseline complete")
    print(f"Dataset: {dataset_info.csv_path} ({dataset_info.used_rows} rows)")
    print(f"Classifier: {args.classifier}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")
    print(f"Outputs: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 1 Bag-of-Words baseline.")
    parser.add_argument(
        "--csv-path",
        default="data/sample_reviews.csv",
        help="Path to CSV containing review text and sentiment labels.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size, useful for 3000-5000 IMDB rows.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Train/test split size for the Part 1 baseline.",
    )
    parser.add_argument(
        "--classifier",
        choices=("logistic_regression", "naive_bayes"),
        default="logistic_regression",
        help="Simple baseline classifier.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/part1",
        help="Directory for Part 1 outputs.",
    )
    return parser.parse_args()


def summarize_dataset(reviews: pd.DataFrame) -> pd.DataFrame:
    enriched = reviews.copy()
    enriched["char_length"] = enriched["review"].str.len()
    enriched["word_count"] = enriched["review"].str.split().str.len()

    label_counts = (
        enriched["sentiment"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )
    label_counts["percent"] = label_counts["count"] / len(enriched)

    length_summary = pd.DataFrame(
        [
            {
                "sentiment": "all",
                "count": len(enriched),
                "avg_chars": enriched["char_length"].mean(),
                "avg_words": enriched["word_count"].mean(),
                "min_words": enriched["word_count"].min(),
                "max_words": enriched["word_count"].max(),
            }
        ]
    )

    return label_counts.merge(length_summary, how="cross")


def save_category_distribution(reviews: pd.DataFrame, output_path: Path) -> None:
    counts = reviews["sentiment"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["#d95f02", "#1b9e77"])
    ax.set_title("Category Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of reviews")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_baseline(classifier: str) -> Pipeline:
    if classifier == "naive_bayes":
        estimator = MultinomialNB()
    else:
        estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    return Pipeline(
        steps=[
            ("vectorizer", CountVectorizer()),
            ("classifier", estimator),
        ]
    )


if __name__ == "__main__":
    main()
