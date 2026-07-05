from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from .data import load_reviews
from .modeling import RANDOM_STATE, build_tfidf_feature_pipeline


CLASSIFIERS = (
    "naive_bayes",
    "logistic_regression",
    "svm",
    "knn",
    "random_forest",
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

    cv = build_cv(reviews["sentiment"], args.cv_folds)
    rows = []
    for classifier in CLASSIFIERS:
        model = build_tfidf_feature_pipeline(
            classifier=classifier,
            ngram_range=(1, args.max_ngram),
            max_features=args.max_features,
        )
        scores = cross_validate(
            model,
            reviews["review"],
            reviews["sentiment"],
            cv=cv,
            scoring=build_scorers(),
            error_score="raise",
        )
        rows.append(
            {
                "classifier": classifier,
                "accuracy_mean": scores["test_accuracy"].mean(),
                "accuracy_std": scores["test_accuracy"].std(),
                "precision_macro_mean": scores["test_precision_macro"].mean(),
                "recall_macro_mean": scores["test_recall_macro"].mean(),
                "f1_macro_mean": scores["test_f1_macro"].mean(),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        by=["f1_macro_mean", "accuracy_mean"], ascending=False
    )
    results.to_csv(output_dir / "classifier_comparison.csv", index=False)

    best_classifier = results.iloc[0]["classifier"]
    best_model = build_tfidf_feature_pipeline(
        classifier=best_classifier,
        ngram_range=(1, args.max_ngram),
        max_features=args.max_features,
    )
    predictions = cross_val_predict(
        best_model,
        reviews["review"],
        reviews["sentiment"],
        cv=cv,
    )
    labels = ["negative", "positive"]
    matrix = confusion_matrix(reviews["sentiment"], predictions, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / "best_confusion_matrix.csv"
    )
    save_confusion_matrix(matrix, labels, output_dir / "best_confusion_matrix.png")

    explanation = {
        "part": 3,
        "dataset_path": str(dataset_info.csv_path),
        "dataset_rows": dataset_info.used_rows,
        "cv_folds": cv.get_n_splits(),
        "features": "TF-IDF unigrams/bigrams plus review length features",
        "best_classifier": best_classifier,
        "selection_rule": "highest mean macro-F1, then highest mean accuracy",
        "note": (
            "Random Forest is the additional independently researched ML algorithm "
            "included beyond the course classifiers."
        ),
    }
    (output_dir / "part3_summary.json").write_text(
        json.dumps(explanation, indent=2), encoding="utf-8"
    )

    print("Part 3 classifier comparison complete")
    print(f"Dataset: {dataset_info.csv_path} ({dataset_info.used_rows} rows)")
    print(f"CV folds: {cv.get_n_splits()}")
    print(results.to_string(index=False))
    print(f"Best classifier: {best_classifier}")
    print(f"Outputs: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 3 classifier comparison.")
    parser.add_argument("--csv-path", default="data/sample_reviews.csv")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-ngram", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--output-dir", default="outputs/part3")
    return parser.parse_args()


def build_cv(labels: pd.Series, requested_folds: int) -> StratifiedKFold:
    min_class_count = labels.value_counts().min()
    if min_class_count < 2:
        raise ValueError("Cross-validation requires at least two samples per class.")
    folds = min(requested_folds, int(min_class_count))
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)


def build_scorers() -> dict:
    return {
        "accuracy": "accuracy",
        "precision_macro": make_scorer(
            precision_score, average="macro", zero_division=0
        ),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }


def save_confusion_matrix(matrix, labels: list[str], output_path: Path) -> None:
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Best Classifier Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
