from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .data import load_reviews
from .modeling import RANDOM_STATE, build_tfidf_feature_pipeline
from .part3_classifier_comparison import build_scorers


LENGTH_FEATURE_NAMES = [
    "char_count",
    "word_count",
    "avg_word_length",
    "exclamation_count",
    "question_count",
]


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

    variants = build_variants(args.classifier)
    rows = []
    for variant in variants:
        model = build_tfidf_feature_pipeline(
            classifier=args.classifier,
            ngram_range=variant["ngram_range"],
            max_features=variant["max_features"],
        )
        if variant["params"]:
            model.set_params(**variant["params"])
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
                "variant": variant["name"],
                "change": variant["change"],
                "accuracy_mean": scores["test_accuracy"].mean(),
                "accuracy_std": scores["test_accuracy"].std(),
                "f1_macro_mean": scores["test_f1_macro"].mean(),
                "f1_macro_std": scores["test_f1_macro"].std(),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        by=["f1_macro_mean", "accuracy_mean"], ascending=False
    )
    comparison.to_csv(output_dir / "improvement_table.csv", index=False)

    best_variant = next(
        variant for variant in variants if variant["name"] == comparison.iloc[0]["variant"]
    )
    best_model = build_tfidf_feature_pipeline(
        classifier=args.classifier,
        ngram_range=best_variant["ngram_range"],
        max_features=best_variant["max_features"],
    )
    if best_variant["params"]:
        best_model.set_params(**best_variant["params"])

    X_train, X_test, y_train, y_test = train_test_split(
        reviews["review"],
        reviews["sentiment"],
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=reviews["sentiment"],
    )
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    errors = build_error_analysis(X_test, y_test, predictions, limit=args.error_limit)
    errors.to_csv(output_dir / "error_analysis.csv", index=False)

    influential = extract_influential_features(best_model, top_n=args.top_n)
    influential.to_csv(output_dir / "influential_words.csv", index=False)

    holdout_metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
    }
    summary = {
        "part": 4,
        "dataset_path": str(dataset_info.csv_path),
        "dataset_rows": dataset_info.used_rows,
        "classifier": args.classifier,
        "cv_folds": cv.get_n_splits(),
        "best_variant": best_variant,
        "holdout_metrics_for_error_analysis": holdout_metrics,
        "classical_limit_conclusion": (
            "Classical TF-IDF models are transparent and fast, but they mostly rely "
            "on local word/phrase evidence and often miss context, sarcasm, negation "
            "scope, and mixed sentiment."
        ),
    }
    (output_dir / "part4_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Part 4 improvement and analysis complete")
    print(f"Dataset: {dataset_info.csv_path} ({dataset_info.used_rows} rows)")
    print(comparison.to_string(index=False))
    print(f"Best variant: {best_variant['name']}")
    print(f"Error rows written: {len(errors)}")
    print(f"Outputs: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 4 tuning and error analysis.")
    parser.add_argument("--csv-path", default="data/sample_reviews.csv")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--classifier",
        choices=("logistic_regression", "svm", "naive_bayes"),
        default="logistic_regression",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--error-limit", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/part4")
    return parser.parse_args()


def build_cv(labels: pd.Series, requested_folds: int) -> StratifiedKFold:
    min_class_count = labels.value_counts().min()
    if min_class_count < 2:
        raise ValueError("Cross-validation requires at least two samples per class.")
    return StratifiedKFold(
        n_splits=min(requested_folds, int(min_class_count)),
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def build_variants(classifier: str) -> list[dict]:
    variants = [
        {
            "name": "tfidf_unigrams",
            "change": "TF-IDF unigrams plus length features",
            "ngram_range": (1, 1),
            "max_features": 20000,
            "params": {},
        },
        {
            "name": "tfidf_bigrams",
            "change": "Add bigrams",
            "ngram_range": (1, 2),
            "max_features": 20000,
            "params": {},
        },
        {
            "name": "tfidf_bigrams_limited",
            "change": "Add bigrams with lower max_features",
            "ngram_range": (1, 2),
            "max_features": 5000,
            "params": {},
        },
    ]

    if classifier in {"logistic_regression", "svm"}:
        variants.append(
            {
                "name": "tfidf_bigrams_tuned_c",
                "change": "Add bigrams and increase regularization strength C",
                "ngram_range": (1, 2),
                "max_features": 20000,
                "params": {"classifier__C": 2.0},
            }
        )
    elif classifier == "naive_bayes":
        variants.append(
            {
                "name": "tfidf_bigrams_tuned_alpha",
                "change": "Add bigrams and tune Naive Bayes alpha",
                "ngram_range": (1, 2),
                "max_features": 20000,
                "params": {"classifier__alpha": 0.5},
            }
        )

    return variants


def build_error_analysis(
    reviews: pd.Series,
    actual: pd.Series,
    predicted,
    limit: int,
) -> pd.DataFrame:
    rows = []
    for review, y_true, y_pred in zip(reviews, actual, predicted):
        if y_true == y_pred:
            continue
        rows.append(
            {
                "review": review,
                "actual": y_true,
                "predicted": y_pred,
                "likely_pattern": infer_error_pattern(str(review)),
                "analysis_note": "Review manually: identify why the model missed it.",
            }
        )
        if len(rows) >= limit:
            break

    if not rows:
        rows.append(
            {
                "review": "",
                "actual": "",
                "predicted": "",
                "likely_pattern": "no holdout errors found",
                "analysis_note": (
                    "Use the real dataset or add manually written challenge examples "
                    "to complete the required five analyzed errors."
                ),
            }
        )
    return pd.DataFrame(rows)


def infer_error_pattern(review: str) -> str:
    lowered = review.lower()
    if any(marker in lowered for marker in ("not ", "never ", "no ")):
        return "negation"
    if any(marker in lowered for marker in ("but", "however", "although", "though")):
        return "mixed sentiment / contrast"
    if "!" in lowered or "?" in lowered:
        return "emphasis or rhetorical tone"
    return "word evidence may be weak or misleading"


def extract_influential_features(model, top_n: int) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    feature_union = model.named_steps["features"]
    tfidf = feature_union.transformer_list[0][1]
    names = list(tfidf.get_feature_names_out()) + LENGTH_FEATURE_NAMES

    if not hasattr(classifier, "coef_"):
        return pd.DataFrame(
            [
                {
                    "category": "not_available",
                    "feature": "",
                    "weight": "",
                    "note": "Feature weights are only available for linear models.",
                }
            ]
        )

    weights = classifier.coef_[0]
    negative_indexes = weights.argsort()[:top_n]
    positive_indexes = weights.argsort()[-top_n:][::-1]
    rows = []
    for index in negative_indexes:
        rows.append(
            {
                "category": "negative",
                "feature": names[index],
                "weight": weights[index],
            }
        )
    for index in positive_indexes:
        rows.append(
            {
                "category": "positive",
                "feature": names[index],
                "weight": weights[index],
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
