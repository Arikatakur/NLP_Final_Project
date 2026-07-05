from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .data import load_reviews
from .modeling import (
    RANDOM_STATE,
    build_count_baseline,
    build_tfidf_feature_pipeline,
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

    comparison_rows = []
    baseline = build_count_baseline("logistic_regression")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    comparison_rows.append(
        metric_row("part1_bow_baseline", y_test, baseline_predictions, "completed")
    )

    classical = build_tfidf_feature_pipeline("logistic_regression", ngram_range=(1, 2))
    classical.fit(X_train, y_train)
    classical_predictions = classical.predict(X_test)
    comparison_rows.append(
        metric_row("part4_best_classical", y_test, classical_predictions, "completed")
    )

    pretrained_status = "not_started"
    pretrained_error = ""
    pretrained_predictions = []
    pretrained_actual = []
    pretrained_texts = []
    if args.skip_pretrained:
        pretrained_status = "skipped"
        comparison_rows.append(
            {
                "version": "part5_pretrained_huggingface",
                "status": pretrained_status,
                "evaluated_rows": 0,
                "accuracy": "",
                "f1_macro": "",
                "note": "Skipped by --skip-pretrained.",
            }
        )
    else:
        evaluation = limit_evaluation_rows(X_test, y_test, args.max_pretrained_samples)
        pretrained_texts = evaluation["review"].tolist()
        pretrained_actual = evaluation["sentiment"].tolist()
        try:
            pretrained_predictions = run_pretrained(
                pretrained_texts,
                model_name=args.model_name,
                batch_size=args.batch_size,
            )
            pretrained_status = "completed"
            comparison_rows.append(
                metric_row(
                    "part5_pretrained_huggingface",
                    pretrained_actual,
                    pretrained_predictions,
                    pretrained_status,
                    evaluated_rows=len(pretrained_actual),
                )
            )
        except Exception as exc:
            pretrained_status = "failed"
            pretrained_error = f"{type(exc).__name__}: {exc}"
            comparison_rows.append(
                {
                    "version": "part5_pretrained_huggingface",
                    "status": pretrained_status,
                    "evaluated_rows": 0,
                    "accuracy": "",
                    "f1_macro": "",
                    "note": pretrained_error,
                }
            )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "final_comparison.csv", index=False)

    recheck = build_error_recheck(
        output_dir=output_dir,
        project_root=Path.cwd(),
        classical_texts=X_test,
        classical_actual=y_test,
        classical_predictions=classical_predictions,
        pretrained_status=pretrained_status,
        pretrained_model_name=args.model_name,
    )
    recheck.to_csv(output_dir / "part4_error_recheck.csv", index=False)

    if pretrained_status == "completed" and not recheck.empty:
        raw_predictions = run_pretrained(
            recheck["review"].astype(str).tolist(),
            model_name=args.model_name,
            batch_size=args.batch_size,
        )
        recheck["pretrained_prediction"] = raw_predictions
        recheck["pretrained_solved_classical_error"] = (
            recheck["pretrained_prediction"] == recheck["actual"]
        )
        recheck.to_csv(output_dir / "part4_error_recheck.csv", index=False)

    summary = {
        "part": 5,
        "dataset_path": str(dataset_info.csv_path),
        "dataset_rows": dataset_info.used_rows,
        "pretrained_model": args.model_name,
        "pretrained_status": pretrained_status,
        "pretrained_error": pretrained_error,
        "raw_text_requirement": "The pretrained model receives original raw text.",
        "tradeoff_summary": (
            "The pretrained model can capture context and word order better than "
            "Bag-of-Words or TF-IDF models, but it costs more runtime, memory, "
            "dependency weight, and is less transparent."
        ),
    }
    (output_dir / "part5_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Part 5 pretrained comparison complete")
    print(f"Dataset: {dataset_info.csv_path} ({dataset_info.used_rows} rows)")
    print(comparison.to_string(index=False))
    if pretrained_status == "failed":
        print(f"Pretrained model did not run: {pretrained_error}")
    print(f"Outputs: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 5 pretrained comparison.")
    parser.add_argument("--csv-path", default="data/sample_reviews.csv")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased-finetuned-sst-2-english",
    )
    parser.add_argument("--max-pretrained-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-pretrained", action="store_true")
    parser.add_argument("--output-dir", default="outputs/part5")
    return parser.parse_args()


def metric_row(
    version: str,
    actual,
    predicted,
    status: str,
    evaluated_rows: int | None = None,
) -> dict:
    if evaluated_rows is None:
        evaluated_rows = len(actual)
    return {
        "version": version,
        "status": status,
        "evaluated_rows": evaluated_rows,
        "accuracy": accuracy_score(actual, predicted),
        "f1_macro": f1_score(actual, predicted, average="macro", zero_division=0),
        "note": "",
    }


def limit_evaluation_rows(
    texts: pd.Series,
    labels: pd.Series,
    max_rows: int,
) -> pd.DataFrame:
    evaluation = pd.DataFrame({"review": texts, "sentiment": labels}).reset_index(
        drop=True
    )
    if max_rows > 0 and len(evaluation) > max_rows:
        sampled_groups = []
        for _, group in evaluation.groupby("sentiment", sort=False):
            group_size = max(1, round(max_rows * len(group) / len(evaluation)))
            sampled_groups.append(
                group.sample(n=min(group_size, len(group)), random_state=RANDOM_STATE)
            )
        evaluation = pd.concat(sampled_groups, ignore_index=True)
        evaluation = evaluation.sample(frac=1, random_state=RANDOM_STATE).head(max_rows)
        evaluation = evaluation[["review", "sentiment"]].reset_index(drop=True)
    return evaluation


def run_pretrained(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> list[str]:
    from transformers import pipeline

    classifier = pipeline("sentiment-analysis", model=model_name)
    outputs = classifier(texts, truncation=True, batch_size=batch_size)
    return [normalize_pretrained_label(output["label"]) for output in outputs]


def normalize_pretrained_label(label: str) -> str:
    value = label.strip().lower()
    if value in {"positive", "pos", "label_1", "1"}:
        return "positive"
    if value in {"negative", "neg", "label_0", "0"}:
        return "negative"
    raise ValueError(f"Unsupported pretrained label: {label}")


def build_error_recheck(
    output_dir: Path,
    project_root: Path,
    classical_texts: pd.Series,
    classical_actual: pd.Series,
    classical_predictions,
    pretrained_status: str,
    pretrained_model_name: str,
) -> pd.DataFrame:
    part4_errors = project_root / "outputs" / "part4" / "error_analysis.csv"
    if part4_errors.exists():
        errors = pd.read_csv(part4_errors)
        if "review" in errors.columns and "actual" in errors.columns:
            errors = errors[["review", "actual"]].dropna().head(5).copy()
        else:
            errors = pd.DataFrame()
    else:
        rows = []
        for review, actual, predicted in zip(
            classical_texts, classical_actual, classical_predictions
        ):
            if actual == predicted:
                continue
            rows.append({"review": review, "actual": actual})
            if len(rows) >= 5:
                break
        errors = pd.DataFrame(rows)

    if errors.empty:
        errors = pd.DataFrame(
            [
                {
                    "review": "",
                    "actual": "",
                    "note": "No classical errors available yet; run Part 4 on real data.",
                }
            ]
        )

    errors["pretrained_model"] = pretrained_model_name
    errors["pretrained_status"] = pretrained_status
    return errors


if __name__ == "__main__":
    main()
