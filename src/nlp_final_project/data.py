from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TEXT_COLUMNS = ("review", "text", "content", "sentence")
LABEL_COLUMNS = ("sentiment", "label", "target", "category")
POSITIVE_LABELS = {"positive", "pos", "1", "true"}
NEGATIVE_LABELS = {"negative", "neg", "0", "false"}


@dataclass(frozen=True)
class DatasetInfo:
    csv_path: Path
    text_column: str
    label_column: str
    used_rows: int


def load_reviews(
    csv_path: str | Path,
    sample_size: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, DatasetInfo]:
    """Load a binary sentiment dataset and normalize its schema."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Add a CSV file or use data/sample_reviews.csv."
        )

    raw = pd.read_csv(path)
    text_column = _find_column(raw, TEXT_COLUMNS, "review text")
    label_column = _find_column(raw, LABEL_COLUMNS, "label")

    df = raw[[text_column, label_column]].copy()
    df.columns = ["review", "sentiment"]
    df["review"] = df["review"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].map(_normalize_label)
    df = df.dropna(subset=["review", "sentiment"])
    df = df[df["review"] != ""].reset_index(drop=True)

    if df["sentiment"].nunique() != 2:
        raise ValueError("Expected exactly two labels after normalization.")

    if sample_size is not None and sample_size > 0 and len(df) > sample_size:
        df = (
            df.groupby("sentiment", group_keys=False)
            .apply(
                lambda group: group.sample(
                    n=max(1, round(sample_size * len(group) / len(df))),
                    random_state=random_state,
                )
            )
            .sample(frac=1, random_state=random_state)
            .head(sample_size)
            .reset_index(drop=True)
        )

    info = DatasetInfo(
        csv_path=path,
        text_column=text_column,
        label_column=label_column,
        used_rows=len(df),
    )
    return df, info


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...], purpose: str) -> str:
    lower_to_actual = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lower_to_actual:
            return lower_to_actual[candidate]
    raise ValueError(
        f"Could not find a {purpose} column. Tried: {', '.join(candidates)}."
    )


def _normalize_label(value: object) -> str | None:
    label = str(value).strip().lower()
    if label in POSITIVE_LABELS:
        return "positive"
    if label in NEGATIVE_LABELS:
        return "negative"
    return None
