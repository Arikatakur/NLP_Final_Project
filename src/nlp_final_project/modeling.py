from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from .features import ReviewLengthFeatures
from .text_processing import preprocess


RANDOM_STATE = 42


def build_count_baseline(classifier: str = "logistic_regression") -> Pipeline:
    estimator = _build_classifier(classifier)
    return Pipeline(
        steps=[
            ("vectorizer", CountVectorizer()),
            ("classifier", estimator),
        ]
    )


def build_preprocessed_count_baseline(classifier: str = "logistic_regression") -> Pipeline:
    estimator = _build_classifier(classifier)
    return Pipeline(
        steps=[
            ("vectorizer", CountVectorizer(preprocessor=preprocess)),
            ("classifier", estimator),
        ]
    )


def build_tfidf_feature_pipeline(
    classifier: str = "logistic_regression",
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_features: int | None = 20000,
) -> Pipeline:
    estimator = _build_classifier(classifier)
    return Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        (
                            "tfidf",
                            TfidfVectorizer(
                                ngram_range=ngram_range,
                                min_df=min_df,
                                max_features=max_features,
                                sublinear_tf=True,
                            ),
                        ),
                        ("length", ReviewLengthFeatures()),
                    ]
                ),
            ),
            ("scale", MaxAbsScaler()),
            ("classifier", estimator),
        ]
    )


def _build_classifier(name: str):
    if name == "naive_bayes":
        return MultinomialNB()
    if name == "svm":
        return LinearSVC(random_state=RANDOM_STATE)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown classifier: {name}")
