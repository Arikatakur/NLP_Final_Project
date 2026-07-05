from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


class ReviewLengthFeatures(BaseEstimator, TransformerMixin):
    """Extract small numeric features from raw review text."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            value = str(text)
            words = value.split()
            word_count = len(words)
            char_count = len(value)
            avg_word_length = char_count / max(word_count, 1)
            rows.append(
                [
                    char_count,
                    word_count,
                    avg_word_length,
                    value.count("!"),
                    value.count("?"),
                ]
            )
        return sparse.csr_matrix(np.asarray(rows, dtype=float))
