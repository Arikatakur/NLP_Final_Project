from __future__ import annotations

import re
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def preprocess(text: str, use_stemming: bool = True) -> str:
    """Normalize, tokenize, remove stop words, and optionally stem text."""
    tokens = TOKEN_RE.findall(str(text).lower())
    cleaned = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    if use_stemming:
        cleaned = [_stem(token) for token in cleaned]
    return " ".join(cleaned)


@lru_cache(maxsize=20000)
def _stem(token: str) -> str:
    try:
        from nltk.stem import PorterStemmer
    except ImportError:
        return _simple_suffix_stem(token)

    return PorterStemmer().stem(token)


def _simple_suffix_stem(token: str) -> str:
    """Small fallback stemmer so preprocessing remains runnable without NLTK."""
    for suffix in ("ingly", "edly", "ing", "edly", "ed", "ly", "ies", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            if suffix == "ies":
                return token[: -len(suffix)] + "y"
            return token[: -len(suffix)]
    return token
