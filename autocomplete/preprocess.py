import re
from typing import List

import nltk


_WHITESPACE_RE = re.compile(r"\s+")


def ensure_nltk_tokenizer() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def normalize_text(text: str) -> str:
    normalized = text.lower()
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def tokenize(text: str) -> List[str]:
    ensure_nltk_tokenizer()
    normalized = normalize_text(text)
    if not normalized:
        return []
    return nltk.word_tokenize(normalized)


def split_to_text_units(data: str) -> List[str]:
    lines = [line.strip() for line in data.splitlines()]
    return [line for line in lines if line]


def tokenize_corpus(data: str) -> List[List[str]]:
    return [tokenize(line) for line in split_to_text_units(data)]
