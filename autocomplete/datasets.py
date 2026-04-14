import random
from pathlib import Path
from typing import List, Sequence, Tuple

from autocomplete.preprocess import split_to_text_units, tokenize_corpus


DEFAULT_RANDOM_SEED = 87


def load_corpus_text(data_path: Path) -> str:
    return data_path.read_text(encoding="utf-8")


def load_corpus_rows(data_path: Path) -> List[str]:
    data = load_corpus_text(data_path)
    return split_to_text_units(data)


def load_tokenized_sentences(data_path: Path) -> List[List[str]]:
    data = load_corpus_text(data_path)
    return tokenize_corpus(data)


def split_train_test(
    tokenized_sentences: Sequence[Sequence[str]],
    train_fraction: float = 0.8,
    shuffle: bool = True,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Tuple[List[List[str]], List[List[str]]]:
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be between 0 and 1 (inclusive of 1).")

    tokenized = [list(sentence) for sentence in tokenized_sentences]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(tokenized)

    train_size = int(len(tokenized) * train_fraction)
    train_data = tokenized[:train_size]
    test_data = tokenized[train_size:]
    return train_data, test_data


def load_train_test_split(
    data_path: Path,
    train_fraction: float = 0.8,
    shuffle: bool = True,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Tuple[List[List[str]], List[List[str]]]:
    tokenized = load_tokenized_sentences(data_path)
    return split_train_test(
        tokenized_sentences=tokenized,
        train_fraction=train_fraction,
        shuffle=shuffle,
        seed=seed,
    )
