"""
Data utilities for character-level modeling of RxNorm drug names.

This module reads `rxnorm-names.csv`, normalizes strings to NFC, constructs a
character vocabulary with start-of-sequence ("^") and end-of-sequence ("$")
tokens, and provides lightweight helpers to encode contexts and build training
pairs for next-character prediction.
"""

import unicodedata
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class ContextPair:
    """Immutable training example for next-character prediction.

    Attributes:
        context (List[int]): Fixed-length list of character ids representing the
            preceding characters (left context) of the target position.
        target (int): Integer id of the next character to be predicted.
    """

    context: List[int]
    target: int


@dataclass(frozen=True)
class RxNormDataConfig:
    """Container for names and vocabulary mappings derived from RxNorm.

    Attributes:
        names (List[str]): Normalized RxNorm strings.
        vocab (List[str]): All characters used, including special tokens "^" and "$".
        char_to_id (Dict[str, int]): Mapping from character to integer id.
        id_to_char (Dict[int, str]): Reverse mapping from integer id back to character.
    """

    names: List[str]
    vocab: List[str]
    char_to_id: Dict[str, int]
    id_to_char: Dict[int, str]

    def get_pair(self, target: str, context: List[int]) -> ContextPair:
        """Create a ContextPair by converting a target character to its id.

        Args:
            target (str): Single-character string to predict (e.g., "a" or "$").
            context (List[int]): Encoded left context as a list of character ids.

        Returns:
            ContextPair: Pair with the same context and the numeric target id.

        Raises:
            KeyError: If the target character is not present in `char_to_id`.
        """
        target = self.char_to_id[target]
        return ContextPair(context=context, target=target)

    def encode(self, context: str) -> List[int]:
        """Encode a string into a list of character ids.

        Args:
            context (str): String to encode.

        Returns:
            List[int]: Integer ids corresponding to the characters in `context`.

        Raises:
            KeyError: If any character is not present in `char_to_id`.
        """
        return [self.char_to_id[char] for char in context]


def load_rxnorm_data() -> RxNormDataConfig:
    """Load, normalize, and index RxNorm names from `rxnorm-names.csv`.

    The CSV is expected to have a column named `STR`. Each string is stripped
    of leading/trailing whitespace and normalized to Unicode NFC form to ensure
    consistent character comparisons. A character vocabulary is then created,
    prefixed with two special tokens:
      - "^": start-of-sequence token used to left-pad contexts
      - "$": end-of-sequence token used as the final target after a name

    Returns:
        RxNormDataConfig: Names, vocabulary, and mapping dictionaries.
    """

    # Read CSV with expected RxNorm string column `STR` and normalize to NFC.
    data = pd.read_csv("rxnorm-names.csv")
    names = data.STR.apply(lambda s: unicodedata.normalize("NFC", s.strip())).tolist()

    # Collect all unique characters present across names (excluding specials).
    chars = sorted(set(char for name in names for char in name))

    # Special tokens: start-of-sequence and end-of-sequence.
    specials = ["^", "$"]

    vocab = specials + chars
    char_to_id = {c: i for i, c in enumerate(vocab)}
    id_to_char = {i: c for c, i in char_to_id.items()}
    return RxNormDataConfig(
        names=names,
        vocab=vocab,
        char_to_id=char_to_id,
        id_to_char=id_to_char,
    )
