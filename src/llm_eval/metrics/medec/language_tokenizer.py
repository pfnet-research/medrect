"""Language-specific tokenizers for ROUGE evaluation."""

from typing import List

import MeCab
from rouge_score.tokenizers import Tokenizer


class JapaneseTokenizer(Tokenizer):
    """MeCab-based tokenizer for Japanese text in rouge-score library.

    Args:
        use_stemmer (bool): If True, uses word stems for tokenization. Defaults to False.
    """

    def __init__(self, use_stemmer: bool = False):
        self._stemmer = use_stemmer
        self.tagger = MeCab.Tagger()
        self.wakati = MeCab.Tagger("-Owakati")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text.

        Args:
            text: Text to be tokenized

        Returns:
            List of tokens
        """
        if not text or not text.strip():
            return []

        if self._stemmer:
            # Use detailed analysis with word stems
            node = self.tagger.parseToNode(text)
            tokens = []
            while node:
                if node.surface:  # Exclude empty strings
                    feature = node.feature.split(",")
                    if len(feature) > 6 and feature[6] != "*":
                        # Use word stem
                        token = feature[6]
                    else:
                        # Use surface form when stem is not available
                        token = node.surface
                    tokens.append(token)
                node = node.next
            return tokens
        else:
            # Simple word segmentation
            wakati_result = self.wakati.parse(text)
            if wakati_result:
                wakati_result = wakati_result.strip()
                return wakati_result.split() if wakati_result else []
            return []


class CharacterTokenizer(Tokenizer):
    """Character-level tokenizer for backward compatibility."""

    def tokenize(self, text: str) -> List[str]:
        """Tokenize at character level."""
        if not text or not text.strip():
            return []
        return list(text.strip())


def get_tokenizer(method: str = "mecab", **kwargs) -> Tokenizer:
    """Get tokenizer instance by specified method.

    Args:
        method: One of "mecab", "char", or "none"
        **kwargs: Parameters passed to the tokenizer

    Returns:
        Tokenizer instance
    """
    if method == "mecab":
        return JapaneseTokenizer(**kwargs)
    elif method == "char":
        return CharacterTokenizer()
    elif method == "none":
        from rouge_score.tokenizers import DefaultTokenizer

        return DefaultTokenizer()
    else:
        raise ValueError(f"Unknown tokenization method: {method}")


def get_tokenizer_for_language(
    language: str, method: str = "auto", **kwargs
) -> Tokenizer:
    """Get tokenizer appropriate for the specified language.

    Args:
        language: Language code ('ja', 'en', 'unknown')
        method: Tokenization method. If "auto", selects based on language.
        **kwargs: Parameters passed to the tokenizer

    Returns:
        Tokenizer instance appropriate for the language
    """
    if method == "auto":
        # Automatic selection based on language
        if language == "ja":
            return JapaneseTokenizer(**kwargs)
        elif language == "en":
            from rouge_score.tokenizers import DefaultTokenizer

            return DefaultTokenizer()
        else:
            # Default to character-level for unknown languages
            return CharacterTokenizer()
    else:
        # Explicitly specified method
        return get_tokenizer(method, **kwargs)
