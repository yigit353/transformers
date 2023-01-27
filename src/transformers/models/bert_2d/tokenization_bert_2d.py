# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for Bert."""

import collections
import math
import os
from typing import List, Optional, Tuple

import unicodedata

from ...tokenization_utils import _is_control, _is_punctuation, _is_whitespace, PreTrainedTokenizer
from ...utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "bert2d-base-uncased": "https://huggingface.co/bert2d-base-uncased/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bert2d-base-uncased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "bert2d-base-uncased": {"do_lower_case": False},
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def is_subword(token: str) -> bool:
    """Returns if a token is a subword"""
    return token.startswith("##")


class Bert2dTokenizer(PreTrainedTokenizer):
    r"""
    Construct a BERT tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask", "token_type_ids", "word_position_ids", "subword_ids"]

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            strip_accents=None,
            do_split_on_punc=True,
            max_intermediate_subword_positions_per_word=1,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="uniform",
            restart_new_sentence=False,
            **kwargs
    ):
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = Bert2dTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
                do_split_on_punc=do_split_on_punc
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)
        self.max_intermediate_subword_positions_per_word = max_intermediate_subword_positions_per_word
        self.subword_embedding_order = subword_embedding_order
        self.intermediate_subword_distribution_strategy = intermediate_subword_distribution_strategy
        self.restart_new_sentence = restart_new_sentence

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").replace("##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def create_word_ids_from_sequence(self, token_ids: List[int]) -> List[int]:
        """Creates word ids for given tokens"""

        tokens = self.convert_ids_to_tokens(token_ids)
        word_ids: List[int] = []
        current_word_id: int = -1
        sentence_restart = False
        restart_new_sentence = self.restart_new_sentence
        if tokens.count(self.sep_token) < 2:
            restart_new_sentence = False
        for token in tokens:
            # If new sentence requires new starting, do not restart at second
            if restart_new_sentence and not sentence_restart and token == self.sep_token:
                current_word_id = 0
                sentence_restart = True
            elif not is_subword(token):
                current_word_id += 1
            # If tokens start with a subword
            elif current_word_id == -1:
                current_word_id = 0
            word_ids.append(current_word_id)

        return word_ids

    def create_subword_ids_from_sequence(self, token_ids: List[int]) -> List[int]:
        """Creates subword ids for the given tokens and parameters"""

        # If tokens are empty return empty subword id list
        if len(token_ids) == 0:
            return []

        def _col_round(x: float) -> int:
            """Colloquial rounding where 0.5 rounds to 1"""
            frac = x - math.floor(x)
            if frac < 0.5:
                return math.floor(x)
            return math.ceil(x)

        def _get_uniform_id(si: int, max_intermediate_subwords: int, num_intermediate_subwords: int) -> int:
            """Calculates uniform id for the given subword index, si, and max and number of intermediate subwords"""
            return _col_round(si * (max_intermediate_subwords - 1) / num_intermediate_subwords)

        def _get_ids_from_subwords(_num_subwords: int, _max_intermediate_subword_positions_per_word: int,
                                   _subword_embedding_order: str, _intermediate_subword_distribution_strategy: str,
                                   _starts_with_subword=False) -> List[int]:
            """Calculate subword ids for given subwords list of a single word"""

            if _starts_with_subword:
                if _num_subwords == 1:
                    return [1]
            elif _num_subwords <= 2:
                return list(range(_num_subwords))

            if _subword_embedding_order == "ending_first":
                # Return with simple cases
                if _starts_with_subword and _num_subwords == 2:
                    return [2, 1]

                _subword_ids: List[int] = [0] if not _starts_with_subword else [1]
                # 2 for root and last token
                num_intermediate_subwords: int = _num_subwords - 2 if not _starts_with_subword else _num_subwords - 1

                # R - L - I1 - I2 - ...
                if num_intermediate_subwords <= _max_intermediate_subword_positions_per_word:
                    for si in range(num_intermediate_subwords):
                        _subword_ids.append(2 + si)
                # if there are more intermediate subwords than allowed
                else:
                    if _intermediate_subword_distribution_strategy == "uniform":
                        # Distribute all indices uniformly between allowed indices
                        for si in range(num_intermediate_subwords):
                            _subword_ids.append(
                                2 + _get_uniform_id(si, _max_intermediate_subword_positions_per_word,
                                                    num_intermediate_subwords))
                    elif _intermediate_subword_distribution_strategy == "leftover_as_last":
                        # Append subword indices that are allowed
                        for si in range(_max_intermediate_subword_positions_per_word):
                            _subword_ids.append(2 + si)
                        # Append rest as last
                        for si in range(num_intermediate_subwords - _max_intermediate_subword_positions_per_word):
                            _subword_ids.append(1)
                    else:
                        raise ValueError("Unsupported intermediate subword distribution strategy")
                _subword_ids.append(1)
                return _subword_ids
            raise ValueError("Unsupported subword embedding order")

        def _extend_subword_ids_for_word(_subword_ids: List[int],
                                         intermediate_subword_distribution_strategy: str,
                                         max_intermediate_subword_positions_per_word: int,
                                         _num_subwords: int,
                                         _start_subword_processed: bool,
                                         _starts_with_subword: bool, subword_embedding_order: str) -> None:
            """Extends subword ids for each word"""
            _subword_ids.extend(
                _get_ids_from_subwords(_num_subwords, max_intermediate_subword_positions_per_word,
                                       subword_embedding_order,
                                       intermediate_subword_distribution_strategy,
                                       _starts_with_subword and not _start_subword_processed)
            )

        tokens = self.convert_ids_to_tokens(token_ids)
        subword_ids: List[int] = []
        num_subwords: int = 0
        starts_with_subword = is_subword(tokens[0])
        start_subword_processed = False
        for token in tokens:
            if not is_subword(token):
                if num_subwords > 0:
                    _extend_subword_ids_for_word(
                        subword_ids, self.intermediate_subword_distribution_strategy,
                        self.max_intermediate_subword_positions_per_word, num_subwords,
                        start_subword_processed,
                        starts_with_subword, self.subword_embedding_order)
                    start_subword_processed = True
                    num_subwords = 0
            num_subwords += 1
        if num_subwords > 0:
            _extend_subword_ids_for_word(
                subword_ids, self.intermediate_subword_distribution_strategy,
                self.max_intermediate_subword_positions_per_word,
                num_subwords, start_subword_processed, starts_with_subword,
                self.subword_embedding_order)
        return subword_ids

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    """
    def __call__(self, *args, **kwargs):
        embeddings = super().__call__(*args, **kwargs)
        embeddings["word_position_ids"] = [
            self.create_word_ids_from_sequence(input_ids) for input_ids in
            embeddings["input_ids"]]
        embeddings["subword_ids"] = [
            self.create_subword_ids_from_sequence(input_ids) for input_ids in
            embeddings["input_ids"]]
        return embeddings
    """


class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
    """

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None,
                 do_split_on_punc=True):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, self.do_split_on_punc, never_split))
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    @staticmethod
    def _is_punctuation(char):
        """Checks whether `chars` is a punctuation character."""
        # The character is not a punctuation character to split
        # if it is ".", "'", or "_"
        if char in ['.', '\'', '_']:
            return False

        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class, but we treat them as punctuation anyway, for
        # consistency.
        if ((33 <= cp <= 47) or (58 <= cp <= 64) or
                (91 <= cp <= 96) or (123 <= cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def _run_split_on_punc(self, text, do_split_on_punc=True, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            is_punctuation = _is_punctuation(char) if do_split_on_punc else self._is_punctuation(char)
            if is_punctuation:
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
                (cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)  #
                or (cp >= 0x20000 and cp <= 0x2A6DF)  #
                or (cp >= 0x2A700 and cp <= 0x2B73F)  #
                or (cp >= 0x2B740 and cp <= 0x2B81F)  #
                or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
