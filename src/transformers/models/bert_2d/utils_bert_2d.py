import math
from typing import List


def is_subword(token: str) -> bool:
    """Returns if a token is a subword"""
    return token.startswith("##")


def create_word_ids(tokens: List[str], restart_new_sentence=False, seperator_token="[SEP]",
                    subword_prefix="##") -> List[int]:
    """Creates word ids for given tokens"""
    word_ids: List[int] = []
    current_word_id: int = -1
    sentence_restart = False
    if tokens.count(seperator_token) < 2:
        restart_new_sentence = False
    for token in tokens:
        # If new sentence requires new starting, do not restart at second
        if restart_new_sentence and not sentence_restart and token == seperator_token:
            current_word_id = 0
            sentence_restart = True
        elif not token.startswith(subword_prefix):
            current_word_id += 1
        # If tokens start with a subword
        elif current_word_id == -1:
            current_word_id = 0
        word_ids.append(current_word_id)

    return word_ids


def _col_round(x: float) -> int:
    """Colloquial rounding where 0.5 rounds to 1"""
    frac = x - math.floor(x)
    if frac < 0.5:
        return math.floor(x)
    return math.ceil(x)


def _get_uniform_id(si: int, max_intermediate_subwords: int, num_intermediate_subwords: int) -> int:
    """Calculates uniform id for the given subword index, si, and max and number of intermediate subwords"""
    return _col_round(si * (max_intermediate_subwords - 1) / num_intermediate_subwords)


def _get_ids_from_subwords(num_subwords: int, max_intermediate_subword_positions_per_word: int,
                           subword_embedding_order: str, intermediate_subword_distribution_strategy: str,
                           starts_with_subword=False) -> List[int]:
    """Calculate subword ids for given subwords list of a single word"""

    if starts_with_subword:
        if num_subwords == 1:
            return [1]
    elif num_subwords <= 2:
        return list(range(num_subwords))

    if subword_embedding_order == "ending_first":
        # Return with simple cases
        if starts_with_subword and num_subwords == 2:
            return [2, 1]

        subword_ids: List[int] = [0] if not starts_with_subword else [1]
        # 2 for root and last token
        num_intermediate_subwords: int = num_subwords - 2 if not starts_with_subword else num_subwords - 1

        # R - L - I1 - I2 - ...
        if num_intermediate_subwords <= max_intermediate_subword_positions_per_word:
            for si in range(num_intermediate_subwords):
                subword_ids.append(2 + si)
        # if there are more intermediate subwords than allowed
        else:
            if intermediate_subword_distribution_strategy == "uniform":
                # Distribute all indices uniformly between allowed indices
                for si in range(num_intermediate_subwords):
                    subword_ids.append(
                        2 + _get_uniform_id(si, max_intermediate_subword_positions_per_word, num_intermediate_subwords))
            elif intermediate_subword_distribution_strategy == "leftover_as_last":
                # Append subword indices that are allowed
                for si in range(max_intermediate_subword_positions_per_word):
                    subword_ids.append(2 + si)
                # Append rest as last
                for si in range(num_intermediate_subwords - max_intermediate_subword_positions_per_word):
                    subword_ids.append(1)
            else:
                raise ValueError("Unsupported intermediate subword distribution strategy")
        subword_ids.append(1)
        return subword_ids
    raise ValueError("Unsupported subword embedding order")


def _extend_subword_ids_for_word(subword_ids: List[int], intermediate_subword_distribution_strategy: str,
                                 max_intermediate_subword_positions_per_word: int, num_subwords: int,
                                 start_subword_processed: bool,
                                 starts_with_subword: bool, subword_embedding_order: str) -> None:
    """Extends subword ids for each word"""
    subword_ids.extend(
        _get_ids_from_subwords(num_subwords, max_intermediate_subword_positions_per_word, subword_embedding_order,
                               intermediate_subword_distribution_strategy,
                               starts_with_subword and not start_subword_processed)
    )


def create_subword_ids(tokens: List[str],
                       max_intermediate_subword_positions_per_word: int,
                       subword_embedding_order: str,
                       intermediate_subword_distribution_strategy: str):
    """Creates subword ids for the given tokens and parameters"""

    # If tokens are empty return empty subword id list
    if len(tokens) == 0:
        return []

    subword_ids: List[int] = []
    num_subwords: int = 0
    starts_with_subword = is_subword(tokens[0])
    start_subword_processed = False
    for token in tokens:
        if not is_subword(token):
            if num_subwords > 0:
                _extend_subword_ids_for_word(subword_ids, intermediate_subword_distribution_strategy,
                                             max_intermediate_subword_positions_per_word, num_subwords,
                                             start_subword_processed,
                                             starts_with_subword, subword_embedding_order)
                start_subword_processed = True
                num_subwords = 0
        num_subwords += 1
    if num_subwords > 0:
        _extend_subword_ids_for_word(subword_ids, intermediate_subword_distribution_strategy,
                                     max_intermediate_subword_positions_per_word,
                                     num_subwords, start_subword_processed, starts_with_subword,
                                     subword_embedding_order)
    return subword_ids
