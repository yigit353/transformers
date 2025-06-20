# coding=utf-8
# Copyright 2025 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT2D model."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, auto_docstring, get_torch_version, logging
from .configuration_bert2d import Bert2DConfig


logger = logging.get_logger(__name__)


def load_tf_weights_in_bert2d(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "bert":
                scope_names[0] = "bert2d"
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


class Bert2DEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config  # Store config for access in methods
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # Additional embeddings for Bert2D
        self.whole_word_embeddings = nn.Embedding(config.max_word_position_embeddings, config.hidden_size)
        self.subword_embeddings = nn.Embedding(
            config.max_intermediate_subword_position_embeddings + 2, config.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Buffers for default IDs, similar to vanilla BERT's position_ids and token_type_ids
        # These are used if word_ids or subword_ids are not provided.
        # Max length for these defaults is config.max_position_embeddings for consistency.
        # Actual length used will be sliced to seq_length at runtime.
        self.register_buffer(
            "default_word_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # Default subword_ids to all zeros.
        self.register_buffer(
            "default_subword_ids", torch.zeros(1, config.max_position_embeddings, dtype=torch.long), persistent=False
        )
        # Buffer for token_type_ids (all zeros)
        self.register_buffer(
            "token_type_ids_buffer", torch.zeros(1, config.max_position_embeddings, dtype=torch.long), persistent=False
        )

    def _detect_left_padding_and_create_ids(
        self, input_ids_tensor: Optional[torch.LongTensor], input_shape, seq_length, device
    ):
        # Fallback to simple defaults if input_ids_tensor is None or pad_token_id is not configured
        if input_ids_tensor is None or self.config.pad_token_id is None:
            default_word_ids_slice = self.default_word_ids[:, :seq_length]
            default_subword_ids_slice = self.default_subword_ids[:, :seq_length]
            if len(input_shape) == 3:  # (batch, num_choices, seq_len)
                generated_word_ids = default_word_ids_slice.expand(input_shape[0], input_shape[1], seq_length)
                generated_subword_ids = default_subword_ids_slice.expand(input_shape[0], input_shape[1], seq_length)
            else:  # (batch, seq_len)
                generated_word_ids = default_word_ids_slice.expand(input_shape[0], seq_length)
                generated_subword_ids = default_subword_ids_slice.expand(input_shape[0], seq_length)
            return generated_word_ids, generated_subword_ids

        pad_token_id = self.config.pad_token_id
        max_allowed_word_id = self.config.max_word_position_embeddings - 1

        # Reshape for easier iteration if 3D
        is_3d = input_ids_tensor.ndim == 3
        if is_3d:
            batch_size_orig, num_choices_orig, _ = input_shape
            reshaped_input_ids = input_ids_tensor.contiguous().view(-1, seq_length)
        else:
            batch_size_orig, _ = input_shape  # Will be used if reshaped_input_ids is empty
            num_choices_orig = 1  # Placeholder
            reshaped_input_ids = input_ids_tensor

        if reshaped_input_ids.shape[0] == 0:  # Empty batch
            if is_3d:
                empty_word_ids = torch.empty((batch_size_orig, num_choices_orig, 0), dtype=torch.long, device=device)
                empty_subword_ids = torch.empty(
                    (batch_size_orig, num_choices_orig, 0), dtype=torch.long, device=device
                )
            else:
                empty_word_ids = torch.empty((batch_size_orig, 0), dtype=torch.long, device=device)
                empty_subword_ids = torch.empty((batch_size_orig, 0), dtype=torch.long, device=device)
            return empty_word_ids, empty_subword_ids

        batch_generated_word_ids = []
        batch_generated_subword_ids = []

        for i in range(reshaped_input_ids.shape[0]):  # Iterate over effective batch
            current_sequence_input_ids = reshaped_input_ids[i]

            num_left_pads = 0
            for token_id_val in current_sequence_input_ids:
                if token_id_val.item() == pad_token_id:
                    num_left_pads += 1
                else:
                    break  # Stop counting at the first non-pad token

            current_word_ids = torch.zeros(seq_length, dtype=torch.long, device=device)
            content_word_id_counter = 0
            for j in range(num_left_pads, seq_length):
                clamped_word_id = min(content_word_id_counter, max_allowed_word_id)
                current_word_ids[j] = clamped_word_id
                content_word_id_counter += 1
            batch_generated_word_ids.append(current_word_ids)

            current_subword_ids = torch.zeros(seq_length, dtype=torch.long, device=device)
            batch_generated_subword_ids.append(current_subword_ids)

        final_word_ids = torch.stack(batch_generated_word_ids)
        final_subword_ids = torch.stack(batch_generated_subword_ids)

        if is_3d:
            final_word_ids = final_word_ids.view(batch_size_orig, num_choices_orig, seq_length)
            final_subword_ids = final_subword_ids.view(batch_size_orig, num_choices_orig, seq_length)

        return final_word_ids, final_subword_ids

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        word_ids: Optional[torch.LongTensor] = None,
        subword_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,  # This argument is not directly used in Bert2D's 2D embedding logic
    ) -> torch.Tensor:
        r"""
        word_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Word IDs for each token in the input sequence. These IDs represent the absolute position of the word to
            which each token belongs. All tokens (subwords) constituting the same word share the same `word_id`.
            For example, in the sentence "Tokenization is useful", if "Tokenization" is split into "Token" and "##ization",
            both "Token" and "##ization" will have the same `word_id` (e.g., 0), "is" will have `word_id` 1, and "useful"
            will have `word_id` 2. These are used to compute word-level absolute position embeddings.
        subword_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Subword IDs for each token in the input sequence. These IDs represent the relative position of a subword
            within its parent word. The specific assignment scheme depends on the tokenizer's configuration
            (e.g., `subword_embedding_order`, `max_intermediate_subword_positions_per_word`). For example, with
            `subword_embedding_order="ending_first"`, the first token of a word typically gets `0`, the last token of the
            same word gets `1`, and intermediate tokens get other IDs (e.g., `2`, `3`, ...). If a word is composed
            of a single token, its `subword_id` might be `0` (or `1` if it's a subword itself that starts the sequence of
            subwords for that "word"). These are used to compute subword-level relative position embeddings.
            Together, `word_ids` and `subword_ids` create a 2D positional ID system.
        """
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length = input_shape[-1]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            buffered_token_type_ids = self.token_type_ids_buffer[:, :seq_length]
            if len(input_shape) == 3:
                token_type_ids = buffered_token_type_ids.expand(input_shape[0], input_shape[1], seq_length)
            else:
                token_type_ids = buffered_token_type_ids.expand(input_shape[0], seq_length)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if word_ids is None:
            # input_ids (tensor) is passed as input_ids_tensor to the helper
            generated_word_ids, _ = self._detect_left_padding_and_create_ids(
                input_ids, input_shape, seq_length, device
            )
            word_ids = generated_word_ids
            if input_ids is not None and self.config.pad_token_id is not None:
                warnings.warn(
                    "`word_ids` was not provided and has been defaulted based on padding (if detected) or simple sequential IDs. "
                    "This behavior is usually not desired for Bert2D models and may lead to unexpected results if the model "
                    "is not explicitly expecting this behavior.",
                    UserWarning,
                )
            else:  # input_ids is None or pad_token_id is None, so simple defaults were used
                warnings.warn(
                    "`word_ids` was not provided. Since `input_ids` or `config.pad_token_id` is unavailable for "
                    "padding detection, `word_ids` defaulted to simple sequential IDs. This behavior is usually "
                    "not desired for Bert2D models.",
                    UserWarning,
                )

        if subword_ids is None:
            _, generated_subword_ids = self._detect_left_padding_and_create_ids(
                input_ids, input_shape, seq_length, device
            )
            subword_ids = generated_subword_ids
            if input_ids is not None and self.config.pad_token_id is not None:
                warnings.warn(
                    "`subword_ids` was not provided and has been defaulted to all zeros (for padding and content). "
                    "This behavior is usually not desired for Bert2D models and may lead to unexpected results if the model "
                    "is not explicitly expecting this behavior.",
                    UserWarning,
                )
            else:  # input_ids is None or pad_token_id is None, so simple defaults were used (all zeros)
                warnings.warn(
                    "`subword_ids` was not provided. Since `input_ids` or `config.pad_token_id` is unavailable for "
                    "padding detection, `subword_ids` defaulted to all zeros. This behavior is usually "
                    "not desired for Bert2D models.",
                    UserWarning,
                )

        whole_word_embeddings = self.whole_word_embeddings(word_ids)
        subword_embeddings = self.subword_embeddings(subword_ids)

        embeddings = inputs_embeds + token_type_embeddings + whole_word_embeddings + subword_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Bert2DSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class Bert2DSdpaSelfAttention(Bert2DSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.dropout_prob = config.attention_probs_dropout_prob
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
            logger.warning_once(
                "Bert2DSdpaSelfAttention is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "non-absolute `position_embedding_type` or `output_attentions=True` or `head_mask`. Falling back to "
                "the manual attention implementation, aif_model_name_to_use_tag=True, but specifying the manual implementation will be required from "
                "Transformers version v5.0.0 onwards. This warning can be removed using the argument "
                '`attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

        bsz, tgt_len, _ = hidden_states.size()
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        is_cross_attention = encoder_hidden_states is not None
        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        attention_mask_to_pass = encoder_attention_mask if is_cross_attention else attention_mask

        if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
            key_layer, value_layer = past_key_value
        else:
            key_layer = self.transpose_for_scores(self.key(current_states))
            value_layer = self.transpose_for_scores(self.value(current_states))
            if past_key_value is not None and not is_cross_attention:
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        if self.require_contiguous_qkv and query_layer.device.type == "cuda" and attention_mask_to_pass is not None:
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        is_causal = (
            True
            if self.is_decoder and not is_cross_attention and attention_mask_to_pass is None and tgt_len > 1
            else False
        )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask_to_pass,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        outputs = (attn_output,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class Bert2DSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


BERT2D_SELF_ATTENTION_CLASSES = {
    "eager": Bert2DSelfAttention,
    "sdpa": Bert2DSdpaSelfAttention,
}


class Bert2DAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BERT2D_SELF_ATTENTION_CLASSES[config._attn_implementation](
            config, position_embedding_type=position_embedding_type
        )
        self.output = Bert2DSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class Bert2DIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Bert2DOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Bert2DLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Bert2DAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = Bert2DAttention(config, position_embedding_type="absolute")
        self.intermediate = Bert2DIntermediate(config)
        self.output = Bert2DOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Bert2DEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Bert2DLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class Bert2DPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Bert2DPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class Bert2DLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = Bert2DPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class Bert2DOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = Bert2DLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class Bert2DOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class Bert2DPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = Bert2DLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


@auto_docstring
class Bert2DPreTrainedModel(PreTrainedModel):
    config_class = Bert2DConfig
    load_tf_weights = load_tf_weights_in_bert2d
    base_model_prefix = "bert2d"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Bert2DLMPredictionHead):  # Ensure bias is initialized for this specific head
            module.bias.data.zero_()


@dataclass
class Bert2DForPreTrainingOutput(ModelOutput):
    """
    Output type of [`Bert2DForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: Optional[torch.FloatTensor] = None
    seq_relationship_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@auto_docstring(
    custom_intro="""
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """
)
class Bert2DModel(Bert2DPreTrainedModel):
    _no_split_modules = ["Bert2DEmbeddings", "Bert2DLayer"]

    def __init__(self, config, add_pooling_layer=True):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.config = config

        self.embeddings = Bert2DEmbeddings(config)
        self.encoder = Bert2DEncoder(config)
        self.pooler = Bert2DPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
        subword_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        word_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)` or `(batch_size, num_choices, sequence_length)` for multiple choice, *optional*):
            Word IDs for each token in the input sequence. These IDs represent the absolute position of the word to
            which each token belongs. All tokens (subwords) constituting the same word share the same `word_id`.
        subword_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)` or `(batch_size, num_choices, sequence_length)` for multiple choice, *optional*):
            Subword IDs for each token in the input sequence. These IDs represent the relative position of a subword
            within its parent word. Together with `word_ids`, they create a 2D positional ID system.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # FIX STARTS HERE: Add a batch dimension to 1D inputs
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                if attention_mask is not None and attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                if token_type_ids is not None and token_type_ids.dim() == 1:
                    token_type_ids = token_type_ids.unsqueeze(0)
                if word_ids is not None and word_ids.dim() == 1:
                    word_ids = word_ids.unsqueeze(0)
                if subword_ids is not None and subword_ids.dim() == 1:
                    subword_ids = subword_ids.unsqueeze(0)
            # FIX ENDS HERE
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape[:2]

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # Default token_type_ids if not provided (similar to vanilla BERT)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids_buffer"):  # Check for our buffer
                buffered_token_type_ids = self.embeddings.token_type_ids_buffer[:, : input_shape[-1]]
                if len(input_shape) == 3:  # (batch, num_choices, seq_len)
                    token_type_ids = buffered_token_type_ids.expand(input_shape[0], input_shape[1], input_shape[-1])
                else:  # (batch, seq_len)
                    token_type_ids = buffered_token_type_ids.expand(input_shape[0], input_shape[-1])
            else:  # Fallback if buffer isn't there (shouldn't happen with new __init__)
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # word_ids and subword_ids will be defaulted inside Bert2DEmbeddings.forward if None

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            word_ids=word_ids,  # Can be None, will be handled by Bert2DEmbeddings
            subword_ids=subword_ids,  # Can be None, will be handled by Bert2DEmbeddings
            token_type_ids=token_type_ids,  # Now guaranteed to be non-None
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@auto_docstring(
    custom_intro="""
    Bert2D Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """
)
class Bert2DForPreTraining(Bert2DPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.bert2d = Bert2DModel(config)
        self.cls = Bert2DPreTrainingHeads(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
        subword_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Bert2DForPreTrainingOutput]:
        r"""
        word_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Word IDs for each token in the input sequence. These IDs represent the absolute position of the word to
            which each token belongs. All tokens (subwords) constituting the same word share the same `word_id`.
        subword_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Subword IDs for each token in the input sequence. These IDs represent the relative position of a subword
            within its parent word. Together with `word_ids`, they create a 2D positional ID system.
        next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see `input_ids` docstring)
            Indices should be in `[0, 1]`:
            - 0 indicates sequence B is not the continuation of sequence A,
            - 1 indicates sequence B is the continuation of sequence A.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert2d(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            word_ids=word_ids,
            subword_ids=subword_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        return Bert2DForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Bert2D Model with a `language modeling` head on top for CLM fine-tuning.
    """
)
class Bert2DLMHeadModel(Bert2DPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        if not config.is_decoder:
            logger.warning("If you want to use `Bert2DLMHeadModel` as a standalone, add `is_decoder=True.`")
        self.bert2d = Bert2DModel(config, add_pooling_layer=False)
        self.cls = Bert2DOnlyMLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
        subword_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        word_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Word IDs for each token in the input sequence. These IDs represent the absolute position of the word to
            which each token belongs. All tokens (subwords) constituting the same word share the same `word_id`.
        subword_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Subword IDs for each token in the input sequence. These IDs represent the relative position of a subword
            within its parent word. Together with `word_ids`, they create a 2D positional ID system.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
        outputs = self.bert2d(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            word_ids=word_ids,
            subword_ids=subword_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        lm_loss = None
        if labels is not None:
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output
        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # `input_ids` is the full current sequence.
        # `model_kwargs` contains various things, including potentially 'word_ids', 'subword_ids' which are the
        # full accumulated versions from the previous step's _update_model_kwargs_for_generation call.

        # Call super(). This will:
        # - If past_key_values exist, slice `input_ids` to just the new token(s).
        # - Potentially slice `attention_mask` and `token_type_ids` if they are present in model_kwargs.
        # - Populate `model_inputs` with these (potentially sliced) inputs.
        # - Carry over other kwargs like 'word_ids', 'subword_ids' from `model_kwargs` into `model_inputs`.
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, **model_kwargs
        )
        # Now, model_inputs["input_ids"] is the current token(s) to be processed (e.g., shape [B, 1] or [B, S_new])
        # model_inputs["word_ids"] (if present) is still the *full* sequence from model_kwargs.
        # model_inputs["subword_ids"] (if present) is still the *full* sequence from model_kwargs.
        # model_inputs["token_type_ids"] (if present) is the *sliced* sequence if it was in model_kwargs,
        # because "token_type_ids" is a standard name that super().prepare_inputs_for_generation knows how to slice.

        if past_key_values is not None:
            # This is an auto-regressive step. We need to slice our custom IDs.
            num_new_tokens = model_inputs["input_ids"].shape[-1]  # Length of the currently processed tokens

            if "word_ids" in model_inputs and model_inputs["word_ids"] is not None:
                # model_inputs["word_ids"] is currently the full sequence from model_kwargs. Slice its end.
                model_inputs["word_ids"] = model_inputs["word_ids"][:, -num_new_tokens:]

            if "subword_ids" in model_inputs and model_inputs["subword_ids"] is not None:
                model_inputs["subword_ids"] = model_inputs["subword_ids"][:, -num_new_tokens:]
        # If first step (past_key_values is None), model_inputs["input_ids"] is the full prompt.
        # Custom IDs in model_inputs (word_ids, subword_ids), if present, are already the full original versions from model_kwargs,
        # so no change needed for them here. Standard IDs like token_type_ids are handled by super().

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # Call parent method first to update standard kwargs like attention_mask, past_key_values, cache_position
        # and token_type_ids.
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

        # Determine device and batch_size from a reliable tensor in model_kwargs or outputs
        # (input_ids in model_kwargs is not yet updated at this stage of the generation loop)
        # The attention_mask is updated by super(), so it's a good candidate.
        # Fallback to outputs.logits if attention_mask is not available.
        reference_tensor = model_kwargs.get("attention_mask", outputs.logits)
        device = reference_tensor.device
        batch_size = reference_tensor.shape[0]

        if "word_ids" in model_kwargs and model_kwargs["word_ids"] is not None:
            current_word_ids = model_kwargs["word_ids"]  # Shape (batch_size, L_old)
            max_allowed_word_id = self.config.max_word_position_embeddings - 1

            if current_word_ids.shape[-1] > 0:
                last_word_id_val = current_word_ids[:, -1:]  # Shape (batch_size, 1)
                # Create new word_ids by incrementing the last one
                new_word_ids_increment = torch.arange(
                    1, num_new_tokens + 1, device=device, dtype=torch.long
                ).unsqueeze(0)  # Shape (1, num_new_tokens)
                new_word_ids_segment_unclamped = last_word_id_val + new_word_ids_increment  # Broadcasting
            else:  # word_ids was empty (e.g. generating from empty prompt and no initial word_ids)
                new_word_ids_segment_unclamped = (
                    torch.arange(0, num_new_tokens, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )  # Starts from 0

            # Clamp the new segment
            new_word_ids_segment = torch.clamp(new_word_ids_segment_unclamped, max=max_allowed_word_id)
            model_kwargs["word_ids"] = torch.cat([current_word_ids, new_word_ids_segment], dim=-1)

        if "subword_ids" in model_kwargs and model_kwargs["subword_ids"] is not None:
            current_subword_ids = model_kwargs["subword_ids"]  # Shape (batch_size, L_old)
            # New tokens are considered root words, so their subword_id is 0
            new_subword_ids_segment = torch.zeros((batch_size, num_new_tokens), dtype=torch.long, device=device)
            model_kwargs["subword_ids"] = torch.cat([current_subword_ids, new_subword_ids_segment], dim=-1)

        return model_kwargs

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@auto_docstring(
    custom_intro="""
    Bert2D Model with a `language modeling` head on top for CLM fine-tuning.
    """
)
class Bert2DForMaskedLM(Bert2DPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        if config.is_decoder:
            logger.warning(
                "If you want to use `Bert2DForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.bert2d = Bert2DModel(config, add_pooling_layer=False)
        self.cls = Bert2DOnlyMLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
        subword_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        word_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Word IDs for each token in the input sequence. These IDs represent the absolute position of the word to
            which each token belongs. All tokens (subwords) constituting the same word share the same `word_id`.
        subword_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Subword IDs for each token in the input sequence. These IDs represent the relative position of a subword
            within its parent word. Together with `word_ids`, they create a 2D positional ID system.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert2d(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            word_ids=word_ids,
            subword_ids=subword_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @classmethod
    def can_generate(cls) -> bool:
        return False


@auto_docstring(
    custom_intro="""
    Bert2D Model with a `next sentence prediction (classification)` head on top.
    """
)
class Bert2DForNextSentencePrediction(Bert2DPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert2d = Bert2DModel(config)
        self.cls = Bert2DOnlyNSPHead(config)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
        subword_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
        r"""
        word_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Word IDs for each token in the input sequence. These IDs represent the absolute position of the word to
            which each token belongs. All tokens (subwords) constituting the same word share the same `word_id`.
        subword_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Subword IDs for each token in the input sequence. These IDs represent the relative position of a subword
            within its parent word. Together with `word_ids`, they create a 2D positional ID system.
        """
        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert2d(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            word_ids=word_ids,
            subword_ids=subword_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        seq_relationship_scores = self.cls(pooled_output)
        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))
        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output
        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Bert2D Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """
)
class Bert2DForSequenceClassification(Bert2DPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert2d = Bert2DModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
        subword_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        word_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Word IDs for each token in the input sequence. These IDs represent the absolute position of the word to
            which each token belongs. All tokens (subwords) constituting the same word share the same `word_id`.
        subword_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Subword IDs for each token in the input sequence. These IDs represent the relative position of a subword
            within its parent word. Together with `word_ids`, they create a 2D positional ID system.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert2d(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            word_ids=word_ids,
            subword_ids=subword_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Bert2D Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """
)
class Bert2DForMultipleChoice(Bert2DPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert2d = Bert2DModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
        subword_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        word_ids (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*):
            Word IDs for each token in the input sequence. These IDs represent the absolute position of the word to
            which each token belongs. All tokens (subwords) constituting the same word share the same `word_id`.
        subword_ids (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*):
            Subword IDs for each token in the input sequence. These IDs represent the relative position of a subword
            within its parent word. Together with `word_ids`, they create a 2D positional ID system.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        word_ids = word_ids.view(-1, word_ids.size(-1)) if word_ids is not None else None
        subword_ids = subword_ids.view(-1, subword_ids.size(-1)) if subword_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        outputs = self.bert2d(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            word_ids=word_ids,
            subword_ids=subword_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Bert2D Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """
)
class Bert2DForTokenClassification(Bert2DPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert2d = Bert2DModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
        subword_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        word_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Word IDs for each token in the input sequence. These IDs represent the absolute position of the word to
            which each token belongs. All tokens (subwords) constituting the same word share the same `word_id`.
        subword_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Subword IDs for each token in the input sequence. These IDs represent the relative position of a subword
            within its parent word. Together with `word_ids`, they create a 2D positional ID system.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert2d(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            word_ids=word_ids,
            subword_ids=subword_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Bert2D Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """
)
class Bert2DForQuestionAnswering(Bert2DPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert2d = Bert2DModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
        subword_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        word_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Word IDs for each token in the input sequence. These IDs represent the absolute position of the word to
            which each token belongs. All tokens (subwords) constituting the same word share the same `word_id`.
        subword_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Subword IDs for each token in the input sequence. These IDs represent the relative position of a subword
            within its parent word. Together with `word_ids`, they create a 2D positional ID system.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert2d(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            word_ids=word_ids,
            subword_ids=subword_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Bert2DForMaskedLM",
    "Bert2DForMultipleChoice",
    "Bert2DForNextSentencePrediction",
    "Bert2DForPreTraining",
    "Bert2DForQuestionAnswering",
    "Bert2DForSequenceClassification",
    "Bert2DForTokenClassification",
    "Bert2DLayer",
    "Bert2DLMHeadModel",
    "Bert2DModel",
    "Bert2DPreTrainedModel",
    "load_tf_weights_in_bert2d",
]
