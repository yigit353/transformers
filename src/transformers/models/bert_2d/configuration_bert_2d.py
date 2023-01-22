# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" BERT model configuration"""
from collections import OrderedDict
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

logger = logging.get_logger(__name__)

BERT_2D_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bert2d-base-uncased": "https://huggingface.co/bert2d-base-uncased/resolve/main/config.json",
    # See all BERT models at https://huggingface.co/models?filter=bert2d
}


class Bert2dConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Bert2dModel`] or a [`TFBert2dModel`]. It is used to
    instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    [bert-base-uncased](https://huggingface.co/bert2d-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Bert2dModel`] or [`TFBert2dModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_word_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        subword_intermediate_position_embeddings (`int`, *optional*, defaults to 1):
            The number of maximum subword intermediate position embeddings
        intermediate_subword_distribution_strategy (`str`, *optional*, defaults to "uniform"):
            The strategy to distribute the subword intermediate position embeddings
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`Bert2dModel`] or [`TFBert2dModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        subword_embedding_order (`str`, *optional*, defaults to `"ending_first"`):
            The order of subword embedding. If `"ending_first"`, if a word has at least one subword
            the subword embedding is caclulated as Root Subword Embedding-Ending Subword Embedding-
            [Intermediate Subword Embedding(s)] depending on the `subword_intermediate_position_embeddings` value.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from transformers import Bert2dConfig, Bert2dModel

    >>> # Initializing a BERT bert-base-uncased style configuration
    >>> configuration = Bert2dConfig()

    >>> # Initializing a model (with random weights) from the bert-base-uncased style configuration
    >>> model = Bert2dModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bert"

    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_word_position_embeddings=512,
            subword_intermediate_position_embeddings=1,
            intermediate_subword_distribution_strategy="uniform",
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            subword_embedding_order="ending_first",
            pad_token_id=0,
            use_cache=True,
            classifier_dropout=None,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_word_position_embeddings = max_word_position_embeddings
        self.subword_intermediate_position_embeddings = subword_intermediate_position_embeddings
        self.intermediate_subword_distribution_strategy = intermediate_subword_distribution_strategy
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.subword_embedding_order = subword_embedding_order
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


class Bert2dOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
                ("word_ids", dynamic_axis),
                ("subword_ids", dynamic_axis)
            ]
        )
