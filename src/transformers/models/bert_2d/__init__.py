# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_bert_2d import BERT_2D_PRETRAINED_CONFIG_ARCHIVE_MAP, Bert2dConfig, Bert2dOnnxConfig
    from .tokenization_bert_2d import BasicTokenizer, Bert2dTokenizer, WordpieceTokenizer

    from .modeling_bert_2d import (
        BERT_2D_PRETRAINED_MODEL_ARCHIVE_LIST,
        Bert2dForMaskedLM,
        Bert2dForMultipleChoice,
        Bert2dForNextSentencePrediction,
        Bert2dForPreTraining,
        Bert2dForQuestionAnswering,
        Bert2dForSequenceClassification,
        Bert2dForTokenClassification,
        Bert2dLayer,
        Bert2dLMHeadModel,
        Bert2dModel,
        Bert2dPreTrainedModel,
        load_tf_weights_in_bert,
    )
else:
    import sys
    from ...utils import _LazyModule

    _import_structure = {
        "configuration_bert_2d": ["BERT_2D_PRETRAINED_CONFIG_ARCHIVE_MAP", "Bert2dConfig", "Bert2dOnnxConfig"],
        "tokenization_bert_2d": ["BasicTokenizer", "Bert2dTokenizer", "WordpieceTokenizer"],
        "modeling_bert_2d": [
            "BERT_2D_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Bert2dForMaskedLM",
            "Bert2dForMultipleChoice",
            "Bert2dForNextSentencePrediction",
            "Bert2dForPreTraining",
            "Bert2dForQuestionAnswering",
            "Bert2dForSequenceClassification",
            "Bert2dForTokenClassification",
            "Bert2dLayer",
            "Bert2dLMHeadModel",
            "Bert2dModel",
            "Bert2dPreTrainedModel",
            "load_tf_weights_in_bert",
        ]}

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
