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

# rely on isort to merge the imports
from ...utils import  _LazyModule, OptionalDependencyNotAvailable, is_tokenizers_available
from ...utils import is_tf_available



from ...utils import is_torch_available




_import_structure = {
    "configuration_bert_2d": ["BERT_2D_PRETRAINED_CONFIG_ARCHIVE_MAP", "Bert2dConfig"],
    "tokenization_bert_2d": ["Bert2dTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_bert_2d_fast"] = ["Bert2dTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_bert_2d"] = [
        "BERT_2D_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Bert2dForMaskedLM",
        "Bert2dForCausalLM",
        "Bert2dForMultipleChoice",
        "Bert2dForQuestionAnswering",
        "Bert2dForSequenceClassification",
        "Bert2dForTokenClassification",
        "Bert2dLayer",
        "Bert2dModel",
        "Bert2dPreTrainedModel",
        "load_tf_weights_in_bert_2d",
    ]



try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_bert_2d"] = [
        "TF_BERT_2D_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFBert2dForMaskedLM",
        "TFBert2dForCausalLM",
        "TFBert2dForMultipleChoice",
        "TFBert2dForQuestionAnswering",
        "TFBert2dForSequenceClassification",
        "TFBert2dForTokenClassification",
        "TFBert2dLayer",
        "TFBert2dModel",
        "TFBert2dPreTrainedModel",
    ]




if TYPE_CHECKING:
    from .configuration_bert_2d import BERT_2D_PRETRAINED_CONFIG_ARCHIVE_MAP, Bert2dConfig
    from .tokenization_bert_2d import Bert2dTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_bert_2d_fast import Bert2dTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_bert_2d import (
            BERT_2D_PRETRAINED_MODEL_ARCHIVE_LIST,
            Bert2dForMaskedLM,
            Bert2dForCausalLM,
            Bert2dForMultipleChoice,
            Bert2dForQuestionAnswering,
            Bert2dForSequenceClassification,
            Bert2dForTokenClassification,
            Bert2dLayer,
            Bert2dModel,
            Bert2dPreTrainedModel,
            load_tf_weights_in_bert_2d,
        )



    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_bert_2d import (
            TF_BERT_2D_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFBert2dForMaskedLM,
            TFBert2dForCausalLM,
            TFBert2dForMultipleChoice,
            TFBert2dForQuestionAnswering,
            TFBert2dForSequenceClassification,
            TFBert2dForTokenClassification,
            TFBert2dLayer,
            TFBert2dModel,
            TFBert2dPreTrainedModel,
        )



else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
