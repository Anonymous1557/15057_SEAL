# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

__version__ = "0.11.2.dev0"

from .auto import (
    AutoPeftModel,
    AutoPeftModelForCausalLM,
    AutoPeftModelForSequenceClassification,
    AutoPeftModelForSeq2SeqLM,
    AutoPeftModelForTokenClassification,
    AutoPeftModelForQuestionAnswering,
    AutoPeftModelForFeatureExtraction,
)
from .mapping import (
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
    PEFT_TYPE_TO_CONFIG_MAPPING,
    get_peft_config,
    get_peft_model,
    inject_adapter_in_model,
)
from .mixed_model import PeftMixedModel
from .peft_model import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
    PeftModelForQuestionAnswering,
    PeftModelForFeatureExtraction,
    get_layer_status,
    get_model_status,
)
from .tuners import (
    AdaptionPromptConfig,
    AdaptionPromptModel,
    LoraConfig,
    LoftQConfig,
    LoraModel,
    LoHaConfig,
    LoHaModel,
    LoKrConfig,
    LoKrModel,
    IA3Config,
    IA3Model,
    AdaLoraConfig,
    AdaLoraModel,
    BOFTConfig,
    BOFTModel,
    PrefixEncoder,
    PrefixTuningConfig,
    PromptEmbedding,
    PromptEncoder,
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
    PromptTuningConfig,
    PromptTuningInit,
    MultitaskPromptTuningConfig,
    MultitaskPromptTuningInit,
    OFTConfig,
    OFTModel,
    PolyConfig,
    PolyModel,
    LNTuningConfig,
    LNTuningModel,
    VeraConfig,
    VeraModel,
    SealConfig,
    KeyConfig,
    SealModel,
)
from .utils import (
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    PeftType,
    TaskType,
    bloom_model_postprocess_past_key_value,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    replace_lora_weights_loftq,
    set_peft_model_state_dict,
    shift_tokens_right,
    load_peft_weights,
    cast_mixed_precision_params,
)
from .config import PeftConfig, PromptLearningConfig
