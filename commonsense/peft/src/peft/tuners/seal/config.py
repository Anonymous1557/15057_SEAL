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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType

from peft.tuners.lora import LoraConfig
from .key_config import KeyConfig


@dataclass
class SealConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`SealModel`].

    Args:
        r (`int`):
            Lora attention dimension (the "rank").
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        lora_alpha (`int`):
            The alpha parameter for Lora scaling.
        lora_dropout (`float`):
            The dropout probability for Lora layers.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for LoRA. Can be 'none', 'all' or 'lora_only'. If 'all' or 'lora_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        use_rslora (`bool`):
            When set to True, uses <a href='https://doi.org/10.48550/arXiv.2312.03732'>Rank-Stabilized LoRA</a> which
            sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it was proven to work better.
            Otherwise, it will use the original default value of `lora_alpha/r`.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
        init_lora_weights (`bool` | `Literal["gaussian", "pissa", "pissa_niter_[number of iters]", "loftq"]`):
            How to initialize the weights of the adapter layers. Passing True (default) results in the default
            initialization from the reference implementation from Microsoft. Passing 'gaussian' results in Gaussian
            initialization scaled by the LoRA rank for linear and layers. Setting the initialization to False leads to
            completely random initialization and is discouraged. Pass `'loftq'` to use LoftQ initialization. Passing
            'pissa' results in the initialization of PiSSA, which converge more rapidly than LoRA and ultimately
            achieve superior performance. Moreover, PiSSA reduces the quantization error compared to QLoRA, leading to
            further enhancements. Passing 'pissa_niter_[number of iters]' initiates Fast-SVD-based PiSSA
            initialization, where [number of iters] indicates the number of subspace iterations to perform FSVD, and
            must be a nonnegative integer. When the [number of iters] is set to 16, it can complete the initialization
            of a 7b model within seconds, and the training effect is approximately equivalent to using SVD. For more
            information, see <a href='https://arxiv.org/abs/2404.02948'>Principal Singular values and Singular vectors
            Adaptation</a>.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.
        rank_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`.
        alpha_pattern (`dict`):
            The mapping from layer names or regexp expression to alphas which are different from the default alpha
            specified by `lora_alpha`.
        megatron_config (`Optional[dict]`):
            The TransformerConfig arguments for Megatron. It is used to create LoRA's parallel linear layer. You can
            get it like this, `core_transformer_config_from_args(get_args())`, these two functions being from Megatron.
            The arguments will be used to initialize the TransformerConfig of Megatron. You need to specify this
            parameter when you want to apply LoRA to the ColumnParallelLinear and RowParallelLinear layers of megatron.
        megatron_core (`Optional[str]`):
            The core module from Megatron to use, defaults to `"megatron.core"`.
        loftq_config (`Optional[LoftQConfig]`):
            The configuration of LoftQ. If this is not None, then LoftQ will be used to quantize the backbone weights
            and initialize Lora layers. Also pass `init_lora_weights='loftq'`. Note that you should not pass a
            quantized model in this case, as LoftQ will quantize the model itself.
        use_dora (`bool`):
            Enable 'Weight-Decomposed Low-Rank Adaptation' (DoRA). This technique decomposes the updates of the weights
            into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the magnitude is
            handled by a separate learnable parameter. This can improve the performance of LoRA especially at low
            ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a bigger overhead than pure
            LoRA, so it is recommended to merge weights for inference. For more information, see
            https://arxiv.org/abs/2402.09353.
        layer_replication (`List[Tuple[int, int]]`):
            Build a new stack of layers by stacking the original model layers according to the ranges specified. This
            allows expanding (or shrinking) the model without duplicating the base model weights. The new layers will
            all have separate LoRA adapters attached to them.
        key_path ('str'):
            key path which contains the key to entangle LoRA model.
    """
    key_config: Union[KeyConfig, dict] = field( # abstraction.
        default=None,
        metadata={
            "help": "The KeyConfig which contains key information"
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.SEAL
        if isinstance(self.key_config, dict):
            pass
    
    def to_dict(self):
        orig_dict = {
            k: getattr(self, k) for k in 
                [
                "auto_mapping", "base_model_name_or_path", 
                "peft_type", "r", "target_modules", "lora_alpha", "lora_dropout", "fan_in_fan_out", 
                 "bias", "modules_to_save", "init_lora_weights", "layers_to_transform", 
                 "layers_pattern", "inference_mode", "task_type", "revision",
                 "rank_pattern", "alpha_pattern", "megatron_config", "megatron_core", "loftq_config",
                 "use_dora"
                ]
        }
        # orig_dict["peft_type"] = "LORA" # hard-coded
        orig_dict["key_config"] = self.key_config.to_dict()
        return orig_dict