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

import math
import warnings
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose

# Base tuner: LoRA
from peft.tuners.lora import LoraLayer, \
Linear as LoraLinear, \
Embedding as LoraEmbedding, \
Conv2d as LoraConv2d 

from .config import SealConfig
from .key_config import KeyConfig


class SEALWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, up, down, const, scale=torch.tensor(1.0)):
        ctx.save_for_backward(up, down, const, scale)
        diff_weight = ((up @ const) @ down) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (up, down, const, scale) = ctx.saved_tensors
        grad_out = grad_out * scale
        grad_up = grad_out @ (const @ down).T
        grad_down = (up @ const).T @ grad_out
        return grad_up, grad_down, None, None


def make_weight(up, down, const, scale) -> torch.Tensor:
    # return SEALWeight.apply(up, down, const, scale)
    # TODO
    #  - if const has dimension of batch, write correct passes
    return (up @ (const * scale)) @ down


# ---------------------------------------------------------------------------------------------------------------------
# Most of codes in below are inspired by https://github.com/kohakublueleaf/lycoris
# ---------------------------------------------------------------------------------------------------------------------

from enum import Enum


class LayerType(Enum):
    EMBEDDING = 0
    LINEAR = 1
    CONV2D = 2


class SealLayer(LoraLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, layer_name: str, key_config: KeyConfig, **kwargs):
        # super().__init__(base_layer, **kwargs)

        self.key_config = key_config
        self.layer_name = layer_name
        self.layer_type: LayerType = None
        self.key_config.assign_adapter(self.layer_name)  # TODO: fix for load peft

        self.lora_shape = dict()
        self.orig_shape = dict()
        self.kernel_size = kwargs.get("kernel_size", None)
        self.stride = kwargs.get("stride", None)
        self.padding = kwargs.get("padding", None)

    def _op(self, *args, **kwargs):
        raise NotImplementedError()

    def op(self, x: torch.Tensor, weight: torch.Tensor, orig_shape: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if orig_shape is not None:
            return self._op(x, weight.view(orig_shape).to(x.dtype), **kwargs)
        else:
            return self._op(x, weight.to(x.dtype), **kwargs)


    def make_weight(self, adapter: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.layer_type == LayerType.EMBEDDING:
            lora_A = self.lora_embedding_A[adapter].T  # Down block.T -> Down block
            lora_B = self.lora_embedding_B[adapter].T  # Up block.T -> Up block
        else:
            lora_A = self.lora_A[adapter].weight  # Down block
            lora_B = self.lora_B[adapter].weight  # Up block

        dtype = lora_A.dtype
        device = lora_A.device

        key = self.key_config.get_now_key_value(self.layer_name).to(device=device, dtype=dtype)
        key_scaling = self.key_config.get_now_key_scale().to(device=device, dtype=dtype)

        if self.layer_type == LayerType.CONV2D:  # conv2d
            # https://github.com/KohakuBlueLeaf/lycoris lycoris:modules:locon.py
            return (
                lora_B.view(lora_B.size(0), -1),  # shape = [out_features, rank]
                lora_A.view(lora_A.size(0), -1),  # shape = [rank, in_features * product(kernel_size)]
                key * key_scaling
            )
        else:
            return (lora_B, lora_A, key * key_scaling)


    def _make_weight(self, adapter: str) -> torch.Tensor:
        if self.layer_type == LayerType.EMBEDDING:
            lora_A = self.lora_embedding_A[adapter].T  # Down block.T -> Down block
            lora_B = self.lora_embedding_B[adapter].T  # Up block.T -> Up block
        else:
            lora_A = self.lora_A[adapter].weight  # Down block
            lora_B = self.lora_B[adapter].weight  # Up block

        dtype = lora_A.dtype
        device = lora_A.device

        key = self.key_config.get_now_key_value(self.layer_name).to(device=device, dtype=dtype)
        key_scaling = self.key_config.get_now_key_scale().to(device=device, dtype=dtype)

        if self.layer_type == LayerType.CONV2D:  # conv2d
            # https://github.com/KohakuBlueLeaf/lycoris lycoris:modules:locon.py
            return make_weight(
                lora_B.view(lora_B.size(0), -1),  # shape = [out_features, rank]
                lora_A.view(lora_A.size(0), -1),  # shape = [rank, in_features * product(kernel_size)]
                key,
                key_scaling,
            )
        else:
            return make_weight(lora_B, lora_A, key, key_scaling)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        raise NotImplementedError("")
        # return self.make_weight(adapter).reshape(self.orig_shape[adapter])

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        layer_type=LayerType.LINEAR,
    ):
        if not hasattr(self, "lora_shape"):
            self.lora_shape = dict()
        if not hasattr(self, "orig_shape"):
            self.orig_shape = dict()
        if not hasattr(self, "extra_kwargs"):
            self.extra_kwargs = dict()
        self.lora_shape[adapter_name] = (self.out_features, self.in_features)
        self.orig_shape[adapter_name] = (self.out_features, self.in_features)
        self.extra_kwargs[adapter_name] = {}
        self.layer_type = layer_type

        # Proceed update of super class
        # super().update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora)
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # Actual trainable parameters
        # TODO: Implement local updates for each layers
        # self._update_layer(adapter_name, r)
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # Post init
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            self.pissa_init(adapter_name, init_lora_weights)
        elif init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
            # print(adapter_name, "use dora")
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def dora_init(self, adapter_name: str) -> None:
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        # temporarily convert fp16 to fp32, as fp16 can cause trouble on CPU with PyTorch < 2.2
        dtype_is_fp16 = lora_A.dtype == torch.float16
        if dtype_is_fp16:
            lora_A = lora_A.float()
            lora_B = lora_B.float()

        scaling = self.scaling[adapter_name]
        with gather_params_ctx(self.get_base_layer().parameters()):
            base_layer = self.get_base_layer()
            weight = dequantize_module_weight(base_layer)
            if weight.data.ndim == 4:  # For handling LoRAs applied to Conv2Ds.
                lora_weight = torch.mm(lora_B.flatten(start_dim=1), lora_A.flatten(start_dim=1))
                lora_weight = lora_weight.reshape(weight.shape)
            else:
                lora_weight = lora_B @ lora_A

            if dtype_is_fp16:
                lora_weight = lora_weight.half()
            weight_norm = self._get_weight_norm(weight, lora_weight, scaling)

        if self.lora_magnitude_vector is None:
            self.lora_magnitude_vector = nn.ParameterDict()
        # print(self.layer_name, self.lora_magnitude_vector, weight_norm.shape)
        self.lora_magnitude_vector[adapter_name] = nn.Parameter(weight_norm, requires_grad=True)
        # add lora_magnitude_vector to the list of learnable parameters
        self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)
        # print(self.lora_magnitude_vector.keys())
    
    def _update_layer(self, adapter_name, r, *args, **kwargs):
        ''' Local `update_layer` function for each layers

        Args:
            adapter_name
            r

        TODO: Implement for each classes
        '''
        raise NotImplementedError("Layer type for Seal should be provided")

    def _get_sealed_out(self, x, lora_weight: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], orig_shape, active_adapter):
        # dtype, device = x.dtype, x.device
        # lora_weight = self.make_weight(self.active_adapter).to(device=device, dtype=dtype)
        batch_size = x.size()[0]
        x = x.view(batch_size, -1, *x.size()[1:])
        
        lora_B, lora_A, scaled_key = lora_weight # Up, Down, Passport
        passport_size = 1 if len(scaled_key.size()) == 2 else scaled_key.size()[0]
        if passport_size == 1:
            scaled_key = scaled_key.squeeze()
            lora_out = torch.einsum(
                "b...i, op, qi, pq -> b...o", 
                x, lora_B, lora_A, scaled_key
            )
        else:
            group_size = batch_size // passport_size
            assert (batch_size % passport_size) != 0, "passport size must be a dividable number by batch size"
            output_stack = []
            for i in range(group_size):
                group_output = torch.einsum(
                    "g...i, op, qi, gpq -> g...o", 
                    x[i:i+group_size, ...], lora_B, lora_A, scaled_key
                )
                output_stack.append(group_output)
            lora_out = torch.stack(output_stack)
        lora_out = lora_out.squeeze() # if batch == 1, remove this
        lora_out = lora_out.reshape(-1, *orig_shape[1:]) # make shape of lora_output correct

        return lora_out
    
    def _get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = transpose(weight, self.fan_in_fan_out)
        weight = weight + scaling * lora_weight.view(weight.shape)
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm
    
    def _apply_dora(self, x, lora_A, lora_B, scaling, active_adapter):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        # lora_weight = lora_B.weight @ lora_A.weight
        
        magnitude = self.lora_magnitude_vector[active_adapter]
        base_layer = self.get_base_layer()
        weight = dequantize_module_weight(base_layer)
        weight = weight.to(x.dtype)
        lora_weight = self.make_weight(active_adapter).type_as(x)
        weight_norm = self._get_weight_norm(weight, lora_weight, scaling)
        # see section 4.3 of DoRA (https://arxiv.org/abs/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        # print("-"*100)
        # print("x          :", x.shape)
        # print("weight     :", weight.shape)
        # print("magnitude  :", magnitude.shape)
        # print("loraweight :", lora_weight.shape)
        # print("weight_norm:", weight_norm.shape)
        # print("-"*100)
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)

        result_dora = F.linear(x, transpose(weight, self.fan_in_fan_out))
        lora_out = self._get_sealed_out(x, lora_weight, result_dora.shape, active_adapter)
        result_dora = (mag_norm_scale - 1) * result_dora + mag_norm_scale * lora_out * scaling
        # result_dora = (mag_norm_scale - 1) * (
        #     F.linear(x, transpose(weight, self.fan_in_fan_out))
        # ) + mag_norm_scale * lora_B(lora_A(x)) * scaling

        # Note: Computation could potentially be accelerated by using the code below instead of calculating X@W again.
        # This is only correct if dropout=0, otherwise results will differ:
        # https://github.com/huggingface/peft/pull/1474#issuecomment-1964682771
        # bias = self.get_base_layer().bias
        # if bias is not None:
        #     result = result - bias
        # result = mag_norm_scale * result + mag_norm_scale * lora_B(lora_A(x)) * scaling
        # if bias is not None:
        #     result = result + bias

        return result_dora

class Linear(LoraLinear, SealLayer):
    _op = F.linear
    def __init__(self, base_layer, adapter_name, **kwargs):
        # print(kwargs)
        key_config = kwargs.pop("key_config", None)
        base_layer = kwargs.pop("base_layer", base_layer)
        adapter_name = kwargs.pop("adapter_name", adapter_name)
        layer_name = kwargs.pop("layer_name", None)
        self.layer_name = layer_name
        
        super().__init__(base_layer, adapter_name, **kwargs)
        SealLayer.__init__(self, base_layer=base_layer, layer_name=layer_name, key_config=key_config, **kwargs)
        

    def _update_layer(self, adapter_name, r, *args, **kwargs):
        ''' Local `update_layer` function for Linear layer

        Args:
            adapter_name
            r
        '''
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
    
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        '''
        TODO: Implement sealed LoRA (including DoRA)
        '''
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                # TODO: Implement sealed LoRA & DoRA
                x = dropout(x)
                if not self.use_dora[active_adapter]:

                    with torch.profiler.record_function("seal_peft_11_linear_forward_part"):
                        lora_weight = (weight.type_as(x) for weight in self.make_weight(active_adapter))
                        lora_out = self._get_sealed_out(x, lora_weight, result.shape, active_adapter)
                    result = result + lora_out * scaling
                else:
                    x = dropout(x)
                    result = result + self._apply_dora(x, lora_A, lora_B, scaling, active_adapter)

            result = result.to(torch_result_dtype)

        return result


class Embedding(LoraEmbedding, SealLayer):
    _op = F.embedding

    def _update_layer(self, adapter_name, r, *args, **kwargs):
        ''' Local `update_layer` function for Embedding layer

        Args:
            adapter_name
            r
        '''
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)


class Conv2d(LoraConv2d, SealLayer):
    _op = F.conv2d

    def _update_layer(self, adapter_name, r, *args, **kwargs):
        ''' Local `update_layer` function for Embedding layer

        Args:
            adapter_name
            r
        '''
        
        kernel_size = self.kwargs["kernel_size"]
        stride = self.kwargs["stride"]
        padding = self.kwargs["padding"]
        self.lora_A[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = nn.Conv2d(r, self.out_features, 1, bias=False)


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    seal_config: SealConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(seal_config.loftq_config)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(seal_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = seal_config.fan_in_fan_out = False
        kwargs.update(seal_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = seal_config.fan_in_fan_out = True
        kwargs.update(seal_config.loftq_config)
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
