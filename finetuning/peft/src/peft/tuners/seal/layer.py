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

import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose

from peft.tuners.lora.layer import LoraLayer

from .config import KeyConfig

class SEALWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, up, down, const, scale=torch.tensor(1.)):
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
    return ((up @ (const * scale)) @ down)

# ---------------------------------------------------------------------------------------------------------------------
# Most of codes in below are inspired by https://github.com/kohakublueleaf/lycoris
# ---------------------------------------------------------------------------------------------------------------------

from enum import Enum
class LayerType(Enum):
    EMBEDDING = 0
    LINEAR = 1
    CONV2D = 2

class SealLayer(BaseTunerLayer):
    def __init__(self, in_features: int, out_features: int, key_config: KeyConfig, layer_name: str, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({}) # Down block
        self.lora_B = nn.ModuleDict({}) # Up block
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

        self.kwargs = kwargs

        # for seal layer
        self.key_config = key_config
        self.layer_name = layer_name
        self.layer_type: LayerType = None
        self.key_config.assign_adapter(self.layer_name) # TODO: fix for load peft

        self.lora_shape = dict()
        self.orig_shape = dict()
        self.extra_kwargs = dict()

    def _op(self, *args, **kwargs): raise NotImplementedError()
        
    def op(self, x: torch.Tensor, weight: torch.Tensor, orig_shape: torch.Tensor=None, **kwargs) -> torch.Tensor:
        if orig_shape is not None:
            return self._op(x, weight.view(orig_shape).to(x.dtype), **kwargs)
        else:
            return self._op(x, weight.to(x.dtype), **kwargs)

    def make_weight(self, adapter: str) -> torch.Tensor:
        if self.layer_type == LayerType.EMBEDDING:
            lora_A = self.lora_embedding_A[adapter].T # Down block.T -> Down block
            lora_B = self.lora_embedding_B[adapter].T # Up block.T -> Up block
        else:
            lora_A = self.lora_A[adapter].weight # Down block
            lora_B = self.lora_B[adapter].weight # Up block

        dtype = lora_A.dtype
        device = lora_A.device
        
        key = self.key_config.get_now_key_value(self.layer_name).to(device=device, dtype=dtype)
        key_scaling = self.key_config.get_now_key_scale().to(device=device, dtype=dtype)

        if self.layer_type == LayerType.CONV2D: # conv2d
            # https://github.com/KohakuBlueLeaf/lycoris lycoris:modules:locon.py
            return make_weight(
                    lora_B.view(lora_B.size(0), -1), # shape = [out_features, rank] 
                    lora_A.view(lora_A.size(0), -1), # shape = [rank, in_features * product(kernel_size)]
                    key, key_scaling
                )
        else:
            return make_weight(lora_B, lora_A, key, key_scaling)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return self.make_weight(adapter).reshape(self.orig_shape[adapter])

    def _update_layer(
            self, adapter_name,
            r, lora_alpha,
            lora_dropout,
            init_lora_weights,
            layer_type
        ):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.layer_type = layer_type
    
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer

        # Actual trainable parameters
        if r > 0:
            if layer_type == LayerType.EMBEDDING:
                weight_A = torch.randn((r, self.in_features))
                weight_B = torch.randn((self.out_features, r))
                self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A) # Down Block.T, why?
                self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B) # Up Block.T, why?
                self._op = F.embedding
            elif layer_type == LayerType.LINEAR:
                self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
                self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
                self._op = F.linear
            elif layer_type == LayerType.CONV2D:
                extra_kwargs = self.extra_kwargs[adapter_name]
                kernel_size = extra_kwargs["kernel_size"]
                stride = extra_kwargs["stride"]
                padding = extra_kwargs["padding"]
                self.lora_A[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
                self.lora_B[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
                self._op = F.conv2d
            else: raise NotImplementedError()
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(self.weight.device, dtype=weight.dtype)

    def merge(self) -> None:
        if isinstance(self, Embedding) and self.active_adapter not in self.lora_embedding_A.keys(): return
        if isinstance(self, Linear) and self.active_adapter not in self.lora_A.keys(): return
        if isinstance(self, Conv2d) and self.active_adapter not in self.lora_A.keys(): return

        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            now_key = self.key_config._get_now_key()[0]
            # print(f"Merging adapter weight for key : {now_key}")
            self.weight.data += self.get_delta_weight(self.active_adapter)
            self.merged = True

    def unmerge(self) -> None:
        if isinstance(self, Embedding) and self.active_adapter not in self.lora_embedding_A.keys(): return
        if isinstance(self, Linear) and self.active_adapter not in self.lora_A.keys(): return
        if isinstance(self, Conv2d) and self.active_adapter not in self.lora_A.keys(): return
        
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            now_key = self.key_config._get_now_key()[0]
            # print(f"Unmerging adapter weight for key : {now_key}")
            self.weight.data -= self.get_delta_weight(self.active_adapter)[0]
            self.merged = False

    def update_layer_conv2d(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        kernel_size = self.kwargs["kernel_size"]
        stride = self.kwargs["stride"]
        padding = self.kwargs["padding"]
        extra_kwargs = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding
        }

        self.lora_shape.update({adapter_name: (self.out_features, self.in_features*kernel_size[0]*kernel_size[1])})
        self.orig_shape.update({adapter_name: (self.out_features, self.in_features, *kernel_size)})
        self.extra_kwargs.update({adapter_name: extra_kwargs})

        self._update_layer(
            adapter_name=adapter_name,
            r=r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            layer_type=LayerType.CONV2D,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        extra_kwargs = {}
        
        self.lora_shape.update({adapter_name: (self.out_features, self.in_features)})
        self.orig_shape.update({adapter_name: (self.out_features, self.in_features)})
        self.extra_kwargs.update({adapter_name: extra_kwargs})

        self._update_layer(
            adapter_name=adapter_name,
            r=r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            layer_type=LayerType.LINEAR,
        )

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        extra_kwargs = {
            "padding_idx": self.padding_idx,
            "max_norm": self.max_norm,
            "norm_type": self.norm_type,
            "scale_grad_by_freq": self.scale_grad_by_freq,
            "sparse": self.sparse,
        }
        
        self.lora_shape.update({adapter_name: (self.out_features, self.in_features)})
        self.orig_shape.update({adapter_name: (self.out_features, self.in_features)})
        self.extra_kwargs.update({adapter_name: extra_kwargs})

        self._update_layer(
            adapter_name=adapter_name,
            r=r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            layer_type=LayerType.EMBEDDING
        )

    def _init_empty_weights(self, cls, *args, **kwargs) -> None:
        # A helper method that allows to initialize the layer of the given class without spending time to initialize the
        # model weights. The implementation is inspired by
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html but this function cannot be used
        # directly.
        # Instead of this approach, it would be possible to bypass the __init__ of the class but that runs the risk of
        # omitting important logic inside that __init__.
        kwargs = kwargs.copy()
        final_device = kwargs.pop("device", "cpu")
        cls.__init__(self, *args, device="meta", **kwargs)
        self.to_empty(device=final_device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Embedding(nn.Embedding, SealLayer): # same as Lora.layer.Embedding
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        key_config: KeyConfig = None,
        layer_name: str = None,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        # self._init_empty_weights(nn.Embedding, num_embeddings, embedding_dim, **kwargs)/
        SealLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim, key_config=key_config, layer_name=layer_name)
        self.update_layer_embedding(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return transpose(self.lora_embedding_B[adapter] @ self.lora_embedding_A[adapter], True) * self.scaling[adapter]

    def _embed(self, input: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = self.weight if weight is None else weight
        return F.embedding(
            input,
            weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_adapter not in self.lora_embedding_A.keys():
            return self._embed(x)

        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        if self.disable_adapters:
            if (self.r[self.active_adapter] > 0) and self.merged:
                self.unmerge()
            result = self._embed(x)
        elif (self.r[self.active_adapter] == 0) or self.merged:
            result = self._embed(x)
        else:
            scaling = self.scaling[self.active_adapter]
            orig_shape = self.orig_shape[self.active_adapter]
            
            # 0. case of time embedding
            #   x           : [batch_size, in_dim]
            #   lora_weight : [batch_size, out_dim, in_dim] 
            #   lora_out    : [batch_size, out_dim]
            #   einsum      : [batch_size, in_dim], [batch_size, out_dim, in_dim]
            #                   -> [batch_size, out_dim]
            #   reshape     : [batch_size, out_dim]
            
            # lora_out : [batch_size, out_dim, in_dim, *kernel_size] | kernel_size = kernel_size or ()
            # orig_out : [?, out_dim, *orig_extra_dim], ? could be None or batch_size or multiple of batch_size
            
            # same as Linear forward
            dtype, device = x.dtype, x.device
            
            kwargs = self.extra_kwargs[self.active_adapter]
            orig_out: torch.Tensor = self.op(x, self.weight, orig_shape, **kwargs)
            lora_weight: torch.Tensor = self.make_weight(self.active_adapter).to(device=device, dtype=dtype)
            
            batch_size = lora_weight.size()[0]

            x = x.view(batch_size, -1, *x.size()[1:])
            lora_out = torch.einsum("b...i, boi -> b...o", x, lora_weight)

            lora_out = lora_out.squeeze()
            lora_out = lora_out.reshape(-1, *orig_out.shape[1:])
            
            result = (orig_out + lora_out) * scaling
        return result


class Linear(nn.Linear, SealLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        key_config: KeyConfig = None,
        layer_name: str = None,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        # this gets the init from nn.Linear's super perspective, i.e.
        # nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        # Note that we don't use self._init_empty_weights() for Linear because it is a bit slower and the benefit of
        # added robustness is not big enough for Linear.

        SealLayer.__init__(self, in_features=in_features, out_features=out_features, key_config=key_config, layer_name=layer_name)
        # Freezing the pre-trained weight matrix

        self.fan_in_fan_out = fan_in_fan_out

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def _linear(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.fan_in_fan_out:
            return self.op(input, self.weight, self.weight.T.shape, bias=self.bias)
        else:
            return self.op(input, self.weight, self.weight.shape, bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_adapter not in self.lora_A.keys():
            return self._linear(x)

        previous_dtype = x.dtype

        if self.disable_adapters:
            if (self.r[self.active_adapter] > 0) and self.merged:
                self.unmerge()
            result = self._linear(x)
        elif (self.r[self.active_adapter] == 0) or self.merged:
            result = self._linear(x)
        else:
            dropout = self.lora_dropout[self.active_adapter]
            scaling = self.scaling[self.active_adapter]

            orig_shape = self.orig_shape[self.active_adapter]

            x = x.to(self.lora_A[self.active_adapter].weight.dtype)
            x = dropout(x)

            # ---------------------------------------------------------------------------------------------------------------------
            # 1. case of text encoder
            #   x           : [batch_size*prompt_block, token_length, in_dim]
            #   lora_weight : [batch_size, out_dim, in_dim] 
            #   lora_out    : [batch_size*prompt_block, token_length, out_dim]
            #   einsum      : [batch_size, prompt_block, token_length, in_dim], [batch_size, out_dim, in_dim]
            #                   -> [batch_size, prompt_block, token_length, out_dim]
            #   reshape     : [batch_size*prompt_block, token_length, out_dim]
            # 2. case of linear layer
            #   x           : [batch_size, ..., in_dim]
            #   lora_weight : [batch_size, out_dim, in_dim]
            #   lora_out    : [batch_size, ..., out_dim]
            #   einsum      : [batch_size, ..., in_dim], [batch_size, out_dim, in_dim]
            #                   -> [batch_size, ..., out_dim]
            #   reshape     : [batch_size, ..., out_dim]
            # ---------------------------------------------------------------------------------------------------------------------
            # lora_out : [batch_size, out_dim, in_dim, *kernel_size] | kernel_size = kernel_size or ()
            # orig_out : [?, out_dim, *orig_extra_dim], ? could be None or batch_size or multiple of batch_size
            # ---------------------------------------------------------------------------------------------------------------------
            
            dtype, device = x.dtype, x.device
            
            kwargs = self.extra_kwargs[self.active_adapter]
            orig_out: torch.Tensor = self.op(x, self.weight, orig_shape, **kwargs)
            lora_weight = self.make_weight(self.active_adapter).to(device=device, dtype=dtype)
            
            batch_size = lora_weight.size()[0]

            x = x.view(batch_size, -1, *x.size()[1:])
            lora_out = torch.einsum("b...i, boi -> b...o", x, lora_weight)

            lora_out = lora_out.squeeze()
            lora_out = lora_out.reshape(-1, *orig_out.shape[1:])
            
            result = (orig_out + lora_out) * scaling
            
        result = result.to(previous_dtype)
        return result


class Conv2d(nn.Conv2d, SealLayer):
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        adapter_name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        key_config: KeyConfig = None,
        layer_name: str = None,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        # self._init_empty_weights(nn.Conv2d, in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        SealLayer.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            key_config=key_config,
            layer_name=layer_name
        )

        self.update_layer_conv2d(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def _conv2d(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_adapter not in self.lora_A.keys():
            return self._conv2d(x)

        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = self._conv2d(x)
        elif (self.r[self.active_adapter] == 0) or self.merged:
            result = self._conv2d(x)
        else:
            dropout = self.lora_dropout[self.active_adapter]
            scaling = self.scaling[self.active_adapter]

            x = x.to(self.lora_A[self.active_adapter].weight.dtype)
            x = dropout(x)
            orig_shape = self.orig_shape[self.active_adapter]
            
            # conv2d batch by batch
            # ---------------------------------------------------------------------------------------------------------------------
            # x           : [batch_size, in_dim, *x_extra_dim]
            # lora_weight : [batch_size, out_dim, in_dim, *kernel_size] 
            # lora_out    : [batch_size, out_dim, *orig_extra_dim]
            # conv2d      : [1, batch_size*in_dim, *x_extra_dim] ⋆ [batch_size*out_dim, in_dim, *kernel_size] by groups=batch_size
            #                 -> [1, batch_size*out_dim, *orig_extra_dim]
            # reshape     : [batch_size, out_dim, *orig_extra_dim]
            # ---------------------------------------------------------------------------------------------------------------------
            # lora_out : [batch_size, out_dim, in_dim, *kernel_size] | kernel_size = kernel_size or ()
            # orig_out : [?, out_dim, *orig_extra_dim], ? could be None or batch_size or multiple of batch_size
            # ---------------------------------------------------------------------------------------------------------------------
            
            dtype, device = x.dtype, x.device
            
            kwargs = self.extra_kwargs[self.active_adapter]
            orig_out: torch.Tensor = self.op(x, self.weight, orig_shape, **kwargs)
            lora_weight = self.make_weight(self.active_adapter).to(device=device, dtype=dtype)
            
            batch_size = lora_weight.size()[0]
            x = x.view(1, -1, *x.size()[2:]) # [batch_size, in_dim, *x_extra_dim] -> [1, batch_size*in_dim, *x_extra_dim] : pseudo-1-batch input
            
            lora_weight = lora_weight.view(-1, *lora_weight.size()[2:]) # [batch_size*out_dim, in_dim, *kernel_size]
            kwargs = self.extra_kwargs[self.active_adapter].copy()
            kwargs['groups'] = batch_size
            lora_out = F.conv2d(x, lora_weight, **kwargs)

            lora_out = lora_out.squeeze()
            lora_out = lora_out.reshape(-1, *orig_out.shape[1:])

            result = (orig_out + lora_out) * scaling
    
        result = result.to(previous_dtype)
        return result
