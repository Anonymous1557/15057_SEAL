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

from peft.import_utils import is_bnb_4bit_available, is_bnb_available, is_eetq_available

from .config import SealConfig, KeyConfig
from .layer import Conv2d, Embedding, Linear
from .model import SealModel


__all__ = ["SealConfig", "KeyConfig", "Conv2d", "Embedding", "Linear", "SealModel"]


def __getattr__(name):
    if (name == "Linear8bitLt") and is_bnb_available():
        from .bnb import Linear8bitLt

        return Linear8bitLt

    if (name == "Linear4bit") and is_bnb_4bit_available():
        from .bnb import Linear4bit

        return Linear4bit

    if (name == "EetqLoraLinear") and is_eetq_available():
        from .eetq import EetqLoraLinear

        return EetqLoraLinear

    raise AttributeError(f"module {__name__} has no attribute {name}")
