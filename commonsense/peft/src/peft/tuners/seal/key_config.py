from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import List, Union

# for SEAL keys
import torch
import numpy as np


IDENTITY_KEY = "identity"


@dataclass
class KeyConfig:
    data: dict                  = field(init=False)
    adpater_mapping: dict       = field(init=False)
    now_key: None               = field(init=False)
    data_key_mapping: dict      = field(init=False)
    r: int                      = field(init=False)
    training: bool              = field(init=False)
    
    def __post_init__(self):
        """create KeyConfig for SealConfig
        """
        self.data = dict()
        """self.data = {
            key_path: {
                key_scale: float
                key_value: torch.Tensor
                key_mapping: dict{
                    adapter_name: index_or_num
                }
            }, ...
        }"""
        self.adpater_mapping = dict()
        """self.adapter_mapping ={
            key_path: {
                adapter_name1: index_or_num,
                adapter_name2: index_or_num, 
                ...
            },  or
            key_path: None,
            ...
        }
        """
        self.now_key: Iterable = None
        self.data_key_mapping: dict = dict()
        self.r: int = None
        self.training: bool = False

    def train(self): self.training = True
    def eval(self): self.training = False    

    def append(self, r: int, key_path: str, scale: float, mapping: dict=None, **kwargs) -> None:
        """append key data for KeyConfig  
            r = rank
            key_path: str = path to key
            scale: float = scale of key
            mapping: dict = specific map for adapter_names
            **kwargs: dict = dictionary which contians additional information of key data"""
        if self.r is None: self.r = r
        else: assert self.r == r

        key_data = dict()
        key_data["key_scale"] = torch.tensor(scale, requires_grad=False)
        key_data["key_value"] = torch.from_numpy(load_key_value_from_path(r, key_path))
        if mapping is not None:
            self.adpater_mapping[key_path] = mapping
            key_data["key_mapping"] = None
        else:
            self.adpater_mapping[key_path] = None
            key_data["key_mapping"] = dict()
        
        self.data[key_path] = key_data
        self.data[key_path].update(**kwargs)

    def append_identity(self, **kwargs):
        assert self.r is not None
        assert IDENTITY_KEY not in self.data.keys(), "already identity exists"

        identity_data = dict()
        identity_data["key_scale"] = torch.tensor(1.0)
        identity_data["key_value"] = torch.eye(self.r, dtype=torch.float).unsqueeze(0)
        
        self.adpater_mapping[IDENTITY_KEY] = None
        identity_data["key_mapping"] = dict()

        self.data[IDENTITY_KEY] = identity_data
        self.data[IDENTITY_KEY].update(**kwargs)

    def assign_adapter(self, adapter_name) -> None:
        """assign adapter_name for KeyConfig.
        It will be used in seal.layer.Linear or seal.layer.Conv2d
        """
        for key_path in self.data.keys():
            if hasattr(self.adpater_mapping, key_path):
                mapping: dict = self.adpater_mapping[key_path]
                assert adapter_name in mapping.keys()
            else:
                mapping: dict = self.data[key_path]["key_mapping"]
                if adapter_name in mapping: continue # or assertion?
                else: mapping[adapter_name] = len(mapping)

    def assign_dataset(self, dataset_identifiers: List[Union[list, tuple]], key_path: str) -> None:
        """assign dataset for key_path"""
        assert key_path in self.data.keys()
        assert any(isinstance(dataset_identifiers, ins) for ins in [list, tuple])
        for data in dataset_identifiers:
            self.data_key_mapping[data] = key_path
        
    def set_now_dataset(self, dataset_identifiers: Iterable) -> None:
        """set dataset as now_key"""
        assert all(dataset_id in self.data_key_mapping.keys() for dataset_id in dataset_identifiers), f"{self.data_key_mapping}, {dataset_identifiers}"
        
        self.now_key = [self.data_key_mapping[dataset_id] for dataset_id in dataset_identifiers]

    def get_now_key_value(self, adapter_name: str) -> torch.Tensor:
        """return stacked value which has shape == [batch, rank, rank] corresponding adapter_name"""
        temp_stack = []
        if len(self._get_now_key()) == 1:
            key_path = self._get_now_key()[0]
            # key = self.data_key_mapping[key_path]
            value = self._get_key_value(key_path, adapter_name)
            temp_stack.append(value)
        else:
            for key_path in self._get_now_key():
                if self.training:
                    # key = self.data_key_mapping[key_path]
                    value = self._get_key_value(key_path, adapter_name)
                else:
                    value = self._get_key_value(key_path, adapter_name)
                temp_stack.append(value)

        temp_stack = torch.stack(temp_stack)
        if len(temp_stack.size()) == 2: temp_stack = temp_stack.unsqueeze(0)
        assert len(temp_stack.size()) == 3
        return temp_stack

    def get_now_key_scale(self) -> torch.Tensor:
        """return stacked scale which has shape == [batch, 1]"""
        temp_stack = []
        if len(self._get_now_key()) == 1:
            key_path = self._get_now_key()[0]
            # key = self.data_key_mapping[key_path]
            scale = self._get_key_scale(key_path)
            temp_stack.append(scale)
        else:
            for key_path in self._get_now_key():
                if self.training:
                    # key = self.data_key_mapping[key_path]
                    scale = self._get_key_scale(key_path)
                else:
                    scale = self._get_key_scale(key_path)
                temp_stack.append(scale)
            
        temp_stack = torch.stack(temp_stack)
        if len(temp_stack.size()) == 1: temp_stack = temp_stack.unsqueeze(1).unsqueeze(2)
        elif len(temp_stack.size()) == 2: temp_stack = temp_stack.unsqueeze(2)
        assert len(temp_stack.size()) == 3
        return temp_stack

    def _get_key_value(self, key_path, adapter_name) -> torch.Tensor:
        """return a key corresponding key_path and adapter_name"""
        if hasattr(self.adpater_mapping, key_path):
            mapping = self.adpater_mapping[key_path]
        else:
            mapping = self.data[key_path]["key_mapping"]
        key_value = self.data[key_path]["key_value"]
        value_length = key_value.shape[0]
        adapter_index = mapping[adapter_name]
        
        target_idx = adapter_index % value_length

        return key_value[target_idx]

    def _get_key_scale(self, key_path) -> float:
        """return a key scale corresponding key_path"""
        return self.data[key_path]["key_scale"]

    def _get_now_key_value(self, adpater_name) -> torch.Tensor:
        now_key = self._get_now_key()
        return self._get_key_value(now_key, adpater_name)

    def _get_now_key_scale(self) -> float:
        now_key = self._get_now_key()
        return self._get_key_scale(now_key)

    def _get_now_key(self) -> Iterable:
        """return now_key"""
        assert self.now_key is not None
        return self.now_key
    
    def _set_now_key(self, now_key: Iterable) -> None:
        """set now_key"""
        assert all(key for key in now_key if key in self.data.keys())
        self.now_key = now_key

    def to_dict(self):
        import json
        from copy import deepcopy
        data = deepcopy(self.data)
        for k in data.keys():
            if "key_value" in data[k]:
                del data[k]["key_value"]
            data[k]["key_scale"] = float(data[k]["key_scale"])
        return data
            
    def load(self, path: str):
        raise NotImplementedError()

# ---------------------------------------------------------------------------
# part of loading key value
# ---------------------------------------------------------------------------

IMAGE_EXT = (".png", ".jpeg", ".jpg", ".webp", ".bmp", ".gif")
MOV_EXT = (".mp4", )
ACCEPTED_EXT = (".npy", *IMAGE_EXT, *MOV_EXT)

def load_key_value_from_path(r, key_path) -> np.ndarray:
    """load key file from key path and crop and resize to r"""
    from PIL import Image, ImageSequence
    if key_path.lower().endswith(".npy"): # for any type of constants
        values = np.load(key_path).astype(np.float32)
        print("[SEAL] load key(s) from npy file.")
    elif key_path.lower().endswith(IMAGE_EXT): # for one image
        # use PIL and argumentation or transforms.
        image = Image.open(key_path) # convert grayscale image
        frames = []
        for frame in ImageSequence.Iterator(image):
            frame = frame.convert("L")
            frame = center_crop_and_resize(frame, r)
            frames.append(np.asarray(frame))
        values = np.asarray(frames, dtype=np.float32)

        values = (values/127.5) - 1.0 # [0, 255] -> [-1., 1.]
        print(f"[SEAL] load key from one image. number of frames : {image.n_frames}")
        del image, frames
    elif key_path.lower().endswith(MOV_EXT): # use cv2 to read
        import cv2
        cap = cv2.VideoCapture(key_path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frame = 1 if count < 100 else count // 100 # cap for many frame video
        frame_count = 0
        frames = []
        while (cap.isOpened()):
            if (frame_count % skip_frame) == 0:
                ret, frame = cap.read()
                if ret is None or frame is None:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = Image.fromarray(frame)
                frame = center_crop_and_resize(frame, r)
                frame = (image/127.5) - 1.0 # [0, 255] -> [-1., 1.]
                frames.append(frame)
            else:
                cap.grab()
            frame_count += 1
        cap.release()
        del cv2, cap # to function
        values = np.asarray(frames, dtype=np.float32)
        print(f"[SEAL] load keys from video. total number of key is {int(values.shape[0])}")
    else:
        ext = key_path.split(".")[-1]
        raise KeyError(f"[SEAL] not accepted format of keys. but got {ext} file.")

    if len(values.shape) == 2: values = np.expand_dims(values, axis=0)
    elif len(values.shape) == 3: pass
    else: raise ValueError(f"wrong type or shape of key, expected length of shape is 2 or 3, but got {values.shape}")

    return values


def center_crop_and_resize(image, dim):
    # get margin, crop and resize image
    width, height = image.size
    # print(width, height, w_margin, h_margin) # debug
    target_edge = min(width, height)
    w_margin = (width - target_edge) // 2 if width > target_edge else 0
    h_margin = (height - target_edge) // 2 if height > target_edge else 0
    image = image.crop((w_margin, h_margin, w_margin+target_edge, h_margin+target_edge))
    image = image.resize((dim, dim))
    return image
