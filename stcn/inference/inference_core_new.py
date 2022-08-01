import cv2
import torch
import pickle
import numpy as np
from typing import Dict
import torch.nn.functional as F
from stcn.inference.inference_memory_bank_efficient import MemoryBankWithFlush
from stcn.networks.eval_network import STCNEval
from stcn.utilities.aggregate import aggregate
from stcn.utilities.tensor_util import pad_divide_by, unpad
from stcn.utilities.referencer import Referencer

REFERENCER = Referencer()

"""
NEEDED
self.processor.set_all_labels(all_labels=list(range(1, self.num_objects + 1)))

self.work_mem_min.setValue(self.processor.get_value("memory.min_mt_frames"))
self.work_mem_max.setValue(self.processor.get_value("memory.max_mt_frames"))
self.long_mem_max.setValue(self.processor.get_value("memory.max_long_elements"))
self.num_prototypes_box.setValue(
    self.processor.get_value("memory.num_prototypes")
)
self.mem_every_box.setValue(self.processor.get_value("mem_every"))

self.current_prob = self.processor.step(image=self.current_image_torch)

max_work_elements = self.processor.get_value("memory.max_work_elements")
max_long_elements = self.processor.get_value("memory.max_long_elements")

curr_work_elements = self.processor.get_value("memory.work_mem.size")
curr_long_elements = self.processor.get_value("memory.long_mem.size")

self.processor.update_config(config=self.config)

self.processor.clear_memory()


"""


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class WorkaroundMemory(AttrDict):
    min_mt_frames = 0
    max_mt_frames = 0
    num_prototypes = 0
    max_long_elements = 1
    max_work_elements = 1

    work_mem = AttrDict(size=0)
    long_mem = AttrDict(size=0)

class InferenceCoreNew:
    """
    Inference module, which performs iterative propagation
    """

    def __init__(
        self,
        network: STCNEval,
        config: dict
    ):
        print("Initiate Inference Core")
        self.network = network
        self.mem_every = config['mem_every']
        self.include_last = config['include_last']
        self.device = config['device']
        self.k = config['num_objects']
        self.top_k = config['top_k']
        self.max_k = config['max_k']

        self.memory = MemoryBankWithFlush(k=self.k-1, top_k=self.top_k, max_k=self.max_k)
        # self.memory = WorkaroundMemory()

        self.clear_memory()

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory.flush()

    def encode_key(self, image):
        result = self.network.encode_key(image.to(self.device))
        return result

    def _encode_masks(self, masks):
        """
        Input masks from _load_mask(), but in shape [B, H, W]
        Output should be one-hot encoding of segmentation masks [B, NC, H, W]
        """

        one_hot = torch.nn.functional.one_hot(
            masks.long(), num_classes=self.k
        )  # (H,W,NC)
        one_hot = one_hot.permute(2, 0, 1)  # (NC,H,W)
        return one_hot.float()

    def efficient_encode(self, ref_frames):
        msk = self._encode_masks(ref_frames)
        msk = msk[1:].unsqueeze(1)
        return msk


    def interact(self, frame, mask):
        prob = aggregate(mask, keep_bg=True)
        # KV pair for the interacting frame
        key_k, _, qf16, _, _ = self.encode_key(frame)
        key_v = self.network.encode_value(
            frame.to(self.device),
            qf16,
            prob[1:].to(self.device),
        )
        key_k = key_k.unsqueeze(2)

        # Propagate
        self.memory.add_memory(key_k, key_v)

    def do_pass(self, frame, is_mem_frame):

        k16, qv16, qf16, qf8, qf4 = self.encode_key(frame)

        out_mask = self.network.segment_with_query(
            self.memory, qf8, qf4, k16, qv16
        )
        out_mask = aggregate(out_mask, keep_bg=True)
        if self.include_last or is_mem_frame:
            prev_value = self.network.encode_value(
                frame.to(self.device), qf16, out_mask[1:].to(self.device)
            )
            prev_key = k16.unsqueeze(2)
            self.memory.add_memory(
                prev_key, prev_value, is_temp=not is_mem_frame
            )

        return out_mask

    #### NEW METHODS

    def update_config(self, config):
        self.mem_every = config['mem_every']
        self.include_last = config['include_last']
        self.memory.update_config(config)

    def set_all_labels(self, all_labels):
        self.all_labels = all_labels

    def step(self, image, mask=None, valid_labels=None, end=False):
        # image: 3*H*W
        # mask: num_objects*H*W or None
        if isinstance(image, str):
            image_np = pickle.loads(image.encode("latin-1"))
            ori_c, ori_h, ori_w = image_np.shape
            # resized = cv2.resize(image_np.transpose(1,2,0), (512,512), 0, 0, interpolation = cv2.INTER_NEAREST)
            # resized = resized.transpose(2,1,0)
            image = torch.from_numpy(image_np)
        else:
            ori_c, ori_h, ori_w = image.shape

        if isinstance(mask, str):
            mask_np = pickle.loads(mask.encode("latin-1"))
            # resized = cv2.resize(np.squeeze(mask_np, 0), (512,512), 0, 0, interpolation = cv2.INTER_NEAREST)
            # resized = np.expand_dims(resized, 0)
            mask = torch.from_numpy(mask_np)

        self.curr_ti += 1
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0)  # add the batch dimension

        is_mem_frame = (
            (self.curr_ti - self.last_mem_ti >= self.mem_every) or (mask is not None)
        ) and (not end)

        need_segment = (self.curr_ti > 0) and (
            (valid_labels is None) or (len(self.all_labels) != len(valid_labels))
        )

        if need_segment:
            result = self.do_pass(image, is_mem_frame)
        else:
            result = None

        if mask is not None:
            mask, _ = pad_divide_by(mask, 16)
            mask = mask.unsqueeze(1)
            # NC, 1 , H, W 
            self.interact(image, mask)
            result = aggregate(mask, dim=0, keep_bg=True)
        
        if is_mem_frame:
            self.last_mem_ti = self.curr_ti

        if result is not None:
            result = unpad(result, self.pad) # (num_obj + 1, H, W)
            result = result.squeeze()

        return result