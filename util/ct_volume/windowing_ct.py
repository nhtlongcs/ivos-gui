import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import List 
import cv2
import progressbar


cfg = {
    "chest": {"lungs": [{"W": 1500, "L": -600}], "mediastinum": [{"W": 350, "L": 50}]},
    "abdomen": {"soft tissues": [{"W": 400, "L": 50}], "liver": [{"W": 150, "L": 30}]},
    "spine": {"soft tissues": [{"W": 250, "L": 50}], "bone": [{"W": 1800, "L": 400}]},
}

def get_cfg(key, cfg=cfg):
    if isinstance(key, str):
        if "-" in key:
            key = key.split("-")
            key = [k.strip() for k in key]

        for k in key:
            if k in cfg:
                cfg = cfg[k]
            else:
                raise ValueError(f"{k} not in cfg")
        return cfg
    
    if isinstance(key, List):
        res = []
        for c in key:
            res += get_cfg(c, cfg)
        return res

def load_ct_info(file_path):
    sitk_image = sitk.ReadImage(file_path)
    if sitk_image is None:
        res = {}
    else:
        origin = sitk_image.GetOrigin() # original used list(reversed(, dont know why
        spacing = sitk_image.GetSpacing()  # original used list(reversed(, dont know why
        direction = sitk_image.GetDirection()
        subdirection = [direction[8], direction[4], direction[0]]
        res = {"sitk_image": sitk_image,
               "npy_image": sitk.GetArrayFromImage(sitk_image),
               "origin": origin,
               "spacing": spacing,
               "direction": direction,
               "subdirection": subdirection}
    return res

def change_axes_of_image(npy_image, orientation):
    '''default orientation=[1, -1, -1]'''
    if orientation[0] < 0:
        npy_image = np.flip(npy_image, axis=0)
    if orientation[1] > 0:
        npy_image = np.flip(npy_image, axis=1)
    if orientation[2] > 0:
        npy_image = np.flip(npy_image, axis=2)
    return npy_image

def convert_2_npy(vol_path):
    image_dict = load_ct_info(vol_path)
    subdirection = image_dict["subdirection"]

    image_dict["npy_image"] = change_axes_of_image(
        image_dict["npy_image"], subdirection
    )
    npy_image = image_dict["npy_image"]
    return npy_image
    
def windowing(npy_image):

    queries = [
        ["spine-bone"],
        ["chest-lungs", "chest-mediastinum"],
        ["abdomen-soft tissues", "abdomen-liver"],
    ]

    stacked = []
    for q in queries:
        c = get_cfg(q)

        args = {
            "name": '_'.join(q),
            "window_level": [x["L"] for x in c],
            "window_width": [x["W"] for x in c],
        }

        WINDOW_LEVEL = [x["L"] for x in c]
        WINDOW_WIDTH = [x["W"] for x in c]

        window_min = None 
        window_max = None
        if isinstance(WINDOW_LEVEL, List) and isinstance(WINDOW_WIDTH, List):
            for i, (l, w) in enumerate(zip(WINDOW_LEVEL, WINDOW_WIDTH)):
                window_min = l - (w // 2) if window_min is None else min(window_min, l - (w // 2))
                window_max = l + (w // 2) if window_max is None else max(window_max, l + (w // 2))
        elif isinstance(WINDOW_LEVEL, int) and isinstance(WINDOW_WIDTH, int):
            window_min = WINDOW_LEVEL - (WINDOW_WIDTH // 2)
            window_max = WINDOW_LEVEL + (WINDOW_WIDTH // 2)
        else: 
            raise ValueError("WINDOW_LEVEL and WINDOW_WIDTH must be int or list of int")

        img = np.clip(npy_image, window_min, window_max)
        img = 255 * ((img - window_min) / (window_max - window_min))
        img = img.astype(np.uint8)
        stacked.append(img)
    
    stacked = np.stack(stacked, axis=-1)

    return stacked
    

def windowing_ct(volume_path, out_dir):
    print("Processing test files")
    test_fileid = osp.basename(volume_path).split(".nii.gz")[0]    
    npy_image = convert_2_npy(volume_path)

    processed = windowing(npy_image)

    print(f'Extracting frames from {volume_path} into {out_dir}...')
    bar = progressbar.ProgressBar(max_value=processed.shape[0])

    # write to output_dir
    for frame_index, slice in enumerate(processed):
        cv2.imwrite((osp.join(out_dir, f"{str(frame_index).zfill(4)}.jpg")), slice)
        bar.update(frame_index)
        bar.finish()
