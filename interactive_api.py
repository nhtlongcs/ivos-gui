import logging
from functools import wraps

import torch
from fastapi import FastAPI

from inference.inference_core import InferenceCore
from inference.interact.fbrs_controller import FBRSController
from inference.interact.s2m.s2m_network import deeplabv3plus_resnet50 as S2M
from inference.interact.s2m_controller import S2MController
from inference.device import detach
from model.network import XMem

from stcn.networks.eval_network import STCNEval
from stcn.inference.inference_core_new import InferenceCoreNew 


import pickle
import numpy as np

# https://stackoverflow.com/questions/23983150/how-can-i-log-a-functions-arguments-in-a-reusable-way-in-python
def arg_logger(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        saved_args = locals()
        try:
            return func(*args, **kwargs)
        except:
            logging.exception("Oh no! My args were: " + str(saved_args))
            raise

    return new_func


app = FastAPI()


processor = None
s2m_model = None
s2m_controller = None
fbrs_controller = None

s2m_model_path = "saves/s2m.pth"
fbrs_model_path = "saves/fbrs.pth"
network_path = "saves/XMem.pth"
stcn_path = "saves/stcn_flare22.pth"
# get attribute from a.b.c


def custom_serializer(obj):
    if isinstance(obj, (bool, int, str, float)):
        return obj
    elif isinstance(obj, np.ndarray):
        return pickle.dumps(obj).decode("latin-1")
    elif torch.is_tensor(obj):
        # return pickle.dumps(obj)
        return custom_serializer(detach(obj.cpu()).numpy())
    elif isinstance(obj, dict):
        return {k: custom_serializer(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [custom_serializer(v) for v in obj]
    elif obj is None:
        return obj
    raise TypeError("Type not serializable", type(obj))


def get_attr(obj, attr_str):
    attr_list = attr_str.split(".")
    for attr in attr_list:
        obj = getattr(obj, attr)
    return obj


@arg_logger
@app.post("/api/network/")
def core_interact(request: dict):
    global processor
    if "var_name" in request:
        try:
            result = get_attr(processor, request["var_name"])
            return {"value": result, "code": 0}
        except:
            return {"value": "error", "code": -1}

    assert "func_name" in request
    request.setdefault("args", {})
    if request.get("args", None) is None:
        request["args"] = {}
    if request["func_name"] == "__init__":
        if processor is None:
            print("loading InferenceCore")
            # network = XMem(model_path=network_path, **request.get("args", {}))
            # processor = InferenceCore(network=network, **request.get("args", {}))
            
            device = torch.device('cuda')
            network = STCNEval(
                key_backbone="resnet50-mod",
                value_backbone="resnet18-mod",
                pretrained=False
            ).to(device).eval()

            state_dict = torch.load(stcn_path)["model"]
            for k in list(state_dict.keys()):
                if k == "value_encoder.conv1.weight":
                    if state_dict[k].shape[1] == 2:
                        pads = torch.zeros((64, 1, 7, 7), device=state_dict[k].device)
                        state_dict[k] = torch.cat([state_dict[k], pads], 1)
            
            network.load_state_dict(state_dict)

            processor = InferenceCoreNew(network=network, config={
                'mem_every': 5,
                'top_k': 20,
                'max_k': 200,
                'num_objects': 14,
                'device': 'cuda',
                'strategy': 'argmax',
                'include_last': True ,
            })
            print("InferenceCore loaded")

    else:
        result = processor.__getattribute__(request["func_name"])(
            **request.get("args", {})
        )
        result = custom_serializer(result)
        return {"code": 0, "result": result}
    return {"code": 0, "result": None}


@arg_logger
@app.post("/api/s2m/")
def s2m_interact(request: dict):
    global s2m_controller
    assert "func_name" in request
    request.setdefault("args", {})
    if request.get("args", None) is None:
        request["args"] = {}
    if request["func_name"] == "__init__":
        if s2m_controller is None:
            print("loading network")
            if s2m_model_path is not None:
                s2m_saved = torch.load(s2m_model_path)
                s2m_model = S2M().cuda().eval()
                s2m_model.load_state_dict(s2m_saved)
            else:
                s2m_model = None
            s2m_controller = S2MController(s2m_net=s2m_model, **request.get("args", {}))
            print("network loaded")
    else:
        result = s2m_controller.__getattribute__(request["func_name"])(
            **request.get("args", {})
        )
        result = custom_serializer(result)
        return {"code": 0, "result": result}
    return {"code": 0, "result": None}


@arg_logger
@app.post("/api/fbrs/")
def fbrs_interact(request: dict):
    global fbrs_controller
    assert "func_name" in request
    request.setdefault("args", {})
    if request.get("args", None) is None:
        request["args"] = {}
    if request["func_name"] == "__init__":
        if fbrs_controller is None:
            print("loading network")
            fbrs_controller = FBRSController(checkpoint_path=fbrs_model_path)
            print("network loaded")

    else:
        result = fbrs_controller.__getattribute__(request["func_name"])(
            **request.get("args", {})
        )
        result = custom_serializer(result)
        return {"code": 0, "result": result}
    return {"code": 0, "result": None}
