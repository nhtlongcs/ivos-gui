import logging
from functools import wraps

import torch
from fastapi import FastAPI

from inference.interact.fbrs_controller import FBRSController
from inference.interact.s2m.s2m_network import deeplabv3plus_resnet50 as S2M
from inference.interact.s2m_controller import S2MController
from model.network import XMem


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


network = None
s2m_model = None
s2m_controller = None
fbrs_controller = None

s2m_model_path = "saves/s2m.pth"
fbrs_model_path = "saves/fbrs.pth"
network_path = "saves/XMem.pth"


@arg_logger
@app.post("/api/network/")
def network_interact(request: dict):
    global network
    assert "func_name" in request
    request.setdefault("args", {})
    if request.get("args", None) is None:
        request["args"] = {}
    if network is None and request["func_name"] == "__init__":
        print("loading network")
        network = XMem(model_path=network_path, **request.get("args", {}))
        print("network loaded")
    else:
        network.__getattribute__(request["func_name"])(**request.get("args", {}))
    return {"code": 0, "status": "Done"}


@arg_logger
@app.post("/api/s2m/")
def s2m_interact(request: dict):
    global s2m_controller
    assert "func_name" in request
    request.setdefault("args", {})
    if request.get("args", None) is None:
        request["args"] = {}
    if s2m_controller is None and request["func_name"] == "__init__":
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
        s2m_controller.__getattribute__(request["func_name"])(**request.get("args", {}))
    return {"code": 0, "status": "Done"}


@arg_logger
@app.post("/api/fbrs/")
def fbrs_interact(request: dict):
    global fbrs_controller
    assert "func_name" in request
    request.setdefault("args", {})
    if request.get("args", None) is None:
        request["args"] = {}
    if fbrs_controller is None and request["func_name"] == "__init__":
        fbrs_controller = FBRSController(checkpoint_path=fbrs_model_path)
        print("network loaded")
    else:
        fbrs_controller.__getattribute__(request["func_name"])(
            **request.get("args", {})
        )
    return {"code": 0, "status": "Done"}
