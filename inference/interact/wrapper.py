from functools import update_wrapper
import requests
from enum import Enum
import numpy as np

HOST = "0.0.0.0"
PORT = "8000"


class UniversalParser(Enum):
    NUMPY = 1
    INT = 2
    LIST = 3

    @classmethod
    def parse(self, object, dst_type):
        if dst_type == UniversalParser.NUMPY:
            return np.array(object)
        elif dst_type == UniversalParser.INT:
            return int(object)
        elif dst_type == UniversalParser.LIST:
            return list(object)


def _wrap(api_url, func_name, func):
    """
    Wraps *func* with additional code.
    """
    # we define a wrapper function. This will execute all additional code
    # before and after the "real" function.

    def wrapped(**kwargs):
        # print("before-call:", func, args, kwargs)
        # print(f"POST requesting to {api_url}")
        cmd = {
            "func_name": func_name,
            "args": kwargs,
        }
        # print(cmd)
        response = requests.post(f"{api_url}", json=cmd)
        # output = func(*args, **kwargs)
        # print("after-call:", func, args, kwargs, output)
        return response.json()
        # return None

    # Use "update_wrapper" to keep docstrings and other function metadata
    # intact
    update_wrapper(wrapped, func)

    # We can now return the wrapped function
    return wrapped


def wrapper(cls, api_url):
    for funcname in dir(cls):
        # We skip anything starting with two underscores. They are most
        # likely magic methods that we don't want to wrap with the
        # additional code. The conditions what exactly we want to wrap, can
        # be adapted as needed.
        if funcname != "__init__":
            if funcname.startswith("__"):
                continue

        # We now need to get a reference to that attribute and check if
        # it's callable. If not it is a member variable or something else
        # and we can/should skip it.
        func = getattr(cls, funcname)
        if not callable(func):
            continue

        # Now we "wrap" the function with our additional code. This is done
        # in a separate function to keep __new__ somewhat clean
        wrapped = _wrap(api_url, funcname, func)

        # After wrapping the function we can attach that new function ont
        # our `Wrapper` instance
        setattr(cls, funcname, wrapped)
    return cls


### USAGE:
# class A:
#     def __init__(self, x):
#         self.a = 1
#         print(x)

#     def forward(self, x):
#         """
#         docstring (to demonstrate `update_wrapper`
#         """
#         print("original forward")
#         return self.a + x

#     def main(self, x):
#         return self.forward(x) + 100


# classA = A

# # print(classA.forward(x=10))
# classA = wrapper(A(5), f"http://{HOST}:{PORT}/api/network/")
# # print(classA.forward(x=10))
