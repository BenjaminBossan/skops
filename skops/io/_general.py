from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import FunctionType
from typing import Any, Type, Union
from zipfile import ZipFile

import numpy as np

from ._types import State
from ._utils import _get_instance, _get_state, _import_obj, get_module, gettype
from .exceptions import UnsupportedTypeException

KeyVal = Union[np.generic, bool, int, float, complex, str, bytes, memoryview]


@dataclass
class StateList(State):
    content: list[State]


@dataclass
class StateDict(State):
    key_types: StateList
    content: dict[str, State]


@dataclass
class StateTuple(State):
    content: tuple[Any, ...]


@dataclass
class _ContentFunction:
    module_path: str
    function: str


@dataclass
class StateFunction(State):
    content: _ContentFunction


@dataclass
class _ContentPartial:
    func: StateFunction
    args: StateTuple
    kwds: StateDict
    namespace: str | None


@dataclass
class StatePartial(State):
    cls: str
    content: _ContentPartial


@dataclass
class StateObject(State):
    content: StateDict | None = None


def dict_get_state(obj: dict[KeyVal, Any], dst: str) -> StateDict:
    key_types = _get_state([type(key) for key in obj.keys()], dst)
    content = {}
    for key_, value in obj.items():
        if isinstance(value, property):
            continue
        if np.isscalar(key_) and hasattr(key_, "item"):
            # convert numpy value to python object
            assert isinstance(key_, np.generic)
            key = key_.item()
        else:
            key = key_

        content[key] = _get_state(value, dst)

    state = StateDict(
        cls=obj.__class__.__name__,
        module=get_module(type(obj)),
        content=content,
        key_types=key_types,
    )
    return state


def dict_get_instance(state: StateDict, src: ZipFile) -> dict[KeyVal, Any]:
    content = gettype(state)()
    key_types = _get_instance(state.key_types, src)
    for k_type, item in zip(key_types, state.content.items()):
        content[k_type(item[0])] = _get_instance(item[1], src)
    return content


def list_get_state(obj: list[Any], dst: str) -> StateList:
    content = []
    for value in obj:
        content.append(_get_state(value, dst))

    state = StateList(
        cls=obj.__class__.__name__,
        module=get_module(type(obj)),
        content=content,
    )
    return state


def list_get_instance(state: StateList, src: ZipFile) -> list[Any]:
    content = gettype(state)()
    for value in state.content:
        content.append(_get_instance(value, src))
    return content


def tuple_get_state(obj: tuple[Any], dst: str) -> StateTuple:
    content = tuple(_get_state(value, dst) for value in obj)
    state = StateTuple(
        cls=obj.__class__.__name__,
        module=get_module(type(obj)),
        content=content,
    )
    return state


def tuple_get_instance(state: StateTuple, src: ZipFile) -> tuple[Any, ...]:
    # Returns a tuple or a namedtuple instance.
    def isnamedtuple(t: Type) -> bool:
        # This is needed since namedtuples need to have the args when
        # initialized.
        b = t.__bases__
        if len(b) != 1 or b[0] != tuple:
            return False
        f = getattr(t, "_fields", None)
        if not isinstance(f, tuple):
            return False
        return all(type(n) == str for n in f)

    cls = gettype(state)

    content = tuple(_get_instance(value, src) for value in state.content)
    if isnamedtuple(cls):
        return cls(*content)
    return content


def function_get_state(obj: FunctionType, dst: str) -> StateFunction:
    state = StateFunction(
        cls=obj.__class__.__name__,
        module=get_module(obj),
        content=_ContentFunction(
            module_path=get_module(obj),
            function=obj.__name__,
        ),
    )
    return state


def function_get_instance(state: StateFunction, src: ZipFile) -> FunctionType:
    module = state.content.module_path
    function = state.content.function
    loaded = _import_obj(module, cls_or_func=function)
    return loaded


def partial_get_state(obj: partial, dst: str) -> StatePartial:
    reduce = obj.__reduce__()
    assert isinstance(reduce, tuple)
    _, _, (func, args, kwds, namespace) = reduce
    state = StatePartial(
        cls="partial",  # don't allow any subclass
        module=get_module(type(obj)),
        content=_ContentPartial(
            func=_get_state(func, dst),
            args=_get_state(args, dst),
            kwds=_get_state(kwds, dst),
            namespace=_get_state(namespace, dst),
        ),
    )
    return state


def partial_get_instance(obj, src):
    content = obj["content"]
    func = _get_instance(content["func"], src)
    args = _get_instance(content["args"], src)
    kwds = _get_instance(content["kwds"], src)
    namespace = _get_instance(content["namespace"], src)
    instance = partial(func, *args, **kwds)  # always use partial, not a subclass
    instance.__setstate__((func, args, kwds, namespace))
    return instance


def type_get_state(obj, dst):
    # To serialize a type, we first need to set the metadata to tell that it's
    # a type, then store the type's info itself in the content field.
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "content": {
            "__class__": obj.__name__,
            "__module__": get_module(obj),
        },
    }
    return res


def type_get_instance(obj, src):
    loaded = _import_obj(obj["content"]["__module__"], obj["content"]["__class__"])
    return loaded


def slice_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
        "content": {
            "start": obj.start,
            "stop": obj.stop,
            "step": obj.step,
        },
    }
    return res


def slice_get_instance(obj, src):
    start = obj["content"]["start"]
    stop = obj["content"]["stop"]
    step = obj["content"]["step"]
    return slice(start, stop, step)


def object_get_state(obj: Any, dst: str) -> StateObject:
    # This method is for objects which can either be persisted with json, or
    # the ones for which we can get/set attributes through
    # __getstate__/__setstate__ or reading/writing to __dict__.
    # try:
    #     # if we can simply use json, then we're done.
    #     return json.dumps(obj)
    # except Exception:
    #     pass

    cls = obj.__class__.__name__
    module = get_module(type(obj))

    # __getstate__ takes priority over __dict__, and if non exist, we only save
    # the type of the object, and loading would mean instantiating the object.
    if hasattr(obj, "__getstate__"):
        attrs = obj.__getstate__()
    elif hasattr(obj, "__dict__"):
        attrs = obj.__dict__
    else:
        return StateObject(cls=cls, module=module)

    content = _get_state(attrs, dst)
    # it's sufficient to store the "content" because we know that this dict can
    # only have str type keys
    state = StateObject(cls=cls, module=module, content=content)
    return state


def object_get_instance(state: StateObject, src: ZipFile) -> object:
    # try:
    #     return json.loads(state)
    # except Exception:
    #     pass

    cls = gettype(state)

    # Instead of simply constructing the instance, we use __new__, which
    # bypasses the __init__, and then we set the attributes. This solves
    # the issue of required init arguments.
    instance = cls.__new__(cls)  # type: ignore

    if state.content is None:
        attrs = {}
    else:
        attrs = _get_instance(state.content, src)
    if not attrs:
        return instance

    if hasattr(instance, "__setstate__"):
        instance.__setstate__(attrs)
    else:
        instance.__dict__.update(attrs)

    return instance


def unsupported_get_state(obj, dst):
    raise UnsupportedTypeException(obj)


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    (dict, dict_get_state),
    (list, list_get_state),
    (tuple, tuple_get_state),
    (slice, slice_get_state),
    (FunctionType, function_get_state),
    (partial, partial_get_state),
    (type, type_get_state),
    (object, object_get_state),
]
# tuples of type and function that creates the instance of that type
GET_INSTANCE_DISPATCH_FUNCTIONS = [
    (dict, dict_get_instance),
    (list, list_get_instance),
    (tuple, tuple_get_instance),
    (slice, slice_get_instance),
    (FunctionType, function_get_instance),
    (partial, partial_get_instance),
    (type, type_get_instance),
    (object, object_get_instance),
]
