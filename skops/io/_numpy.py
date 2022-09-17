from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

import numpy as np

from ._general import StateDict, StateTuple, function_get_instance
from ._types import State
from ._utils import _get_instance, _get_state, _import_obj, get_module


@dataclass
class StateNdarrayDisk(State):
    file: str
    content: None


@dataclass
class StateNdarrayJson(State):
    content: list[State]
    shape: StateTuple


# @dataclass
# class _ContentMaskedArray:
#     data: State
#     mask: State


# @dataclass
# class StateMaskedArray(State):
#     content: _ContentMaskedArray


@dataclass
class StateRandomState(State):
    content: StateDict


@dataclass
class StateRandomGenerator(State):
    content: StateDict


@dataclass
class _ContentUfunc:
    module_path: str
    function: str


@dataclass
class StateUfunc(State):
    content: _ContentUfunc


@dataclass
class StateDtype(State):
    content: StateNdarrayDisk | StateNdarrayJson


def ndarray_get_state(
    obj: np.ndarray | np.generic, dst: str
) -> StateNdarrayDisk | StateNdarrayJson:
    cls = obj.__class__.__name__
    module = get_module(type(obj))

    try:
        f_name = f"{uuid4()}.npy"
        with open(Path(dst) / f_name, "wb") as f:
            np.save(f, obj, allow_pickle=False)
        return StateNdarrayDisk(cls=cls, module=module, file=f_name, content=None)
    except ValueError:
        # Object arrays cannot be saved with allow_pickle=False, therefore we
        # convert them to a list and recursively call _get_state on it.
        if obj.dtype != object:
            raise TypeError(f"numpy arrays of dtype {obj.dtype} are not supported yet")

        obj_serialized = _get_state(obj.tolist(), dst)
        state = StateNdarrayJson(
            cls=cls,
            module=module,
            content=obj_serialized,
            shape=_get_state(obj.shape, dst),
        )

    return state


def ndarray_get_instance(
    state: StateNdarrayDisk | StateNdarrayJson, src: ZipFile
) -> np.ndarray | np.generic:
    if isinstance(state, StateNdarrayDisk):
        val = np.load(io.BytesIO(src.read(state.file)), allow_pickle=False)
        # Coerce type, because it may not be conserved by np.save/load. E.g. a
        # scalar will be loaded as a 0-dim array.
        if state.cls != "ndarray":
            cls = _import_obj(state.module, state.cls)
            val = cls(val)
    else:
        # We explicitly set the dtype to "O" since we only save object arrays
        # in json.
        shape = _get_instance(state.shape, src)
        tmp = [_get_instance(s, src) for s in state.content]
        # TODO: this is a hack to get the correct shape of the array. We should
        # find a better way to do this.
        if len(shape) == 1:
            val = np.ndarray(shape=len(tmp), dtype="O")
            for i, v in enumerate(tmp):
                val[i] = v
        else:
            val = np.array(tmp, dtype="O")
    return val


# def maskedarray_get_state(obj: np.ma.MaskedArray, dst: str) -> StateMaskedArray:
#     state = StateMaskedArray(
#         cls=obj.__class__.__name__,
#         module=get_module(type(obj)),
#         content=_ContentMaskedArray(
#             data=_get_state(obj.data, dst),
#             mask=_get_state(obj.mask, dst),
#         ),
#     )
#     return state


# def maskedarray_get_instance(state: StateMaskedArray, src: ZipFile) -> np.ma.MaskedArray:
#     data = _get_instance(state.content.data, src)
#     mask = _get_instance(state.content.mask, src)
#     return np.ma.MaskedArray(data, mask)


def random_state_get_state(obj: np.random.RandomState, dst: str) -> StateRandomState:
    content = _get_state(obj.get_state(legacy=False), dst)
    state = StateRandomState(
        cls=obj.__class__.__name__,
        module=get_module(type(obj)),
        content=content,
    )
    return state


def random_state_get_instance(
    state: StateRandomState, src: ZipFile
) -> np.random.RandomState:
    cls = _import_obj(state.module, state.cls)
    random_state = cls()
    content = _get_instance(state.content, src)
    random_state.set_state(content)
    return random_state


def random_generator_get_state(
    obj: np.random.Generator, dst: str
) -> StateRandomGenerator:
    bit_generator_state = _get_state(obj.bit_generator.state, dst)
    state = StateRandomGenerator(
        cls=obj.__class__.__name__,
        module=get_module(type(obj)),
        content=bit_generator_state,
    )
    return state


def random_generator_get_instance(
    state: StateRandomGenerator, src: ZipFile
) -> np.random.Generator:
    # first restore the state of the bit generator
    bit_generator_state = _get_instance(state.content, src)
    bit_generator = _import_obj("numpy.random", bit_generator_state["bit_generator"])()
    bit_generator.state = bit_generator_state

    # next create the generator instance
    cls = _import_obj(state.module, state.cls)
    random_generator = cls(bit_generator=_get_instance(bit_generator, src))
    return random_generator


# For numpy.ufunc we need to get the type from the type's module, but for other
# functions we get it from objet's module directly. Therefore set a especial
# get_state method for them here. The load is the same as other functions.
def ufunc_get_state(obj: np.ufunc, dst: str) -> StateUfunc:
    state = StateUfunc(
        # cast to str explicitly because mypy thinks it's a Callable[[ufunc], str]
        cls=str(obj.__class__.__name__),  # ufunc
        module=get_module(type(obj)),  # numpy
        content=_ContentUfunc(
            module_path=get_module(obj),
            function=obj.__name__,
        ),
    )
    return state


def dtype_get_state(obj: np.dtype, dst: str) -> StateDtype:
    # we use numpy's internal save mechanism to store the dtype by
    # saving/loading an empty array with that dtype.
    tmp: np.ndarray = np.ndarray(0, dtype=obj)
    state = StateDtype(
        cls="dtype",
        module="numpy",
        content=ndarray_get_state(tmp, dst),
    )
    return state


def dtype_get_instance(state: StateDtype, src: ZipFile) -> np.dtype:
    # we use numpy's internal save mechanism to store the dtype by
    # saving/loading an empty array with that dtype.
    tmp = ndarray_get_instance(state.content, src)
    return tmp.dtype


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    (np.generic, ndarray_get_state),
    (np.ndarray, ndarray_get_state),
    # (np.ma.MaskedArray, maskedarray_get_state),
    (np.ufunc, ufunc_get_state),
    (np.dtype, dtype_get_state),
    (np.random.RandomState, random_state_get_state),
    (np.random.Generator, random_generator_get_state),
]
# tuples of type and function that creates the instance of that type
GET_INSTANCE_DISPATCH_FUNCTIONS = [
    (np.generic, ndarray_get_instance),
    (np.ndarray, ndarray_get_instance),
    # (np.ma.MaskedArray, maskedarray_get_instance),
    (np.ufunc, function_get_instance),
    (np.dtype, dtype_get_instance),
    (np.random.RandomState, random_state_get_instance),
    (np.random.Generator, random_generator_get_instance),
]
