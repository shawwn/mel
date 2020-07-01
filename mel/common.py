import numpy as np


import tensorflow
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_util


try:
  import torch
except ModuleNotFoundError:
  torch = None


def is_lib_type(value, libname):
  if pylist(value):
    for x in value:
      if is_lib_type(x, libname):
        return True
    return False
  if pydict(value):
    for k, v in value.items():
      if is_lib_type(v, libname):
        return True
    return False
  name = type(value).__module__
  if name == libname:
    return True
  if '.' in name:
    return name.split('.')[0] == libname
  return False


def is_tf_type(value):
  return is_lib_type(value, 'tensorflow')
  # if tensorflow.is_tensor(value):
  #   return True
  # if isinstance(value, tensorflow.Variable):
  #   return True
  # if isinstance(value, tensorflow.TensorShape):
  #   return True
  # if isinstance(value, tensorflow.Dimension):
  #   return True
  # return False


def is_torch_type(value):
  return is_lib_type(value, 'torch')
  # if torch is None:
  #   return False
  # if torch.is_tensor(value):
  #   return True
  # return False


def is_np_type(value):
  return is_lib_type(value, 'numpy')
  #return type(value).__module__ == 'numpy'
  # for k, v in np.typeDict.items():
  #   if isinstance(value, v):
  #     return True
  # return False


def pynum(value):
  return isinstance(value, float) or isinstance(value, int)


def pybool(value):
  return value is True or value is False


def pystr(value):
  return isinstance(value, str) or isinstance(value, bytes)


def pynil(value):
  return value is None


def pyatom(value):
  return pynum(value) or pybool(value) or pystr(value) or pynil(value) or is_np_type(value)


def pylist(value):
  return isinstance(value, list) or isinstance(value, tuple)


def pydict(value):
  return isinstance(value, dict)


def pyobj(value):
  return pylist(value) or pydict(value)


def is_np(value, strict=False, deep=True):
  if isinstance(value, np.ndarray):
    return True
  if not strict and pyatom(value):
    return True
  if deep and pylist(value):
    return all([is_np(x, strict=strict, deep=deep) for x in value])
  if deep and pydict(value):
    return all([is_np(v, strict=strict, deep=deep) for k, v in value.items()])
  return False


def is_tf(value, strict=False, deep=True):
  if tensorflow.is_tensor(value):
    return True
  if isinstance(value, tensorflow.Variable):
    return True
  if isinstance(value, tensorflow.TensorShape):
    return True
  if isinstance(value, tensorflow.Dimension):
    return True
  if not strict and pyatom(value):
    return True
  if deep and pylist(value):
    return all([is_tf(x, strict=strict, deep=deep) for x in value])
  if deep and pydict(value):
    return all([is_tf(v, strict=strict, deep=deep) for k, v in value.items()])
  return False


def is_tf_variable(value):
  if isinstance(value, tensorflow.Variable):
    return True
  return False


def is_tf_tensor(value):
  if tensorflow.is_tensor(value):
    return True
  return False


def is_torch_tensor(value):
  if torch is None:
    return False
  if torch.is_tensor(value):
    return True
  return False


def is_tf_constant(value):
  try:
    return constant_op.is_constant(value)
  except AttributeError:
    # TODO: is there a cleaner way to accomplish this?
    return False


def thru(*args, **kws):
  if len(args) > 0:
    return args[0]


def const(value, lib=None, deep=True, axis=0):
  if isinstance(value, tensorflow.TensorShape):
    return val(value)
  if isinstance(value, tensorflow.Dimension):
    return val(value)
  if torch:
    if isinstance(value, torch.Size):
      return val(value)
  lib = as_lib(lib, hint=value)
  #value = val(value, deep=deep)
  if deep and pydict(value):
    value = {k: const(v, lib=lib, deep=deep, axis=axis) for k, v in value.items()}
    return value
  if pyatom(value):
    if lib == tensorflow:
      value = tensorflow.constant(value)
    elif torch and lib == torch:
      value = torch.tensor(value)
    else:
      assert lib == np
      value = np.array(value)
  if deep and pylist(value):
    value = [const(x, lib=lib, deep=deep) for x in value]
  if pylist(value) and len(value) > 0:
    value = stack(value, lib=lib, axis=axis)
  return value

from pprint import pprint as pp

def ppr(x):
  pp(x)
  return x

def stack(value, axis=-1, lib=None):
  lib = as_lib(lib, hint=value)
  if not hasattr(lib, 'stack'):
    raise NotImplementedError
  if lib == torch:
    value = torch.stack([const(x, lib=lib) for x in value], dim=axis)
  else:
    value = lib.stack(value, axis=axis)
  return value


def arange(a, b, lib=None):
  lib = as_lib(lib, hint=a)
  if lib == tensorflow:
    return tensorflow.range(a, b)
  if torch and lib == torch:
    return torch.arange(a, b)
  assert lib == np
  return np.arange(a, b)


def linspace(a, b, i, lib=None):
  lib = as_lib(lib, hint=a)
  return lib.linspace(a, b, i)


def meshgrid(*args, lib=None):
  lib = as_lib(lib, hint=args[0])
  return lib.meshgrid(*args)


def zeros_like(value, lib=None):
  lib = as_lib(lib, hint=value)
  return lib.zeros_like(value)


def ones_like(value, lib=None):
  lib = as_lib(lib, hint=value)
  return lib.ones_like(value)


# mel.swizzle(mel.stack([mel.cumsum(v,1-i) for i, v in enumerate(mel.swizzle(mel.fill([8,4], [1.0/4.0, 1.0/8.0])))]))[0]

def lingrid(shape, Ux, Uy, Vx, Vy, lib=None):
  lib = as_lib(lib, hint=Ax)
  shape = const(shape, lib=lib)
  dims = shapeof(shape)
  pp(dims)
  assert len(dims) == 1
  rank = dims[0] - 1
  Ux = const(Ux, lib=lib)
  Uy = const(Uy, lib=lib)
  Vx = const(Vx, lib=lib)
  Vy = const(Vy, lib=lib)
  H, W = swizzle(shape)
  Udx = (Vx - Ux) / W
  Udy = (Vy - Uy) / H
  return fill(shape, [ABx, 0.0])

  
def is_pil_image(value):
  return isinstance(value, PIL.Image.Image)


def val(value, eager=False, session=None, deep=True):
  assert eager in [True, False]
  if isinstance(value, tensorflow.TensorShape):
    value = value.as_list()
  if isinstance(value, tensorflow.Dimension):
    value = value.value
  if torch:
    if isinstance(value, torch.Size):
      value = list(value)
  #if is_tf(value, strict=True) and not pyobj(value) and constant_op.is_constant(value):
  if is_tf_constant(value):
    value = tensor_util.constant_value(value)
  if deep and pylist(value):
    value = [val(x, eager=eager, session=session, deep=deep) for x in value]
  if deep and pydict(value):
    value = {k: val(v, eager=eager, session=session, deep=deep) for k, v in value.items()}
  if is_tf_variable(value) or is_tf_tensor(value):
    if not eager:
      raise ValueError("val called on tensorflow variable {} but eager is False".format(value))
    # TODO: batch multiple nested reads.
    result = value.eval(session=session)
    return val(result, eager=eager, session=session, deep=deep)
  if is_torch_tensor(value):
    if not eager:
      raise ValueError("val called on torch tensor {} but eager is False".format(value))
    result = value.numpy()
    return val(result, eager=eager, session=session, deep=deep)
  if is_pil_image(value):
    result = np.array(value)
    if result.dtype == np.uint8:
      result = result.astype(np.float32) / 255.0
    return result
  if not is_np(value):
    raise ValueError("val called with value of unknown type: {}".format(value))
  return value


import importlib


def as_lib(lib=None, hint=None):
  if isinstance(lib, str):
    if lib == 'tf':
      lib = 'tensorflow'
    if lib == 'np':
      lib = 'numpy'
    if lib == 'tensorflow':
      lib = tensorflow
    elif lib == 'numpy':
      lib = np
    elif lib == 'torch':
      lib = torch
    else:
      lib = importlib.__import__(lib)
    return lib
  if lib is None:
    if is_tf_type(hint):
      return tensorflow
    if is_torch_type(hint):
      return torch
    return np
  assert lib is tensorflow or lib is np or lib is torch
  return lib


dtype_remap = {
    'bool': 'y',

    'bfloat': 'b16',
    'bfloat16': 'b16',

    'float16': 'f16',

    'f': 'f32',
    'float': 'f32',
    'float32': 'f32',

    'd': 'f64',
    'double': 'f64',
    'float64': 'f64',

    'c': 'f128',
    'complex': 'f128',
    'complex64': 'f128',

    'byte': 'u8',
    'uint8': 'u8',
    'ushort': 'u16',
    'uint16': 'u16',
    'uint': 'u32',
    'ulong': 'u64',
    'ulong64': 'u64',

    'u': 'u32',
    'unsigned': 'u32',
    'uint32': 'u32',

    'int8': 'i8',

    'short': 'i16',
    'int16': 'i16',

    'i': 'i32',
    'signed': 'i32',
    'int': 'i32',
    'int32': 'i32',

    'l': 'i64',
    'long': 'i64',
    'long64': 'i64',
    'resource': 'r',
    'variant': 'v',
}


for v in list(dtype_remap.values()):
  dtype_remap[v] = v


dtype_long = {
    'y': 'bool',
    'b16': 'bfloat16',
    'f16': 'float16',
    'f32': 'float32',
    'f64': 'float64',
    'f128': 'complex64',
    'u8': 'uint8',
    'u16': 'uint16',
    'u32': 'uint32',
    'u64': 'uint64',
    'i8': 'int8',
    'i16': 'int16',
    'i32': 'int32',
    'i64': 'int64',
    'r': 'resource',
    'v': 'variant',
}


for v in list(dtype_remap.values()):
  assert v in dtype_long


dtype_specs = {
    np.int64: 'i64',
    np.int32: 'i32',
    np.int16: 'i16',
    np.int8: 'i8',
    np.uint64: 'u64',
    np.uint32: 'u32',
    np.uint16: 'u16',
    np.uint8: 'u8',
    np.float64: 'f64',
    np.float32: 'f32',
    np.complex: 'f128',
    np.bool_: 'y',

    tensorflow.int64: 'i64',
    tensorflow.int32: 'i32',
    tensorflow.int16: 'i16',
    tensorflow.int8: 'i8',
    tensorflow.uint64: 'u64',
    tensorflow.uint32: 'u32',
    tensorflow.uint16: 'u16',
    tensorflow.uint8: 'u8',
    tensorflow.float64: 'f64',
    tensorflow.float32: 'f32',
    tensorflow.float16: 'f16',
    tensorflow.bfloat16: 'b16',
    tensorflow.complex64: 'f128',
    tensorflow.bool: 'y',
}

if torch:
  dtype_specs.update({
    torch.int64: 'i64',
    torch.int32: 'i32',
    torch.int16: 'i16',
    torch.int8: 'i8',
    #torch.uint64: 'u64',
    #torch.uint32: 'u32',
    #torch.uint16: 'u16',
    torch.uint8: 'u8',
    torch.float64: 'f64',
    torch.float32: 'f32',
    torch.complex64: 'f128',
    torch.bool: 'y',
  })


dtype_specs.update({x: x for x in list(set(dtype_remap.values()))})

spec_to_dtype = {
    'np': {v: k for k, v in dtype_specs.items() if is_np_type(k)},
    'tf': {v: k for k, v in dtype_specs.items() if is_tf_type(k)},
    'torch': {v: k for k, v in dtype_specs.items() if is_torch_type(k)},
}

def as_dtype(value, lib=None, deep=True):
  lib = as_lib(lib, hint=value)
  if isinstance(value, str):
    if value not in dtype_remap:
      raise ValueError("Unknown dtype spec: {}".format(value))
    value = dtype_remap[value]
    assert value in dtype_long
    value = dtype_long[value]
    if not hasattr(lib, value):
      raise ValueError("Library {} does not support dtype {}".format(lib, value))
    value = getattr(lib, value)
  if deep and pylist(value):
    return [as_dtype(x, lib=lib, deep=deep) for x in value]
  if deep and pydict(value):
    return {k: as_dtype(v, lib=lib, deep=deep) for k, v in value.items()}
  if pyatom(value):
    value = const(value)
  if lib == np:
    value = np.result_type(value).type
  else:
    if hasattr(value, 'dtype'):
      value = value.dtype
  return value


def dtype_spec(value, deep=True):
  if pylist(value):
    if deep:
      value = [dtype_spec(x, deep=deep) for x in value]
    return value
  if pydict(value):
    if deep:
      value = {k: dtype_spec(v, deep=deep) for k, v in value.items()}
    return value
  # if pystr(value):
  #   return value
  if isinstance(value, np.dtype):
    value = value.type
  value = dtype_specs[value]
  return value


_swiz = {
    'x': 0,
    'y': 1,
    'z': 2,
    'w': 3,

    'u': 0,
    'v': 1,
    's': 2,
    't': 3,

    'r': 0,
    'g': 1,
    'b': 2,
    'a': 3,
}


def index(c):
  if isinstance(c, str):
    c = _swiz[c]
  return c


def spec(value):
  return dtype_spec(as_dtype(value))


def component(value, c):
  if c == '0':
    return zeros_like(value[..., 0])
  if c in list('0123456789ABCDEF'):
    #percent = (ord(c) - ord('0')) / 10.0
    percent = int(c, 16) / 0xF
    if spec(value) in ['u8', 'i64']:
      percent *= 0xFF
      percent = int(percent)
    return ones_like(value[..., 0]) * percent
  i = index(c)
  return value[..., i]
  

def channels(value):
  value = const(value)
  c = shapeof(value)[-1]
  return c

# swizzle([1,2,3,4], '.xyzw') => [1,2,3,4]
# swizzle([1,2,3,4], '.zyzw') => [3,2,3,4]
# swizzle([1,2,3,4], '.yyy') => [2,2,2]
def swizzle(value, by=None):
  value = const(value)
  if by is None:
    c = shapeof(value)[-1]
    assert pynum(c)
    return [component(value, i) for i in range(c)]
  if isinstance(by, str):
    return [component(value, c) for c in by]
  elif pylist(by):
    return [swizzle(value, by=spec) for spec in by]
  else:
    raise NotImplementedError
    # value = {k: swizzle(v, deep=deep) for k, v in value.items()}
    # return value


def swizzled(value, by, lib=None):
  value = swizzle(value, by=by)
  value = stack(value, axis=-1, lib=lib)
  return value


def cumsum(value, by=None, lib=None):
  if by is None:
    by = 'xy'
  if isinstance(by, str):
    by = [index(c) for c in by]
  if not pylist(by):
    by = [by]
  lib = as_lib(lib, hint=value)
  value = const(value, lib=lib)
  for i in by:
    value = lib.cumsum(value, i)
  return value



def _pyflat(value):
  if pylist(value):
    for x in value:
      yield from _pyflat(x)
  elif pydict(value):
    for k, v in value.items():
      yield from _pyflat(v)
  else:
    yield value

def pyflat(value):
  return list(_pyflat(value))

import functools
import operator

import PIL.Image

def cast(value, dtype=None, lib=None, hint=None):
  if dtype == 'img':
    if spec(value)[0] == 'f':
      value *= 255.0
    value = cast(value, dtype='u8', lib=np)
    return PIL.Image.fromarray(value)
  if pylist(value):
    value = stack(value, axis=-1, lib=lib)
  lib = as_lib(lib, hint=value)
  if dtype is None:
    if hint is None:
      hint = value
    dtypes = pyflat(as_dtype(hint, lib=lib, deep=True))
    if not functools.reduce(operator.eq, dtypes):
      raise ValueError("dtypes not equal for {}".format(value))
    dtype = dtypes[0]
  dtypes = pyflat(as_dtype(dtype, lib=lib, deep=True))
  if not functools.reduce(operator.eq, dtypes):
    raise ValueError("dtypes not equal for {}".format(value))
  dtype = dtypes[0]
  if lib == np:
    return np.array(value).astype(dtype)
  elif lib == tensorflow:
    return tensorflow.cast(value, dtype=dtype)
  else:
    assert lib == torch
    return torch.cast(value, dtype=dtype)


def shapeof(value):
  if hasattr(value, 'shape'):
    value = value.shape
  if isinstance(value, tensorflow.TensorShape):
    return val(value)
  if isinstance(value, tensorflow.Dimension):
    return val(value)
  if torch:
    if isinstance(value, torch.Size):
      return val(value)
  if pylist(value):
    value = [shapeof(x) for x in value]
  if pydict(value):
    value = {k: shapeof(v) for k, v in value.items()}
  return value


def fill(shape, value, lib=None):
  if pylist(value):
    return stack([fill(shape, c) for c in value], axis=-1, lib=lib)
  value = const(value, lib=lib)
  if shapeof(value) != []:
    return stack([fill(shape, c) for c in swizzle(value)])
  if lib is None or pystr(lib):
    lib1 = lib
    lib = as_lib(lib1, hint=value)
    if lib == np:
      lib = as_lib(lib1, hint=shape)
  #assert pylist(shape)
  if lib == tensorflow:
    shape = const(shape, lib=lib)
    return tensorflow.fill(shape, value)
  elif lib == np:
    r = np.zeros(shape, dtype=as_dtype(value, lib=np))
    r.fill(value)
    return r
  else:
    assert torch and lib == torch
    spec = dtype_spec(as_dtype(value))
    dtype = spec_to_dtype['torch'][spec]
    r = torch.zeros(shape, dtype=dtype)
    r.fill_(value)
    return r


def shaped(value, lib=None):
  return shapeof(const(value, lib=lib))


def rank(value, lib=None):
  return len(shaped(value, lib=lib))


def grid(shape, value=None, lib=None):
  #lib = as_lib(lib, hint=shape)
  shape = const(shape, lib=lib)
  if value is None:
    value = [0.0, 0.0, 0.0, 0.0]
  value = fill(shape, [0.0, 0.0, 0.0, 0.0], lib=lib)
  return value



def vec4(value, lib=None):
  lib = as_lib(lib, hint=value)
  last = tensorflow.unstack(value, axis=-1)
  if len(last) == 4:
    return value
  rest = []
  for i in range(len(last)):
    #rest.append(jk/
    pass
  
  #return val(last.shape)
  return last



