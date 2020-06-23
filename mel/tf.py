import tensorflow as tf
import numpy as np
import functools


def op_scope(fn, name=None):
    if name is None:
        name = fn.__name__
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with tf.name_scope(fn.__name__):
            return fn(*args, **kwargs)
    return _fn


def pynum(u):
  return isinstance(u, float) or isinstance(u, int)


@op_scope
def clamp(v, min=0., max=1.):
  if pynum(v):
    return np.clip(v, min, max)
  else:
    return tf.clip_by_value(v, min, max)


@op_scope
def wrap(v, wrap_mode="reflect"):
  assert wrap_mode in ["clamp", "wrap", "reflect"]
  if wrap_mode == "wrap":
    return tf.math.floormod(v, 1.0)
  elif wrap_mode == "reflect":
    return tf.abs(tf.math.floormod(v, 2.0) - 1.0)
  elif wrap_mode == "clamp":
    return clamp(v)


@op_scope
def iround(u):
  if pynum(u):
    return u // 1.0
  else:
    return i32(tf.math.floordiv(f32(u), 1.0))


@op_scope
def lsh(u, by):
  if pynum(u) and pynum(by):
    return int(u) << by
  else:
    return tf.bitwise.left_shift(u, by)


@op_scope
def rsh(u, by):
  if pynum(u) and pynum(by):
    return int(u) >> by
  else:
    return tf.bitwise.right_shift(u, by)

@op_scope
def sign(u):
  if pynum(u):
    return np.sign(u)
  else:
    return tf.sign(u)

@op_scope
def min2(a, b):
  if pynum(a) and pynum(b):
    return min(a, b)
  else:
    return tf.minimum(a, b)


@op_scope
def max2(a, b):
  if pynum(a) and pynum(b):
    return max(a, b)
  else:
    return tf.maximum(a, b)


@op_scope
def min3(a, b, c):
  if pynum(a) and pynum(b) and pynum(c):
    return min(a, b, c)
  else:
    return tf.minimum(a, tf.minimum(b, c))


@op_scope
def max3(a, b, c):
  if pynum(a) and pynum(b) and pynum(c):
    return max(a, b, c)
  else:
    return tf.maximum(a, tf.maximum(b, c))


@op_scope
def f32(u):
  if pynum(u):
    return float(u)
  else:
    return tf.cast(u, tf.float32)


@op_scope
def i32(u):
  if pynum(u):
    return int(u)
  else:
    return tf.cast(u, tf.int32)


@op_scope
def sample(tex, uv, method="bilinear", wrap_mode="reflect"):
  assert method in ["nearest", "bilinear"]
  wh = tf.shape(tex[:, :, 0])
  #gather_op = tf.gather_nd
  gather_op = tf.raw_ops.GatherNd # use tf.raw_ops.GatherNd directly, to support resource variables inside a TPU computation
  get = lambda u, v: gather_op(params=tex, indices=tf.stack([
    clamp(iround(u), 0, wh[0] - 1),
    clamp(iround(v), 0, wh[1] - 1),
    ], 1))
  uv = wrap(uv, wrap_mode)
  u = uv[:, 0]
  v = uv[:, 1]
  u *= f32(wh)[0]
  v *= f32(wh)[1]
  if method == "nearest":
    return get(u, v)
  elif method == "bilinear":
    # https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c#L105
    ix = u - 0.5
    iy = v - 0.5

    # get NE, NW, SE, SW pixel values from (x, y)
    ix_nw = iround(ix)
    iy_nw = iround(iy)
    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    sub = lambda a, b: f32(a) - f32(b)

    # get surfaces to each neighbor:
    nw = sub(ix_se , ix)    * sub(iy_se , iy);
    ne = sub(ix    , ix_sw) * sub(iy_sw , iy);
    sw = sub(ix_ne , ix)    * sub(iy    , iy_ne);
    se = sub(ix    , ix_nw) * sub(iy    , iy_nw);

    nw_val = f32(get(ix_nw, iy_nw))
    ne_val = f32(get(ix_ne, iy_ne))
    sw_val = f32(get(ix_sw, iy_sw))
    se_val = f32(get(ix_se, iy_se))

    a = lambda x: x[:, tf.newaxis]
    out = nw_val * a(nw)
    out += ne_val * a(ne)
    out += sw_val * a(sw)
    out += se_val * a(se)
    return out


@op_scope
def resize(img, size, method="bilinear", wrap_mode="reflect"): #, preserve_aspect_ratio=False):
  assert method in ["nearest", "bilinear", "area"]
  assert wrap_mode in ["clamp", "wrap", "reflect"]
  y, x = tf.meshgrid(
      tf.range(0.0, size[0] + 0.0),
      tf.range(0.0, size[1] + 0.0))
  num_frags = tf.reduce_prod(tf.shape(x))
  uv = tf.stack([
    #1.0 - tf.reshape(y, [-1]) / f32(size[1]),
    #tf.reshape(x, [-1]) / f32(size[0]),

    #1.0 - tf.reshape(y, [-1]) / f32(size[1]),
    #tf.reshape(x, [-1]) / f32(size[0]),

    tf.reshape(y, [-1]) / f32(size[0]),
    tf.reshape(x, [-1]) / f32(size[1]),
    #tf.zeros([num_frags], dtype=tf.float32)
    ], axis=1)

  re = lambda out: tf.transpose(tf.reshape(out, [size[1], size[0], -1]), [1,0,2])
  if method == "nearest" or method == "bilinear":
    return re(sample(img, uv, method=method, wrap_mode=wrap_mode))

  assert method == "area"
  assert "Resize in area mode not yet working"
  uv_00 = uv
  uv_10 = tf.stack([
    uv_00[:, 0] + 1.0 / size[0],
    uv_00[:, 1] + 0.0 / size[1],
    ], axis=1)
  uv_01 = tf.stack([
    uv_00[:, 0] + 0.0 / size[0],
    uv_00[:, 1] + 1.0 / size[1],
    ], axis=1)
  uv_11 = tf.stack([
    uv_00[:, 0] + 1.0 / size[0],
    uv_00[:, 1] + 1.0 / size[1],
    ], axis=1)
  wh = f32(tf.shape(img[:, :, 0]))

  UV_00 = tf.reduce_min([uv_00, uv_10, uv_01, uv_11], axis=0)
  UV_11 = tf.reduce_max([uv_00, uv_10, uv_01, uv_11], axis=0)
  UV_01 = tf.stack([UV_00[:, 0], UV_11[:, 1]], 1)
  UV_10 = tf.stack([UV_11[:, 0], UV_00[:, 1]], 1)

  img_sum = tf.cumsum(tf.cumsum(f32(img), 0), 1)

  tex_00 = f32(sample(img_sum, UV_00, "bilinear"))
  tex_10 = f32(sample(img_sum, UV_10, "bilinear"))
  tex_01 = f32(sample(img_sum, UV_01, "bilinear"))
  tex_11 = f32(sample(img_sum, UV_11, "bilinear"))

  R = wh[0]*(uv_11[:, 0] - uv_00[:, 0]) * wh[1]*(uv_11[:, 1] - uv_00[:, 1])

  out = re(tex_11 - tex_10 - tex_01 + tex_00) / R[0]
  out = clamp(out, 0.0, 255.0)
  return out


if __name__ == "__main__":
  import sys
  args = sys.argv[1:]
  sess = tf.InteractiveSession()
  with open(args[0], 'rb') as f:
    img = sess.run(tf.io.decode_image(f.read(), channels=3))
  outfile = args[1]
  w = 128 if len(args) <= 2 else int(args[2])
  h = 128 if len(args) <= 3 else int(args[3])
  method = "area" if len(args) <= 4 else args[4]
  wrap_mode = "reflect" if len(args) <= 5 else args[5]
  img2 = sess.run(resize(img, [w, h], method=method, wrap_mode=wrap_mode))
  with open(args[1], 'wb') as f:
    f.write(sess.run(tf.image.encode_png(img2)))
