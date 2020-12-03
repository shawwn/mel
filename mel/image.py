import PIL.Image
import unicodedata
import os
import numpy as np

from . import common as mop

tf = mop.tensorflow

import functools


def op_scope(fn, name=None):
    if name is None:
        name = fn.__name__
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with tf.name_scope(fn.__name__):
            return fn(*args, **kwargs)
    return _fn


@op_scope
def sequential_for(fn, begin, end, *args):
    def _cond(i, *xs):
        return tf.less(i, end)
    def _body(i, *xs):
        ops, *ys = fn(i, *xs)
        with tf.control_dependencies(ops):
            return [i + 1] + list(ys)
    return tf.while_loop(_cond, _body, [begin] + list(args))


def tf_either(x, *args):
  if len(args) <= 0:
    return x
  if len(args) == 1:
    return tf.logical_or(x, *args)
  return tf.logical_or(x, tf_either(*args))


def tf_in(x, *args):
  return tf_either(*[tf.equal(x, c) for c in args])


@op_scope
def tri_cross(v1, v2):
    return [v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]]



@op_scope
def tri_dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


@op_scope
def tri_expand_dims(v, axis):
    return (tf.expand_dims(v[0], axis),
            tf.expand_dims(v[1], axis),
            tf.expand_dims(v[2], axis))


@op_scope
def tri_gather(v, inds):
    return (tf.gather(v[0], inds),
            tf.gather(v[1], inds),
            tf.gather(v[2], inds))

    


@op_scope
def barycentric(verts, p, eps):
    ab = verts[2] - verts[0]
    ac = verts[1] - verts[0]
    pa = verts[0] - p
    u = tri_cross(
        [ab[0], ac[0], pa[..., 0]],
        [ab[1], ac[1], pa[..., 1]])
    v = [u[0] / u[2], u[1] / u[2]]
    bc = [1. - v[0] - v[1], v[1], v[0]]
    valid = tf.logical_and(
        tf.abs(u[2]) >= 1.0 - eps,
        tf.reduce_all(tf.stack(bc, axis=1) >= 0 - eps, axis=1))
    return bc, valid


@op_scope
def clamp(v, min=0., max=1.):
    #return tf.minimum(tf.maximum(v, min), max)
    return tf.clip_by_value(v, min, max)


@op_scope
def wrap(v, mode):
  assert mode in ["clamp", "wrap", "reflect"]
  if mode == "wrap":
    return tf.floormod(v, 1.0)
  elif mode == "reflect":
    return tf.abs(tf.floormod(v, 2.0) - 1.0)
  elif mode == "clamp":
    return clamp(v)


@op_scope
def iround(u):
  return tf.to_int32(tf.floordiv(tf.to_float(u), 1.0))


@op_scope
def sample(tex, uv, mode="bilinear", wrap_mode="wrap", unpack=False):
  assert mode in ["nearest", "bilinear"]
  #wh = tf.shape(tex if unpack else tex[:, :, 0])
  wh = tf.shape(tex)
  grab = lambda u, v: tf.gather_nd(tex, tf.stack([
    clamp(iround(v), 0, wh[1] - 1),
    #clamp(wh[1] - iround(v) - 1, 0, wh[1] - 1),
    clamp(iround(u), 0, wh[0] - 1),
    ], 1))
  get = lambda u, v: (unpack_colors(grab(u, v), 1) if unpack else grab(u, v))
  if mode == "nearest":
    uv = wrap(uv, wrap_mode) * tf.to_float(wh)
    u = uv[..., 0]
    v = uv[..., 1]
    return get(u, v)
  elif mode == "bilinear":
    # https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c#L105
    uv = wrap(uv, wrap_mode) * tf.to_float(wh)
    ix = uv[..., 0] - 0.5
    iy = uv[..., 1] - 0.5
    # get NE, NW, SE, SW pixel values from (x, y)
    ix_nw = iround(ix)
    iy_nw = iround(iy)
    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1
    sub = lambda a, b: tf.to_float(a) - tf.to_float(b)
    # get surfaces to each neighbor:
    nw = sub(ix_se , ix)    * sub(iy_se , iy);
    ne = sub(ix    , ix_sw) * sub(iy_sw , iy);
    sw = sub(ix_ne , ix)    * sub(iy    , iy_ne);
    se = sub(ix    , ix_nw) * sub(iy    , iy_nw);
    nw_val = tf.to_float(get(ix_nw, iy_nw))
    ne_val = tf.to_float(get(ix_ne, iy_ne))
    sw_val = tf.to_float(get(ix_sw, iy_sw))
    se_val = tf.to_float(get(ix_se, iy_se))
    #a = lambda x: x[..., tf.newaxis]
    a = lambda x: x
    out = nw_val * a(nw)
    out += ne_val * a(ne)
    out += sw_val * a(sw)
    out += se_val * a(se)
    return out


def tf_draw(src, img, x0, y0, x1, y1, u0, v0, u1, v1):
  img_shape = tf.shape(img)
  src_shape = tf.shape(src)
  img_h = img_shape[0]
  img_w = img_shape[1]
  src_h = src_shape[0]
  src_w = src_shape[1]
  def _fn(i, color, px, py):
    def _draw():
      with tf.name_scope('_draw'):
        # x0 = offset + px
        # x1 = offset + px + w
        # y0 = py
        # y1 = py + 16
        # u0 = 0
        # u1 = w / 16
        # v0 = 0
        # v1 = 16 / 16
        minx = tf.clip_by_value(x0 + 0, 0, img_w)
        maxx = tf.clip_by_value(x1 + 1, 0, img_w)
        miny = tf.clip_by_value(y0 + 0, 0, img_h)
        maxy = tf.clip_by_value(y1 + 1, 0, img_h)
        verts = tf.cast([
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
            ], dtype=tf.float32)
        uvs = tf.cast([
            [u0, v0],
            [u1, v0],
            [u1, v1],
            [u0, v1],
            ], dtype=tf.float32)
        triangles = tf.constant([[0,1,2], [0,2,3]], tf.int32)
        c = tf.cast(color, tf.float32)
        for tri in range(2):
          indices = tf.gather(triangles, tri)
          verts_i = tf.gather(verts, indices)
          uvs_i = tf.gather(uvs, indices)
          y_, x_ = tf.meshgrid(tf.range(minx, maxx), tf.range(miny, maxy))
          p = tf.stack([y_, x_], -1); p = tf.reshape(p, [-1, 2]); p = tf.cast(p, tf.float32)
          bc, valid = barycentric(verts_i, p, eps=0.0*0.5 / tf.cast(img_w, tf.float32))
          p = tf.boolean_mask(p, valid)
          bc = [tf.boolean_mask(bc[k], valid) for k in range(3)]
          u = tri_dot(uvs_i[..., 0], bc)
          v = tri_dot(uvs_i[..., 1], bc)
          uv = tf.stack([u, v], -1)
          value = tf.cast(sample(src, uv), tf.float32)
          inds = tf.cast(tf.stack([p[..., 1], p[..., 0]], axis=-1), tf.int32)
          c = clamp(tf.tensor_scatter_add(c, inds, value), 0, 255)
        c = tf.cast(c, tf.uint8)
        return c, px, py
    color, px, py = _draw()
    updates = []
    return updates, color, px, py
  _, color, px, py = sequential_for(_fn, 0, 1, img, x0, y0)
  return color, px, py


