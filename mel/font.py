import PIL.Image
import unicodedata
import os
import numpy as np

from . import common as mop


def read_glyph_sizes(path):
  global glyph_width
  global glyph_offset
  global char_width
  glyph_width = np.array([x for x in open(os.path.join(path, 'font', 'glyph_sizes.bin'), 'rb').read()])
  glyph_offset = np.zeros_like(glyph_width)
  glyph_offset[ord('l')] = -1
  glyph_offset[ord('i')] = -1
  glyph_offset[ord('t')] = -1
  glyph_offset[ord('`')] = -1
  glyph_offset[ord('.')] = -2
  glyph_offset[ord('!')] = -2
  glyph_offset[ord('[')] = -1
  glyph_offset[ord('(')] = -1
  glyph_offset[ord(')')] = -1
  glyph_width[ord('!')] += 2
  glyph_width[ord('[')] += 2
  glyph_width[ord('(')] += 2
  glyph_width[ord('{')] += 2
  glyph_width[ord("'")] += 2
  glyph_width[ord(".")] += 1
  glyph_width[ord("<")] += 1
  glyph_width[ord(">")] -= 1
  glyph_offset[ord("<")] = -2
  glyph_offset[ord(">")] = -2
  glyph_width[ord("\\")] += 1
  glyph_width[ord("/")] -= 1
  glyph_offset[ord("/")] = -1
  glyph_offset[ord("'")] = -2
  glyph_offset[ord(",")] = -2
  glyph_width[ord('"')] += 1
  glyph_offset[ord('"')] = -1
  glyph_offset[ord('f')] = -1
  #glyph_offset[ord('m')] = -1
  #glyph_width[ord(":")] += 2
  glyph_offset[ord(":")] = -2
  glyph_offset[ord(';')] = -2
  char_width = [4 for _ in range(256)]
  with PIL.Image.open(os.path.join(path, 'textures', 'font', 'ascii.png')) as im:
    img = im.convert('RGBA')
  lvt_3_2_ = img.width
  lvt_4_1_ = img.height
  lvt_5_1_ = np.array(img).reshape([-1, 4])[:, 3]
  #bufferedimage.getRGB(0, 0, lvt_3_2_, lvt_4_1_, lvt_5_1_, 0, lvt_3_2_);
  lvt_6_1_ = lvt_4_1_ // 16;
  lvt_7_1_ = lvt_3_2_ // 16;
  lvt_9_1_ = 8.0 / float(lvt_7_1_);
  for lvt_10_1_ in range(256):
    j1 = lvt_10_1_ % 16
    k1 = lvt_10_1_ // 16
    if lvt_10_1_ == 32:
      char_width[lvt_10_1_] = 4
    l1 = lvt_7_1_ - 1
    while l1 >= 0:
      i2 = j1 * lvt_7_1_ + l1;
      flag1 = True;
      for j2 in range(lvt_6_1_):
        if not flag1:
          break
        k2 = (k1 * lvt_7_1_ + j2) * lvt_3_2_;
        #if (lvt_5_1_[i2 + k2] >> 24 & 255) != 0:
        if lvt_5_1_[i2 + k2] != 0:
          flag1 = False
      if not flag1:
        break
      l1 -= 1
    l1 += 1
    char_width[lvt_10_1_] = int(0.5 + l1 * lvt_9_1_) + 1
  global unicode_chars
  unicode_chars = np.zeros([65536, 16, 16], dtype=np.uint8)
  shape = None
  for page in range(256):
    imgpath = os.path.join(path, 'textures', 'font', 'unicode_page_%02x.png' % page)
    if os.path.isfile(imgpath):
      with PIL.Image.open(imgpath) as im:
        if shape is None:
          shape = (im.width, im.height)
        else:
          assert im.width == shape[0] and im.height == shape[1]
        sheet = np.array(im.convert('RGBA'))[:, :, 3]
    else:
      sheet = np.zeros(shape, dtype=np.uint8)
    chars = sheet.reshape(sheet.shape[0]//16, 16, -1, 16).swapaxes(1,2).reshape(-1, 16, 16)
    for i in range(256):
      unicode_chars[page*256 + i] = chars[i]
  global unicode_widths
  unicode_widths = np.array([get_char_width(chr(i)) for i in range(65536)], dtype=np.int8)

class TensorflowFont:
  def __init__(self, session=None):
    sesison = session or tf.get_default_session()
    self.session = session
    #self.glyph_width = tf.constant(glyph_width, dtype=tf.int32)
    #self.char_width = tf.constant(char_width, dtype=tf.int32)
    self.widths = tf.constant(unicode_widths, dtype=tf.int32)
    self.chars = tf.constant(unicode_chars, dtype=tf.int32)
    self.offsets = tf.constant(glyph_offset, dtype=tf.int32)

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

# https://github.com/BananaPuncher714/BrickBoard/blob/afa613aafb13564c97b0ce1a9e7880e5b95d1234/io/github/bananapuncher714/brickboard/objects/MinecraftFontContainer.java#L77

def get_char_width(c, space_width=2):
  character = ord(c)
  if character == 160:
    return space_width # forge: display nbsp as space. MC-2595
  if character == 167:
    return -1
  if c == ' ':
    return space_width
  if c == '\t':
    return 2*space_width
  if c == '\n':
    return 0
  try:
    i = "\u00c0\u00c1\u00c2\u00c8\u00ca\u00cb\u00cd\u00d3\u00d4\u00d5\u00da\u00df\u00e3\u00f5\u011f\u0130\u0131\u0152\u0153\u015e\u015f\u0174\u0175\u017e\u0207\u0000\u0000\u0000\u0000\u0000\u0000\u0000 !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\u0000\u00c7\u00fc\u00e9\u00e2\u00e4\u00e0\u00e5\u00e7\u00ea\u00eb\u00e8\u00ef\u00ee\u00ec\u00c4\u00c5\u00c9\u00e6\u00c6\u00f4\u00f6\u00f2\u00fb\u00f9\u00ff\u00d6\u00dc\u00f8\u00a3\u00d8\u00d7\u0192\u00e1\u00ed\u00f3\u00fa\u00f1\u00d1\u00aa\u00ba\u00bf\u00ae\u00ac\u00bd\u00bc\u00a1\u00ab\u00bb\u2591\u2592\u2593\u2502\u2524\u2561\u2562\u2556\u2555\u2563\u2551\u2557\u255d\u255c\u255b\u2510\u2514\u2534\u252c\u251c\u2500\u253c\u255e\u255f\u255a\u2554\u2569\u2566\u2560\u2550\u256c\u2567\u2568\u2564\u2565\u2559\u2558\u2552\u2553\u256b\u256a\u2518\u250c\u2588\u2584\u258c\u2590\u2580\u03b1\u03b2\u0393\u03c0\u03a3\u03c3\u03bc\u03c4\u03a6\u0398\u03a9\u03b4\u221e\u2205\u2208\u2229\u2261\u00b1\u2265\u2264\u2320\u2321\u00f7\u2248\u00b0\u2219\u00b7\u221a\u207f\u00b2\u25a0\u0000".index(c)
  except ValueError:
    i = -1
  if character > 0 and i != -1 and False:
    return char_width[i]
  if glyph_width[character] != 0:
    j = glyph_width[character] & 255;
    k = j >> 4;
    l = j & 15;
    l += 1;
    return (l - k) // 2 + 1;
  return 4


def iterate_unicode_chars(s):
  for c in unicodedata.normalize('NFC',s):
    idx = ord(c)
    page, iy, ix = (idx//256, (idx%256)//16, (idx%256)%16)
    char_width = get_char_width(c)
    yield c, char_width, page, ix, iy

def draw(bg, img, x, y):
  if mop.is_pil_image(bg):
    if not mop.is_pil_image(img):
      img = mop.val(img, eager=True)
      img = mop.cast(img, 'img')
    bg.paste(img, (x, y), mask=img.convert('RGBA'))
    return bg
  else:
    raise NotImplementedError

def fill_text(bg, text, x, y, line_height=17):
  sx = x
  sy = y
  for c, char_width, page, ix, iy in iterate_unicode_chars(text):
    #path = '/Users/bb/Library/Application Support/minecraft/versions/1.13.2/1.13.2/assets/minecraft/textures/font/unicode_page_%02x.png' % page
    #img = PIL.Image.open(path)
    img = unicode_pages[page]
    assert img.width % 16 == 0
    assert img.height % 16 == 0
    iw = img.width // 16
    ih = img.height // 16
    box = [iw*ix, ih*iy, iw*(ix+1), ih*(iy+1)]
    print(x, y, c, ix, iy, box)
    if c == '\n':
      x = sx
      y += line_height
    else:
      x += char_width
      glyph = img.crop(box)
      if c not in [' ', '\r', '\t']:
        bg = draw(bg, glyph, x, y)
      #x += glyph.width
      x += char_width
  return bg


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
def barycentric(verts, p):
    ab = verts[2] - verts[0]
    ac = verts[1] - verts[0]
    pa = verts[0] - p
    u = tri_cross(
        [ab[0], ac[0], pa[..., 0]],
        [ab[1], ac[1], pa[..., 1]])
    v = [u[0] / u[2], u[1] / u[2]]
    bc = [1. - v[0] - v[1], v[1], v[0]]
    valid = tf.logical_and(
        tf.abs(u[2]) >= 1.0,
        tf.reduce_all(tf.stack(bc, axis=1) >= 0, axis=1))
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
def sample(tex, uv, mode="nearest", wrap_mode="wrap", unpack=False):
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
    nw_val = get(ix_nw, iy_nw)
    ne_val = get(ix_ne, iy_ne)
    sw_val = get(ix_sw, iy_sw)
    se_val = get(ix_se, iy_se)
    a = lambda x: x[..., tf.newaxis]
    out = nw_val * a(nw)
    out += ne_val * a(ne)
    out += sw_val * a(sw)
    out += se_val * a(se)
    return out



def viewport(x, y, width, height):
    hw = width * 0.5
    hh = height * 0.5
    return np.array([
        [hw, 0., 0., hw + x],
        [0., hh, 0., hh + y],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]], dtype=np.float32)




def tf_format_text(img, x, y, text, *args, line_height=17, text_spacing=1, fixed_width=True):
  text = tf.strings.format(text, args)
  return tf_fill_text(img=img, x=x, y=y, text=text, line_height=line_height, text_spacing=text_spacing, fixed_width=fixed_width)

def tf_fill_text(img, x, y, text, line_height=17, text_spacing=1, fixed_width=True):
  P = tf.strings.unicode_decode(text, 'UTF-8')
  num_chars = tf.shape(P)[0]
  img_shape = tf.shape(img)
  img_h = img_shape[0]
  img_w = img_shape[1]
  def _fn(i, color, px, py):
    character = tf.gather(P, i)
    width = tf.gather(tff.widths, character)
    if fixed_width:
      offset = 0
      w = tf.maximum(8, width*2)
    else:
      offset = tf.gather(tff.offsets, character)
      w = width*2
    w += text_spacing
    def _draw():
      with tf.name_scope('_draw'):
        x0 = offset + px
        x1 = offset + px + w
        y0 = py
        y1 = py + 16
        u0 = 0
        u1 = w / 16
        v0 = 0
        v1 = 16 / 16
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
          bc, valid = barycentric(verts_i, p)
          p = tf.boolean_mask(p, valid)
          bc = [tf.boolean_mask(bc[k], valid) for k in range(3)]
          u = tri_dot(uvs_i[..., 0], bc)
          v = tri_dot(uvs_i[..., 1], bc)
          uv = tf.stack([u, v], -1)
          glyph = tf.gather(tff.chars, character)
          value = tf.cast(sample(glyph, uv), tf.float32)
          inds = tf.cast(tf.stack([p[..., 1], p[..., 0]], axis=-1), tf.int32)
          c = clamp(tf.tensor_scatter_add(c, inds, value), 0, 255)
        c = tf.cast(c, tf.uint8)
        return c, px + w, py
    def _whitespace():
      return color, px + w, py
    def _newline():
      return color, x, py + line_height
    color, px, py = tf.cond(
        tf_in(character, ord('\n')),
        _newline,
        lambda: tf.cond(
          tf_in(character, ord('\r'), ord('\t'), ord(' ')),
          _whitespace,
          _draw))
    updates = []
    return updates, color, px, py
  _, color, px, py = sequential_for(_fn, 0, num_chars, img, x, y)
  return color, px, py

#r(tf_format_text(tf.zeros([32,16], dtype=tf.uint8), 0, 0, "He"))

if __name__ == "__main__":
  read_glyph_sizes("/Users/bb/Library/Application Support/minecraft/versions/1.13.2/1.13.2/assets/minecraft")
  s = 'Hello there!\n\tThis is a text test.\n噢，這個類別目?似乎是空的。\n請之後?回來查找新內容，而如果您是一?創作者，\n%s。\n'
  lines = s.split('\n')
  for i in range(256):
    s += chr(i)
    if i % 16 == 0 and i > 0:
      s += '\n'
  #s += ''.join([chr(i) for i in range(ord('a'), ord('z')+1)]) + '\n'
  #s = "Hello there!\nThis is a text test."
  #img = PIL.Image.new('RGBA', (512, 512), 0xFFFF00FF)
  #img = fill_text(img, s, 0, 0)
  #img.save('foo.png')

  import tensorflow as tf
  sess = tf.InteractiveSession()
  r = sess.run
  tff = TensorflowFont(session=sess)
  mop.cast(r(tf_format_text(tf.ones([16*5,16*10], dtype=tf.uint8)*128, 1, 1, s))[0], 'img').save('foo.png')
  mop.cast(r(tf_fill_text(tf.ones([1024,512], dtype=tf.uint8)*128, 10, 1, open('README.md').read(), fixed_width=False))[0], 'img').save('bar.png')
  mop.cast(r(tf_fill_text(tf.ones([1024,512], dtype=tf.uint8)*128, 10, 1, open('README.md').read(), fixed_width=True))[0], 'img').save('baz.png')
  # S = s.split('\n')[1]
  # P = tf.strings.unicode_decode(tf.constant(S, dtype=tf.string), 'UTF-8')
  # C = tf.gather(tff.chars, P)
  # W = tf.gather(tff.widths, P)
  # Z = tf.zeros_like(W)
  # X = tf.cumsum(W*2, -1) - W
  # strides = r((tf.ragged.range(X-W, X+W) - tf.reshape(X-W, [-1, 1])).to_tensor())
  # import pdb; pdb.set_trace()
  # tf.strings.unicode_decode(tf.constant(s, dtype=tf.string), 'UTF-8')
