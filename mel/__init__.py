from .common import *
from . import tf
from . import font


# from . import _version
# __version__ = _version.__version__


# Version info -- read without importing
import os
_locals = {}
__dir__ = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(__dir__, "_version.py")) as fp:
  exec(fp.read(), None, _locals)
__version__ = _locals["__version__"]


# mel.cast(mel.swizzled(mel.fill([8,4], [90, 100, 200]), 'bgrF'), 'img').save('check.png')

# pp([[v, hex(int(('%05.2f' % (v/255.0*16)).split('.')[0]))[-1].upper()] for v in range(0,256,5)])

# with open('../media/smile.png', 'rb') as f: smile = r(tf.io.decode_image(f.read()))
# h = 40; w = 40; mel.cast(r(tf.gather_nd(smile, mel.cast(mtf.wrap(mel.thru(mel.swizzled((mel.stack(mel.meshgrid(mel.linspace(w-0.5, 1.0-0.5, w) / w, mel.linspace(h-0.5, 1.0-0.5, h) / h))), 'vu')) * 4)*10, 'i32'))), 'img').save('check2.png')
