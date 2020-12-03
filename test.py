s = '噢，這個類別目?似乎是空的。\n請之後?回來查找新內容，而如果您是一?創作者，\n%s。'
import mel
import mel.font
import mel.image
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import PIL.Image
tf1.disable_v2_behavior()

sess = tf1.InteractiveSession()

mel.font.read_glyph_sizes("/Users/bb/Library/Application Support/minecraft/versions/1.13.2/1.13.2/assets/minecraft")
font = mel.font.init_font(sess)

src = tf.ones([10, 10], dtype=tf.uint8) * 255
with PIL.Image.open('../media/smile.png') as f:
  src = f.convert('RGBA')
  src = mel.cast(src, 'u8')
  src = src[:, :, 0]
img = tf.zeros([256,256], dtype=tf.uint8)
img, px, py = mel.font.tf_fill_text(font, img, 0, 0, s)
img, px, py = mel.image.tf_draw(src, img, 0, 0, 256, 256, 0.0, 0.0, 1.0, 1.0)
img = mel.val(img, True)
print(img)
print(img.shape)
img = mel.cast(img, 'img')
img.save('foo.png')

