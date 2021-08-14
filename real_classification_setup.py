import tensorflow as tf
import os

from PIL import Image
from PIL import ImageDraw

# Based on https://stackoverflow.com/q/49986522/859277 and
# https://www.tensorflow.org/tutorials/load_data/tfrecord#read_the_tfrecord_file

#import pudb; pu.db #a breakpoint

testdir = "real_raw"
output_dir = "real/train"

classcounts = {"back": 0, "front": 0, "fronttwo": 0, "full": 0, "mid": 0}

for inputdir in os.listdir(testdir):
  tfrecord_dir = os.path.join(testdir, inputdir, "default.tfrecord")
  dataset = tf.data.TFRecordDataset(tfrecord_dir) 
  it = iter(dataset)
  for value in it:    #this line breaks too

    image_feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
    }
    parsed = tf.io.parse_single_example(value, image_feature_description)

    classname = str(parsed['image/object/class/text'].values.numpy()[0])
    classname = classname[2:len(classname)-1]
    classcounts[classname] += 1
    if not os.path.isdir(os.path.join(output_dir, classname)):
        os.mkdir(os.path.join(output_dir, classname))

    tf_image = tf.image.decode_jpeg(parsed['image/encoded'])
    image = Image.fromarray(tf_image.numpy())

    x1 = parsed['image/object/bbox/xmin'].values[0].numpy() * image.width
    x2 = parsed['image/object/bbox/xmax'].values[0].numpy() * image.width
    y1 = parsed['image/object/bbox/ymin'].values[0].numpy() * image.height
    y2 = parsed['image/object/bbox/ymax'].values[0].numpy() * image.height

    image_cropped = image.crop((x1, y1, x2, y2))

    fnum = f'{classcounts[classname]:05d}'
    image_cropped.save(os.path.join(output_dir, classname, classname + "_" + fnum + ".jpg"))