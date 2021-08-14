import tensorflow as tf

from PIL import Image
from PIL import ImageDraw

# Based on https://stackoverflow.com/q/49986522/859277 and
# https://www.tensorflow.org/tutorials/load_data/tfrecord#read_the_tfrecord_file

#import pudb; pu.db #a breakpoint

raw_image_dataset = tf.data.TFRecordDataset('data/file.tfrecord-00000-of-00001') 

count = 0
classes = set()
 
for value in raw_image_dataset: 
    image_feature_description = {
        'image/object/class/text': tf.io.VarLenFeature(tf.string),   # or is it this line that breaks? I can't tell
    }
    parsed = tf.io.parse_single_example(value, image_feature_description)

    classes.add(parsed['image/object/class/text'].values.numpy()[0])
    count += 1

print(count, classes)

it = iter(raw_image_dataset)

# for i in range(IMAGES_TO_SHOW):
while True:
    #pu.db #a breakpoint
    value = next(it)    #this line breaks too

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

    print(parsed['image/object/class/text'].values.numpy()[0])
    tf_image = tf.image.decode_jpeg(parsed['image/encoded'])
    image = Image.fromarray(tf_image.numpy())

    draw = ImageDraw.Draw(image)
    #import pudb; pu.db #a breakpoint
    x1 = parsed['image/object/bbox/xmin'].values[0].numpy() * image.width
    x2 = parsed['image/object/bbox/xmax'].values[0].numpy() * image.width
    y1 = parsed['image/object/bbox/ymin'].values[0].numpy() * image.height
    y2 = parsed['image/object/bbox/ymax'].values[0].numpy() * image.height

    #import pudb; pu.db #a breakpoint
    draw.rectangle((x1, y1, x2, y2))
    #image.show()
    image.save('data/boundingbox.jpg')
    input()