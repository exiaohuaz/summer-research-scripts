import os

import tensorflow as tf

from object_detection.protos import string_int_label_map_pb2
from object_detection.utils import dataset_util

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

from google.protobuf import text_format

import glob
import re
import json
import os

LABEL_MAP_NAME = 'data/label_map.pbtxt'

#labels
labels = []
label_map = string_int_label_map_pb2.StringIntLabelMap()
label = "default"
lid = 1

#constants
height = 480 # Image height   #get from actual file
width = 853 # Image width
image_format = b'jpg'

def create_tf_example(item):
  #print('Creating tfexample...')

  xmins = []
  xmaxs = [] 
  ymins = [] 
  ymaxs = [] 
  classes_text = [] 
  classes = [] 
  filename = item["filename"] + ".jpg"
  print("Filename: ", filename)
  
  with open(filename, "rb") as f2:
    encoded_image_data = f2.read() #solved by adding "rb" to open. specifies that it's binary?
    values = item["annotations"][0]["values"]
    for value in values:
      print('New value...')

      classes_text.append(label.encode())
      classes.append(lid)
      print("Label: ", label, " ID: ", lid)
      xmins.append((value["x"])/width)
      ymins.append((value["y"])/height)
      xmaxs.append((value["x"] + value["width"])/width)
      ymaxs.append((value["y"] + value["height"])/height)
      print('xmin: ', (value["x"]), ' ymin: ', (value["y"]), ' xmax: ', (value["x"] + value["width"]), ' ymax: ', (value["y"] + value["height"]))
    tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(filename.encode()),
    'image/source_id': dataset_util.bytes_feature(filename.encode()),
    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    'image/format': dataset_util.bytes_feature(image_format),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
  return tf_example

def main():
  index = 0
  num_shards = 1
  output_filebase = 'data/file.tfrecord'

  newlabel = string_int_label_map_pb2.StringIntLabelMapItem()
  newlabel.name = label
  newlabel.id = lid
  label_map.item.append(newlabel)

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
      tf_record_close_stack, output_filebase, num_shards)

    """
    for folder in os.listdir("captures"):
      for capture in os.listdir("captures/" + folder):
        print(folder + "/" + capture)
        with open(os.path.join("captures/" + folder, capture), 'r') as f1:
        #with open('captures_000.json') as f1:
          data = json.load(f1)
          data = data["captures"]

          for item in data:
            tf_example = create_tf_example(item)
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
            index = index + 1
    """
    for capture in os.listdir("captures/"):
      print(capture)
      with open(os.path.join("captures/", capture), 'r') as f1:
      #with open('captures_000.json') as f1:
        data = json.load(f1)
        data = data["captures"]

        for item in data:
          tf_example = create_tf_example(item)
          output_shard_index = index % num_shards
          output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
          index = index + 1

  with open(LABEL_MAP_NAME, 'w') as f:
    f.write(text_format.MessageToString(label_map))

if __name__ == '__main__':
  main()