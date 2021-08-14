import argparse

import numpy as np
import common
import cv2
import os

import imagehash
from PIL import Image

import tensorflow as tf
from tensorflow.python.framework import ops


DETECTOR_ONES_SIZE = (1, 480, 640, 3)
CLASSIFIER_ONES_SIZE = (1, 224, 224, 3)
THRESHOLD = 0.4

DIFF_THRESHOLD = 10
NUM_SAME_HASH = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('object_detector_path', type=str)
    parser.add_argument('video', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()

    object_detector = tf.saved_model.load(args.object_detector_path)
    ones = tf.ones(DETECTOR_ONES_SIZE, dtype=tf.uint8)
    object_detector(ones)

    count = 0
    frames_same_hash = 0
    last_hash = None
    last_hash_count = 0
    last_hash_new = False

    video_capture = cv2.VideoCapture(args.video)
    _, frame = video_capture.read()

    while frame is not None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = object_detector(np.expand_dims(rgb_frame, 0))

        scores = detections['detection_scores'][0].numpy()
        boxes = detections['detection_boxes'][0].numpy()

        im_height, im_width = rgb_frame.shape[:2]

        for score, box in zip(scores, boxes):
            if score < THRESHOLD:
                continue

            # from https://github.com/tensorflow/models/blob/39f98e30e7fb51c8b7ad58b0fc65ee5312829deb/research/object_detection/utils/visualization_utils.py#L1232
            ymin, xmin, ymax, xmax = box

            # from https://github.com/tensorflow/models/blob/39f98e30e7fb51c8b7ad58b0fc65ee5312829deb/official/vision/detection/utils/object_detection/visualization_utils.py#L192
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)

            # Based on https://stackoverflow.com/a/15589825/859277
            cropped = frame[int(top):int(bottom), int(left):int(right)]
            _, jpeg = cv2.imencode('.jpg', cropped)
            
            filename = '{}.jpg'.format(count)
            filepath = os.path.join(args.output_path, filename)

            with open(filepath, 'wb') as f:
                f.write(jpeg.tobytes())
            print('.', end='')

            count += 1
            break

        _, frame = video_capture.read()

    print(count)


if __name__ == '__main__':
    main()
