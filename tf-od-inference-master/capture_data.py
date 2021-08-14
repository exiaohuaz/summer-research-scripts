import argparse

from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2
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


class InferenceEngine(cognitive_engine.Engine):
    def __init__(self, object_detector_path, label, count_prefix):
        self._object_detector = tf.saved_model.load(object_detector_path)
        ones = tf.ones(DETECTOR_ONES_SIZE, dtype=tf.uint8)
        self._object_detector(ones)

        self._label = label
        self._count_prefix = count_prefix

        self._cropped_img_dir = os.path.join('/output', self._label)
        os.makedirs(self._cropped_img_dir, exist_ok=True)

        self._count = 0
        self._frames_same_hash = 0
        self._last_hash = None
        self._last_hash_count = 0

        self._saved_images_hashes = []

    def handle(self, input_frame):
        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)

        if len(input_frame.payloads) == 0:
            status = gabriel_pb2.ResultWrapper.Status.SUCCESS
            result_wrapper = cognitive_engine.create_result_wrapper(status)

            return result_wrapper

        img_data = input_frame.payloads[0]
        np_data = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detections = self._object_detector(np.expand_dims(img, 0))

        scores = detections['detection_scores'][0].numpy()
        boxes = detections['detection_boxes'][0].numpy()

        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)

        im_height, im_width = img.shape[:2]

        predicted_class_label = 'none'

        for score, box in zip(scores, boxes):
            if score < THRESHOLD:
                continue

            # from https://github.com/tensorflow/models/blob/39f98e30e7fb51c8b7ad58b0fc65ee5312829deb/research/object_detection/utils/visualization_utils.py#L1232
            ymin, xmin, ymax, xmax = box

            # from https://github.com/tensorflow/models/blob/39f98e30e7fb51c8b7ad58b0fc65ee5312829deb/official/vision/detection/utils/object_detection/visualization_utils.py#L192
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)

            # Based on https://stackoverflow.com/a/15589825/859277
            cropped = img[int(top):int(bottom), int(left):int(right)]

            img_to_send = cropped
            img_to_send = cv2.cvtColor(img_to_send, cv2.COLOR_RGB2BGR)
            _, jpeg_img = cv2.imencode('.jpg', img_to_send)
            img_data = jpeg_img.tobytes()                        

            img_hash = imagehash.phash(Image.fromarray(cropped))
            if ((self._last_hash is None) or
                ((img_hash - self._last_hash) >= DIFF_THRESHOLD)):
                self._last_hash = img_hash
                self._last_hash_count = 0

                self._img_data_for_last_hash = img_data
                self._last_hash_new = True
                print('new frame')
                for saved_images_hash in self._saved_images_hashes:
                    if (saved_images_hash - img_hash) < DIFF_THRESHOLD:
                        self._last_hash_new = False
                        print('existing hash')
                        break
            else:
                self._last_hash_count += 1

            if ((self._last_hash_count != NUM_SAME_HASH) or
                (not self._last_hash_new)):
                break

            self._saved_images_hashes.append(self._last_hash)

            filename = '{}_{}.jpg'.format(self._count_prefix, self._count)            
            filepath = os.path.join(self._cropped_img_dir, filename)
                        
            with open(filepath, 'wb') as f:
                # We do not save the most recent image because the hash
                # value might have drifted
                f.write(self._img_data_for_last_hash)
            print(self._count)

            self._count += 1

            break

        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = img_data

        result_wrapper.results.append(result)

        return result_wrapper


def main():
    common.configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('object_detector_path', type=str)
    parser.add_argument('label', type=str)
    parser.add_argument('count_prefix', type=int)
    args = parser.parse_args()

    def engine_factory():
        return InferenceEngine(
            args.object_detector_path, args.label, args.count_prefix)

    common.run_engine(engine_factory)


if __name__ == '__main__':
    main()
