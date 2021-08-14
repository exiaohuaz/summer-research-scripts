import argparse

from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2
import numpy as np
import common
import cv2

import tensorflow as tf
from tensorflow.python.framework import ops


DETECTOR_ONES_SIZE = (1, 480, 640, 3)
CLASSIFIER_ONES_SIZE = (1, 224, 224, 3)
THRESHOLD = 0.4


class InferenceEngine(cognitive_engine.Engine):
    def __init__(self, object_detector_path, image_classifier_path, labels_path):
        self._object_detector = tf.saved_model.load(object_detector_path)
        ones = tf.ones(DETECTOR_ONES_SIZE, dtype=tf.uint8)
        self._object_detector(ones)

        self._classifier = tf.saved_model.load(image_classifier_path)
        it = iter(self._classifier.signatures.values())
        self._classifier_dtype = next(it).inputs[0].dtype        
        ones = tf.ones(CLASSIFIER_ONES_SIZE, dtype=self._classifier_dtype)
        self._classifier(ones)

        self._classifier_labels = np.array(
            open(labels_path).read().splitlines())

        self._count = 0

    def handle(self, input_frame):
        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)
        
        np_data = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detections = self._object_detector(np.expand_dims(img, 0))

        scores = detections['detection_scores'][0].numpy()
        boxes = detections['detection_boxes'][0].numpy()

        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)

        im_height, im_width = img.shape[:2]
        img_to_send = img

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
            img = cv2.resize(
                cropped, (CLASSIFIER_ONES_SIZE[1], CLASSIFIER_ONES_SIZE[2]))
            img = np.float32(img)
            img /= 255.0

            tensor = ops.convert_to_tensor(
                img[np.newaxis, ...], dtype=self._classifier_dtype)
            classifier_result = self._classifier(tensor)
            predicted_class = np.argmax(classifier_result[0], axis=-1)
            print(self._classifier_labels[predicted_class], score)

            predicted_class_label = self._classifier_labels[predicted_class]
            break            

        img_to_send = cv2.cvtColor(img_to_send, cv2.COLOR_BGR2RGB)        
        _, jpeg_img = cv2.imencode('.jpg', img_to_send)
        img_data = jpeg_img.tobytes()

        # if predicted_class_label == '4screws':
        #     with open('/sterling/3screws/wrong_4_{}.jpg'.format(self._count), 'wb') as f:
        #         f.write(img_data)
        #     self._count += 1

        # with open('/sterling/raw_{}.jpg'.format(self._count), 'wb') as f:
        #     f.write(input_frame.payloads[0])

        # with open('/sterling/cropped_{}.jpg'.format(self._count), 'wb') as f:
        #     f.write(img_data)            
        # if self._count == 1000:
        #     raise Exception
            
        # self._count += 1

        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = img_data

        result_wrapper.results.append(result)

        return result_wrapper


def main():
    common.configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('object_detector_path', type=str)
    parser.add_argument('image_classifier_path', type=str)
    parser.add_argument('labels_path', type=str)
    args = parser.parse_args()

    def engine_factory():
        return InferenceEngine(
            args.object_detector_path, args.image_classifier_path, args.labels_path)

    common.run_engine(engine_factory)


if __name__ == '__main__':
    main()
