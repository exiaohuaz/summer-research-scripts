import logging
import argparse

from gabriel_server import cognitive_engine
from gabriel_server import local_engine
from gabriel_protocol import gabriel_pb2
import numpy as np
import cv2

from tensorflow.python.framework import ops
import tensorflow as tf
import tensorflow_hub as hub

SOURCE_NAME = 'roundtrip'
INPUT_QUEUE_MAXSIZE = 60
PORT = 9099
NUM_TOKENS = 2


class InferenceEngine(cognitive_engine.Engine):
    def __init__(self, saved_model_path, labels_path):
        self._classify = hub.load(saved_model_path)
        it = iter(self._classify.signatures.values())
        self._model_dtype = next(it).inputs[0].dtype
        self._labels = np.array(open(labels_path).read().splitlines())
    
    def handle(self, input_frame):
        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)

        np_data = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        resized_img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img = np.float32(img)
        img /= 255.0

        # Based on
        # https://github.com/tensorflow/tensorflow/blob/042a6923d7f9ff05e7514bf6011e4ca30be70113/tensorflow/python/keras/engine/data_adapter.py#L1011
        tensor = ops.convert_to_tensor(
            img[np.newaxis, ...], dtype=self._model_dtype)
        output = self._classify(tensor)
        predicted_class = np.argmax(output[0], axis=-1)
        print(self._labels[predicted_class])

        _, jpeg_img = cv2.imencode('.jpg', resized_img)
        img_data = jpeg_img.tobytes()
        
        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = img_data

        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        result_wrapper.results.append(result)

        return result_wrapper


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('saved_model_path', type=str)
    parser.add_argument('labels_path', type=str)
    args = parser.parse_args()

    def engine_factory():
        return InferenceEngine(args.saved_model_path, args.labels_path)

    local_engine.run(
        engine_factory, SOURCE_NAME, INPUT_QUEUE_MAXSIZE, PORT, NUM_TOKENS)


if __name__ == '__main__':
    main()
