import logging
import argparse

from gabriel_server import cognitive_engine
from gabriel_server import local_engine
from gabriel_protocol import gabriel_pb2
import numpy as np
import cv2

import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


SOURCE_NAME = 'roundtrip'
INPUT_QUEUE_MAXSIZE = 60
PORT = 9099
NUM_TOKENS = 2


class InferenceEngine(cognitive_engine.Engine):
    def __init__(self, host_and_port):
        channel = grpc.insecure_channel(host_and_port)
        self._service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    
    def handle(self, input_frame):
        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)

        np_data = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        resized_img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img /= 255.0

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'knex'
        tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
        request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(tensor))
        result_future = self._service.Predict.future(request, 5.0)  # 5 seconds

        def callback(result_future):
            response = np.array(
                result_future.result().outputs['dense'].float_val)
            print(response)

        result_future.add_done_callback(callback)
            
        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = input_frame.payloads[0]

        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        result_wrapper.results.append(result)

        return result_wrapper


def main():
    logging.basicConfig(level=logging.INFO)

    def engine_factory():
        return InferenceEngine('127.0.0.1:8500')

    local_engine.run(
        engine_factory, SOURCE_NAME, INPUT_QUEUE_MAXSIZE, PORT, NUM_TOKENS)


if __name__ == '__main__':
    main()
