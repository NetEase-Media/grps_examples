# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2024/6/19
# Brief  Grpc client demo. Complete interface description can be learned from docs/2_Interface.md

import sys
import time

import cv2
import grpc
import numpy as np
from grps_apis.grps_pb2 import DataType, GenericTensor, GrpsMessage
from grps_apis.grps_pb2_grpc import GrpsServiceStub

PREDICT_PATH = '/grps/v1/infer/predict'


class Client:
    def __init__(self, server):
        with open('./data/imagenet1000_clsid_to_human.txt') as f:
            self.__synset = eval(f.read())

        self.__server = server

    def __preprocess(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = img[:, :, ::-1]  # BGR -> RGB
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])  # normalize
        img = np.transpose(img, (2, 0, 1))  # (224, 224, 3)
        img = np.expand_dims(img, axis=0)  # (1, 3, 224, 224)

        request = GrpsMessage()
        request.gtensors.tensors.append(
            GenericTensor(
                name='',  # name is not defined, will use default name.
                dtype=DataType.DT_FLOAT32,
                shape=[1, 3, 224, 224],
                flat_float32=img.flatten().tolist()))
        return request

    def __predict(self, request):
        # predict with gtensors.
        grpc_client = GrpsServiceStub(channel=grpc.insecure_channel(self.__server))

        begin = time.time()
        out = grpc_client.Predict(request)
        print('predict time: {} ms'.format((time.time() - begin) * 1000))
        return out

    def __postprocess(self, response):
        # gtensors to numpy
        scores = np.array(response.gtensors.tensors[0].flat_float32).reshape(response.gtensors.tensors[0].shape)

        scores = np.array(scores)[0]
        top1 = np.argmax(scores)
        return self.__synset[top1]

    def __call__(self, img_path):
        req_data = self.__preprocess(img_path)
        scores = self.__predict(req_data)
        return self.__postprocess(scores)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: python3 grpc_client.py <server> <img_path>')
        sys.exit(1)
    server = sys.argv[1]
    img_path = sys.argv[2]

    # create client and predict
    client = Client(server)
    print(client(img_path))
