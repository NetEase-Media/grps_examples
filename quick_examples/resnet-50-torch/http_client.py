# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/8
# Brief  http client demo. Complete interface description can be learned from docs/2_Interface.md
import sys
import time

import cv2
import numpy as np
import requests

PREDICT_PATH = '/grps/v1/infer/predict'

USE_NDARRAY = False


class Client:
    def __init__(self, server):
        with open('./data/ImageNetLabels.txt') as f:
            self.__synset = f.readlines()

        self.__server = 'http://' + server

    def __preprocess(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = img[:, :, ::-1]  # BGR -> RGB
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])  # normalize
        img = np.transpose(img, (2, 0, 1))  # (224, 224, 3)
        img = np.expand_dims(img, axis=0)  # (1, 3, 224, 224)

        if USE_NDARRAY:
            return {'ndarray': img.tolist()}
        else:
            gtensors = {
                'tensors': [{
                    'name': '',  # name is not defined, will use default name.
                    'dtype': 'DT_FLOAT32',
                    'shape': [1, 3, 224, 224],
                    'flat_float32': img.flatten().tolist()
                }]
            }
            return {'gtensors': gtensors}

    def __predict(self, req_data):
        begin = time.time()
        if USE_NDARRAY:
            response = requests.post(self.__server + PREDICT_PATH, json=req_data, params={'return-ndarray': 'true'})
        else:
            response = requests.post(self.__server + PREDICT_PATH, json=req_data)
        print('predict time: {} ms'.format((time.time() - begin) * 1000))
        if USE_NDARRAY:
            return response.json()['ndarray']
        else:
            return np.array(response.json()['gtensors']['tensors'][0]['flat_float32']).reshape(
                response.json()['gtensors']['tensors'][0]['shape'])

    def __postprocess(self, scores):
        scores = np.array(scores)[0]
        top1 = np.argmax(scores)
        return self.__synset[top1]

    def __call__(self, img_path):
        req_data = self.__preprocess(img_path)
        scores = self.__predict(req_data)
        return self.__postprocess(scores)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: python3 http_client.py <server> <img_path>')
        sys.exit(1)
    server = sys.argv[1]
    img_path = sys.argv[2]

    # create client and predict
    client = Client(server)
    print(client(img_path))
