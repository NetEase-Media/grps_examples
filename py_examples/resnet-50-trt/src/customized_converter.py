# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Customized converter of model, including pre-process and post-process.
import os

import cv2
import numpy as np
from grps_framework.apis.grps_pb2 import GrpsMessage
from grps_framework.context.context import GrpsContext
from grps_framework.converter.converter import Converter, converter_register
from grps_framework.logger.logger import clogger
from concurrent.futures import ThreadPoolExecutor


class YourConverter(Converter):
    """Your converter."""

    def __init__(self):
        super().__init__()
        self.__synset = None
        self.__batch_tp = ThreadPoolExecutor(max_workers=os.cpu_count())

    def init(self, path=None, args=None):
        """
        Init converter.

        Args:
            path: Path.
            args: More args.

        Raises:
            Exception: If init failed, can raise exception and exception will be caught by server and show error message
            to user when start service.
        """
        super().init(path, args)
        clogger.info('your converter init, path: {}, args: {}'.format(path, args))
        with open(path) as f:
            self.__synset = eval(f.read())

    def preprocess(self, inp: GrpsMessage, context: GrpsContext):
        """
        Preprocess.

        Args:
            inp: Input message from client or previous model(multi model sequential mode).
            context: Grps context of current request.

        Returns:
            Pre-processed data which is input of model inferer.

        Raises:
            Exception: If preprocess failed, can raise exception and exception will be caught by server and return error
            message to client.
        """
        img_data = inp.bin_data
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = np.array(img)[np.newaxis, :].astype("float32")
        img = img[:, :, :, ::-1]  # BGR -> RGB
        img = np.float32(img) / 255.0
        img[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
        img[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
        img = img.transpose((0, 3, 1, 2))
        return img

    def postprocess(self, inp, context: GrpsContext) -> GrpsMessage:
        """
        Postprocess.

        Args:
            inp: Input to be post-processed, which is output of model inferer.
            context: Grps context of current request.

        Returns:
            Post-processed data with GrpsMessage format to client or next model(multi model sequential mode).

        Raises:
            Exception: If postprocess failed, can raise exception and exception will be caught by server and return error
            message to client.
        """
        inp = inp['495'] # out tensor name.
        label = np.argmax(inp[0])
        out = GrpsMessage(str_data=self.__synset[label].replace('\n', ''))
        return out

    def batch_preprocess(self, inps: list, contexts: list):
        """
        Batch preprocess.

        Args:
            inps: Input messages from client or previous model(multi model sequential mode).
            contexts: Grps contexts of current requests.

        Returns:
            Pre-processed data which is input of model inferer.

        Raises:
            Exception: If preprocess failed, can raise exception and exception will be caught by server and return error
            message to client.
        """
        imgs_future = []

        def decode_fn(img_data):
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            img = np.array(img).astype("float32")
            img = img[:, :, ::-1]  # BGR -> RGB
            img = np.float32(img) / 255.0
            img[:, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
            img[:, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
            img = img.transpose((2, 0, 1))
            return img

        for inp in inps:
            img_data = inp.bin_data
            imgs_future.append(self.__batch_tp.submit(decode_fn, img_data))

        imgs = [future.result() for future in imgs_future]
        return np.array(imgs)

    def batch_postprocess(self, inp, contexts: list) -> list:
        """
        Batch postprocess.

        Args:
            inp: Input to be post-processed, which is output of model inferer.
            contexts: Grps contexts of current requests.

        Returns:
            Post-processed data with GrpsMessage format to client or next model(multi model sequential mode).

        Raises:
            Exception: If postprocess failed, can raise exception and exception will be caught by server and return
            error message to client.
        """
        inp = inp['495'] # out tensor name.
        labels = np.argmax(inp, axis=1)
        outs = []
        for label in labels:
            out = GrpsMessage(str_data=self.__synset[label].replace('\n', ''))
            outs.append(out)
        return outs


converter_register.register('your_converter', YourConverter())
