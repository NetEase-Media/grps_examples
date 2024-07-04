# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Customized converter of model, including pre-process and post-process.
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from grps_framework.apis.grps_pb2 import GrpsMessage
from grps_framework.context.context import GrpsContext
from grps_framework.converter.converter import Converter, converter_register
from grps_framework.logger.logger import clogger


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
            path: Attachment path.
            args: More args.

        Raises:
            Exception: If init failed, can raise exception and exception will be caught by server and show error message
            to user when start service.
        """
        super().init(path, args)
        clogger.info('your converter init, path: {}, args: {}'.format(path, args))
        with open(path) as f:
            self.__synset = f.readlines()

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
        img = torch.from_numpy(img).to('cuda')
        img = img.float().div(255)
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear')
        # normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).to('cuda')
        std = torch.tensor([0.229, 0.224, 0.225]).to('cuda')
        channels = torch.split(img, 1, 1)
        channels_list = list(channels)
        for i in range(3):
            channels_list[i] = channels_list[i].sub(mean[i]).div(std[i])
        channels = tuple(channels_list)
        img = torch.cat(channels, 1)
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
        label = np.argmax(inp.cpu().detach().numpy()[0])
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
        imgs_futures = []

        def decode_fn(img_data):
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            img = torch.from_numpy(img).to('cuda')
            img = img.float().div(255)
            img = img.permute(2, 0, 1).unsqueeze(0)
            img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear')
            # normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).to('cuda')
            std = torch.tensor([0.229, 0.224, 0.225]).to('cuda')
            channels = torch.split(img, 1, 1)
            channels_list = list(channels)
            for i in range(3):
                channels_list[i] = channels_list[i].sub(mean[i]).div(std[i])
            channels = tuple(channels_list)
            img = torch.cat(channels, 1)
            return img

        for inp in inps:
            img_data = inp.bin_data
            imgs_futures.append(self.__batch_tp.submit(decode_fn, img_data))

        imgs = [future.result() for future in imgs_futures]

        return torch.cat(imgs, 0)

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
        labels = np.argmax(inp.cpu().detach().numpy(), axis=1)
        outs = []
        for label in labels:
            out = GrpsMessage(str_data=self.__synset[label].replace('\n', ''))
            outs.append(out)
        return outs


converter_register.register('your_converter', YourConverter())
