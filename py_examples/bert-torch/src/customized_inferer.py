# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Customized deep learning model inferer. Including model load and model infer.
from grps_framework.context.context import GrpsContext
from grps_framework.model_infer.inferer import ModelInferer, inferer_register
from grps_framework.logger.logger import clogger
from transformers import AutoModelForMaskedLM
import numpy as np
import torch


class YourInferer(ModelInferer):
    def __init__(self):
        super().__init__()
        self.model_name = 'bert-base-chinese'
        self.model = None

    def init(self, path, device=None, args=None):
        """
        Initiate model inferer class with model path and device.

        Args:
            path: Model path, it can be a file path or a directory path.
            device: Device to run model.
            args: More args.

        Raises:
            Exception: If init failed, can raise exception. Will be caught by server and show error message to user when
            start service.
        """
        super(YourInferer, self).init(path, device, args)
        if not self._device:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device in ['cuda', 'gpu']:
            self._device = 'cuda'
        elif device == 'cpu':
            self._device = 'cpu'
        else:
            raise ValueError('Invalid device: {}, must be cuda, gpu or cpu.'.format(device))
        clogger.info('your infer init, path: {}, device: {}, args: {}.'.format(path, device, args))

    def load(self):
        """
        Load model from model path.

        Returns:
            True if load model successfully, otherwise False.

        Raises:
            Exception: If load failed, can raise exception and exception will be caught by server and show error message
            to user when start service.
        """
        self.model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-chinese").to(self._device)
        self.model.eval()
        clogger.info('your inferer loaded, path: {}'.format(self._path))
        return True

    def infer(self, inp, context: GrpsContext):
        """
        The inference function is used to make a prediction call on the given input request.

        Args:
            context: grps context
            inp: Model infer input, which is output of converter preprocess function. When in `no converter mode`, will
            skip converter preprocess and directly use GrpsMessage as input.

        Returns:
            Model infer output, which will be input of converter postprocess function. When in `no converter mode`, it
            will skip converter postprocess and should directly use GrpsMessage as output.

        Raises:
            Exception: If infer failed, can raise exception and exception will be caught by server and return error
            message to client.
        """
        input_ids = inp['input_ids'].to(self._device)
        outputs = self.model(input_ids)
        sample = outputs[0][0].detach().cpu().numpy()

        pred = np.argmax(sample, axis=1)
        return {'pred': pred}


# Register
inferer_register.register('your_inferer', YourInferer())
