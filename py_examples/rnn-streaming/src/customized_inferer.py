# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Customized deep learning model inferer. Including model load and model infer.
import time
import torch
import numpy as np
from grps_framework.apis import grps_pb2
from grps_framework.context.context import GrpsContext
from grps_framework.logger.logger import clogger
from grps_framework.model_infer.inferer import ModelInferer, inferer_register
from grps_framework.model_infer.torch_inferer import TorchModelInferer


class YourInferer(TorchModelInferer):
    def __init__(self):
        self.word_to_ix = {}
        self.ix_to_word = {}
        self.pred_len = 0

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
        self.pred_len = args['pred_length']
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
        with open('./data/word_to_ix.txt', 'r') as f:
            for line in f:
                word, index = line.strip().split()
                self.word_to_ix[word] = int(index)
                self.ix_to_word[int(index)] = word
        TorchModelInferer.load(self)
        clogger.info('your inferer load.')
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
        # Prepare first input tensor.
        inp_s = inp.str_data
        predicted_word = inp_s
        hidden = torch.zeros(1, 1, 100)
        index_list = np.array([self.word_to_ix[w] for w in inp_s.split()])
        if len(index_list) != 2:
            raise Exception('input length must be 2')
        inp = torch.as_tensor(index_list[-2:], dtype=torch.long)

        for p in range(self.pred_len):
            if context.if_streaming():  # If streaming, send predicted word to client
                if context.if_disconnected():
                    break
                context.stream_respond(grps_pb2.GrpsMessage(str_data=predicted_word))

            output, hidden = TorchModelInferer.infer(self, [inp, hidden], context)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(1).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted word to string and use as next input
            predicted_word = self.ix_to_word[top_i.item()]
            index_list[0] = index_list[1]
            index_list[1] = top_i

            if not context.if_streaming():
                inp_s += " " + predicted_word

        if context.if_streaming():  # Send the last predicted word to client.
            context.stream_respond(grps_pb2.GrpsMessage(str_data=predicted_word))

        return grps_pb2.GrpsMessage(str_data=inp_s)


# Register
inferer_register.register('your_inferer', YourInferer())
