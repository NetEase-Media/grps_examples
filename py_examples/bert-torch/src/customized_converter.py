# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Customized converter of model, including pre-process and post-process.
from grps_framework.apis.grps_pb2 import GrpsMessage
from grps_framework.context.context import GrpsContext
from grps_framework.converter.converter import Converter, converter_register
from grps_framework.logger.logger import clogger
from transformers import AutoTokenizer
import torch


class YourConverter(Converter):
    """Your converter."""

    def __init__(self):
        super().__init__()
        self.model_name = 'bert-base-chinese'
        self.tokenizer = None
        self.mask_token_id = 0

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

        self.mask_token_id = args['mask_token_id']
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

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
        samples = [inp.str_data]
        tokenizer_text = [self.tokenizer.tokenize(i) for i in samples]
        input_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in tokenizer_text]

        mask_pos = []
        for i in range(0, len(input_ids[0])):
            if input_ids[0][i] == self.mask_token_id:
                mask_pos.append(i)
        context.put_user_data('mask_pos', mask_pos)
        clogger.info('your converter preprocess, mask_pos: {}'.format(mask_pos))

        input_ids = torch.LongTensor(input_ids)
        return {'input_ids': input_ids}

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
        pred = inp['pred']
        output = self.tokenizer.convert_ids_to_tokens(pred)

        out = GrpsMessage()
        mask_pos = context.get_user_data('mask_pos')
        clogger.info('your converter postprocess, mask_pos: {}'.format(mask_pos))
        out.str_data = ''
        for pos in mask_pos:
            if out.str_data != '':
                out.str_data += '||'
            out.str_data += output[pos]

        return out


converter_register.register('your_converter', YourConverter())
