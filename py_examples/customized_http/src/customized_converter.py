# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Customized converter of model, including pre-process and post-process.
import http
import json
import traceback

from flask import Response
from grps_framework.apis.grps_pb2 import GrpsMessage
from grps_framework.context.context import GrpsContext
from grps_framework.converter.converter import Converter, converter_register
from grps_framework.logger.logger import clogger


class YourConverter(Converter):
    """Your converter."""

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
        try:
            clogger.info('your converter preprocess.')
            request = context.get_http_request()
            if request.content_type != 'application/json':
                raise Exception('content_type error, unsupported type: {}'.format(request.content_type))

            request_json = request.get_json()
            if 'a' not in request_json or 'b' not in request_json:
                raise Exception('request json error, need key a and b, request_json: {}'.format(request_json))

            a = float(request_json['a'])
            b = float(request_json['b'])
            return {'a': a, 'b': b}
        except Exception as e:
            error_message = traceback.format_exc()
            # context.set_has_err(True)
            context.set_err_msg(error_message)
            context.set_http_response(Response(response=error_message, status=http.HTTPStatus.INTERNAL_SERVER_ERROR,
                                               content_type="text/plain"))
            clogger.error('your inferer infer error: {}'.format(error_message))

        return {}

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
        try:
            clogger.info('your converter postprocess.')
            if len(inp) == 0 or 'c' not in inp:
                raise Exception('infer output error, need key c, inp: {}'.format(inp))

            context.set_http_response((json.dumps(inp), http.HTTPStatus.OK, {"Content-Type": "application/json"}))
        except Exception as e:
            error_message = traceback.format_exc()
            # context.set_has_err(True)
            context.set_err_msg(error_message)
            context.set_http_response(
                (error_message, http.HTTPStatus.INTERNAL_SERVER_ERROR, {"Content-Type": "text/plain"}))
            clogger.error('your inferer infer error: {}'.format(error_message))
        return GrpsMessage()


converter_register.register('your_converter', YourConverter())
