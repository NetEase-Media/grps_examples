# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Local unittest.
import unittest
import flask

from grps_framework.apis.grps_pb2 import GrpsMessage
from grps_framework.context.context import GrpsContext
from grps_framework.test import GrpsTest
# import to register customized converter and inferer.
from src.customized_converter import converter_register
from src.customized_inferer import inferer_register


class MyTestCase(GrpsTest):
    def test_infer(self):
        self.assertGreater(len(converter_register.converter_dict), 0)
        self.assertGreater(len(inferer_register.model_inferer_dict), 0)
        self.test_init()

        # Build flask request
        data = {
            "a": 1,
            "b": 2,
        }
        request_obj = flask.Request(environ={'REQUEST_METHOD': 'POST'})
        request_obj._cached_json = (data, data)
        request_obj.headers = {'Content-Type': 'application/json'}

        context = GrpsContext(http_request=request_obj)
        _ = self.executor.infer(GrpsMessage(), context)  # Use customized http, request msg and response msg is useless.

        # Check http response.
        self.assertEqual(context.has_err(), False)
        response_obj = context.get_http_response()
        self.assertEqual(response_obj.status_code, 200)
        self.assertEqual(response_obj.headers['Content-Type'], 'application/json')
        self.assertIsNotNone(response_obj.response)
        self.assertEqual(response_obj.response[0], b'{"c": 3.0}')


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(MyTestCase('test_infer'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
