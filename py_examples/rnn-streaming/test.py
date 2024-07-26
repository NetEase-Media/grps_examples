# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Local unittest.
import unittest

from grps_framework.apis.grps_pb2 import GrpsMessage
from grps_framework.context.context import GrpsContext
from grps_framework.test import GrpsTest
# import to register customized converter and inferer.
from src.customized_inferer import inferer_register


class MyTestCase(GrpsTest):
    def test_infer(self):
        self.assertGreater(len(inferer_register.model_inferer_dict), 0)

        self.test_init()

        grps_in = GrpsMessage()
        # Add your codes to set input as follows:
        grps_in.str_data = 'this process'

        # Infer.
        context = GrpsContext()
        grps_out = self.executor.infer(grps_in, context)

        self.assertEqual(context.has_err(), False)

        # Check your result as follows:
        #self.assertEqual(grps_out.str_data, 'this process however afforded mean')


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(MyTestCase('test_infer'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
