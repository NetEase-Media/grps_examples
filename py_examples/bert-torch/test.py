# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Local unittest.
import unittest

from grps_framework.test import GrpsTest
from grps_framework.context.context import GrpsContext
from grps_framework.apis.grps_pb2 import GrpsMessage, GenericTensor, DataType
import src.customized_converter
import src.customized_inferer


class MyTestCase(GrpsTest):
    def test_infer(self):
        self.test_init()

        grps_in = GrpsMessage()
        # Add your codes to set input as follows:
        grps_in.str_data = '[CLS] 中国的首都是哪里？ [SEP] 北京是 [MASK] 国的首都。 [SEP]'

        # Infer.
        context = GrpsContext()
        grps_out = self.executor.infer(grps_in, context)

        # Check your result as follows:
        self.assertEqual(grps_out.str_data, '中')


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(MyTestCase('test_infer'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
