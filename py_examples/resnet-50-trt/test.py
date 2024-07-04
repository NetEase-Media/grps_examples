# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Local unittest.
import unittest

from grps_framework.context.context import GrpsContext
from grps_framework.test import GrpsTest
from grps_framework.apis.grps_pb2 import GrpsMessage, GenericTensor, DataType
import src.customized_converter


class MyTestCase(GrpsTest):
    def test_infer(self):
        self.test_init()

        # Build input.
        grps_in = GrpsMessage()
        with open('./data/tabby.jpeg', 'rb') as f:
            grps_in.bin_data = f.read()

        # Infer.
        context = GrpsContext()
        grps_out = self.executor.infer(grps_in, context)

        # Check result.
        self.assertEqual(grps_out.str_data, 'tabby, tabby cat')


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(MyTestCase('test_infer'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
