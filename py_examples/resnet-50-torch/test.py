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

        grps_in = GrpsMessage()
        # Add your codes to set input as follows:
        # grps_in.str_data = 'hello grps'
        with open('./data/tabby.jpeg', 'rb') as f:
            grps_in.bin_data = f.read()
        # grps_in.bin_data = b'hello grps'
        # grps_in.gTensors.names.append('inp')
        # gtensor = GenericTensor(dtype=DataType.DT_FLOAT32, shape=[1, 2], flat_float32=[1, 2])
        # grps_in.gTensors.tensors.append(gtensor)

        # # Infer.
        context = GrpsContext()
        grps_out = self.executor.infer(grps_in, context)

        # Check your result as follows:
        # self.assertEqual(grps_out.str_data, 'hello grps')
        # self.assertEqual(grps_out.bin_data, b'hello grps')
        # self.assertEqual(grps_out.gTensors.names[0], 'inp')
        # gtensor = GenericTensor(dtype=DataType.DT_FLOAT32, shape=[1, 2], flat_float32=[1, 2])
        # self.assertEqual(grps_out.gTensors.tensors[0], gtensor)
        self.assertEqual(grps_out.str_data, 'tabby')


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(MyTestCase('test_infer'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
