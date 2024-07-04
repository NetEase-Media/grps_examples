# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Grpc client demo. Complete interface description can be learned from docs/2_Interface.md
import sys

import grpc
from grps_apis.grps_pb2 import GrpsMessage, GenericTensor, DataType
from grps_apis.grps_pb2_grpc import GrpsServiceStub


def grpc_request(server):
    conn = grpc.insecure_channel(server)
    client = GrpsServiceStub(channel=conn)

    # predict streaming.
    request = GrpsMessage(str_data='this process')
    response = client.PredictStreaming(request)
    for resp in response:
        print(resp)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 grpc_client.py <server>')
        sys.exit(1)

    grpc_request(sys.argv[1])
