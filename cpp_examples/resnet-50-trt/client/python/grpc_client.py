# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Grpc client demo. Complete interface description can be learned from docs/2_Interface.md

import sys

import grpc
from grps_apis.grps_pb2 import GrpsMessage
from grps_apis.grps_pb2_grpc import GrpsServiceStub


def grpc_request(server, img_path):
    conn = grpc.insecure_channel(server)
    client = GrpsServiceStub(channel=conn)

    request = GrpsMessage()
    with open(img_path, 'rb') as f:
        request.bin_data = f.read()
    response = client.Predict(request)
    print(response)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python grpc_client.py <server> <img_path>')
        sys.exit(1)
    server = sys.argv[1]
    img_path = sys.argv[2]
    grpc_request(server, img_path)
