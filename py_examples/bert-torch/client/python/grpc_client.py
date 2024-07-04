# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Grpc client demo. Complete interface description can be learned from docs/2_Interface.md
import sys

import grpc
from grps_apis.grps_pb2 import GrpsMessage
from grps_apis.grps_pb2_grpc import GrpsServiceStub


def grpc_request(server, inp):
    conn = grpc.insecure_channel(server)
    client = GrpsServiceStub(channel=conn)

    # predict with str_data.
    request = GrpsMessage(str_data=inp)
    response = client.Predict(request)
    print('Predict response: {}, decoded str_data: {}'.format(response, response.str_data))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python grpc_client.py <server> <inp>')
        sys.exit(1)
    server = sys.argv[1]
    inp = sys.argv[2]
    grpc_request(server, inp)
