# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Http client demo. Complete interface description can be learned from docs/2_Interface.md.
import sys

import requests


def http_request(server, inp):
    url = f'http://{server}'

    # predict with str_data.
    response = requests.post(url + '/grps/v1/infer/predict', json={'str_data': inp}).json()
    print(response)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python http_client.py <server> <inp>')
        sys.exit(1)
    server = sys.argv[1]
    inp = sys.argv[2]
    http_request(server, inp)
