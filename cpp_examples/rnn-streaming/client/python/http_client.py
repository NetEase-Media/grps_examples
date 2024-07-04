# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Http client demo. Complete interface description can be learned from docs/2_Interface.md.
import sys

import requests


def http_request(server):
    url = f'http://{server}'

    # streaming predict.
    response = requests.post(url + '/grps/v1/infer/predict', json={'str_data': 'this process'},
                             params={'streaming': 'true'}, stream=True)
    for trunk in response.iter_content(chunk_size=None):
        print(trunk)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 http_client.py <server>')
        sys.exit(1)

    http_request(sys.argv[1])
