# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Http client demo. Complete interface description can be learned from docs/2_Interface.md.

import sys

import requests


def http_request(server, img_path):
    url = server

    # predict with bin_data.
    with open(img_path, 'rb') as f:
        img_data = f.read()
    response = requests.post(url + '/grps/v1/infer/predict', data=img_data,
                             headers={'content-type': 'application/octet-stream'}).json()
    print(response)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python http_client.py <server> <img_path>')
        sys.exit(1)
    server = sys.argv[1]
    img_path = sys.argv[2]
    http_request(server, img_path)
