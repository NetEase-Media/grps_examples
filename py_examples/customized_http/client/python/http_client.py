# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/9/5
# Brief  Http client demo. Complete interface description can be learned from docs/2_Interface.md.
import sys

import requests


def http_request(server):
    url = f'http://{server}'

    # check liveness.
    response = requests.get(url + '/grps/v1/health/live').json()
    print(response)

    # online server.
    response = requests.get(url + '/grps/v1/health/online').json()
    print(response)

    # check readiness.
    response = requests.get(url + '/grps/v1/health/ready').json()
    print(response)

    # predict with str_data.
    response = requests.post(url + '/custom_predict', json={'a': '1', 'b': '2.0'})
    print(response.status_code, response.text, response.headers['Content-Type'])

    # except 500 and error about need key a and b
    response = requests.post(url + '/custom_predict', json={'a': '1'})
    print('\nexcept 500 and error about need key a and b\n')
    print(response.status_code, response.text, response.headers['Content-Type'])

    # except 500 and error about ccc could not convert to float.
    response = requests.post(url + '/custom_predict', json={'a': '1', 'b': 'ccc'})
    print('\nexcept 500 and error about ccc could not convert to float.\n')
    print(response.status_code, response.text, response.headers['Content-Type'])

    # get server metadata.
    response = requests.get(url + '/grps/v1/metadata/server').json()
    print(response)

    # get model metadata.
    response = requests.post(url + '/grps/v1/metadata/model', json={'str_data': 'your_model'}).json()
    print(response)

    # offline server.
    response = requests.get(url + '/grps/v1/health/offline').json()
    print(response)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 http_client.py <server>')
        sys.exit(1)

    http_request(sys.argv[1])
