# Http client demo. Complete interface description can be learned from docs/2_Interface.md.

import base64
import sys

import requests


def http_request(server, prompt):
    url = f'http://{server}/generate'

    data = {
        'prompt': prompt,
        'sampling_params': {
            'temperature': 0.1,
            'top_p': 0.5,
            'max_tokens': 4096
        }
    }

    response = requests.post(url, json=data)
    if response.status_code != 200:
        print(f'Request failed, status code: {response.status_code}')
        return

    print(response.content.decode('utf-8'), flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 http_client.py <server> <prompt>')
        sys.exit(1)

    while True:
        http_request(sys.argv[1], sys.argv[2])
