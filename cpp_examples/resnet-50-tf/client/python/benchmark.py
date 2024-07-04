# Copyright 2022 netease. All rights reserved.
# Author zhaochaochao@corp.netease.com
# Date   2023/5/20
# Brief benchmark for resnet-50-tf grps++ server.
import sys
import threading
import time

import requests

PREDICT_PATH = '/grps/v1/infer/predict'

if __name__ == '__main__':
    # parse server and img_path and concurrency from command line
    if len(sys.argv) != 4:
        print('usage: python3 benchmark.py <server> <img_path> <concurrency>')
        sys.exit(1)
    server = sys.argv[1]
    img_path = sys.argv[2]
    concurrency = int(sys.argv[3])

    # prepare request data
    with open(img_path, 'rb') as f:
        img = f.read()
    # prepare request headers
    headers = {'Content-Type': 'application/octet-stream'}


    def send_request():
        while True:
            begin = time.time()
            requests.post(server + PREDICT_PATH, data=img, headers=headers)
            end = time.time()
            print('time cost: {} ms'.format((end - begin) * 1000))


    bench_threads = []
    for i in range(concurrency):
        t = threading.Thread(target=send_request)
        t.start()
        bench_threads.append(t)
    for t in bench_threads:
        t.join()
