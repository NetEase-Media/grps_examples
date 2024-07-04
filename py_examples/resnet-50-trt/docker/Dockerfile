# copyright : netease news pctr team
# author    : zhaochaochao@corp.netease.com
# date      : 2024-01-12
# brief     : Build grps customized project image.

# --------Building stage.--------
FROM registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda11.3_cudnn8.2_trt7.2.3_py3.8 AS build

# grps archive.
RUN mkdir -p /my_grps
ADD conf /my_grps/conf
ADD data /my_grps/data
ADD src /my_grps/src
ADD requirements.txt /my_grps/requirements.txt
ADD test.py /my_grps/test.py
RUN cd /my_grps && \
    grpst archive . --skip_unittest --output_path server.mar

# --------Release stage.--------
FROM registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda11.3_cudnn8.2_trt7.2.3_py3.8

RUN apt update && apt install libgl1-mesa-glx -y

# Intall requirements.
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/

WORKDIR /my_grps
COPY --from=build /my_grps/server.mar /my_grps/server.mar
ENV LANG C.UTF-8
CMD ["/bin/sh", "-c", "grpst start server.mar"]
