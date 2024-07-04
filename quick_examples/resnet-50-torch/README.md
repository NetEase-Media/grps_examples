# resnet-50-torch

torch版本resnet-50图片分类服务，通过零编码快速部署方式对外提供模型服务。

## 1. docker快速部署

```bash
# 拷贝模型相关文件
cp -r ../../cpp_examples/resnet-50-torch/data/* ./data/

# docker可以跟上--gpus='"device=num"'参数指定使用部分gpu，如--gpus='"device=0,1"'
# 可以通过--port参数指定端口号，这里使用默认端口7080,7081(http,grpc)，更多参数见grpst torch_serve -h
docker run -itd --runtime=nvidia --name="resnet50_torch_online" \
-v $(pwd):/my_grps -w /my_grps \
-p 7080:7080 -p 7081:7081 \
registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda10.1_cudnn7.6.5_tf2.3.0_torch1.8.1_py3.7 \
grpst torch_serve ./data/resnet50_pretrained.pt

# 查看日志
docker logs -f resnet50_torch_online
```

## 2. client请求

client中将图片解析为tensor，以tensor格式请求服务，将请求结果转换为label。<br>
[下载grps_apis pip依赖](https://github.com/NetEase-Media/grps/blob/master/apis/grps_apis/python_gens)

```bash
# 安装依赖
apt update && apt install libgl1-mesa-glx -y
pip3 install opencv-python
pip3 install requests
pip3 install grps_apis-1.1.0-py3-none-any.whl

# http client访问
python3 http_client.py 127.0.0.1:7080 ./data/dog.jpg
# 输出如下：
# predict time: 251.54638290405273 ms
# Samoyed

# grpc client访问，使用grpc访问更高效，主要因为传递tensors使用了protobuf序列化
python3 grpc_client.py 127.0.0.1:7081 ./data/dog.jpg
# 输出如下：
# predict time: 34.65390205383301 ms
# Samoyed
```

## 3. 关闭docker服务

```bash
docker rm -f resnet50_torch_online
```
