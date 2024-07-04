## resnet-50-trt-batching

trt版本resnet-50图片分类服务。通过自定义converter（基于opencv开发），可以实现接口层支持图片的输入，直接返回label输出。支持打开dynamic
batching模式，需要使用grps1.1.0以上版本。

## 1. 工程结构

```text
|-- client                              # 客户端样例
|-- conf                                # 配置文件
|   |-- inference.yml                   # 推理配置
|   |-- server.yml                      # 服务配置
|-- data                                # 数据文件
|-- docker                              # docker镜像构建
|-- second_party                        # 第二方依赖
|   |-- grps-server-framework           # grps框架依赖
|-- src                                 # 自定义源码
|   |-- customized_converter.cc/.h      # 自定义前后处理转换器
|   |-- grps_server_customized.cc/.h    # 自定义库初始化
|   |-- main.cc                         # 本地单元测试
|-- third_party                         # 第三方依赖
|-- build.sh                            # 构建脚本
|-- CMakelists.txt                      # 工程构建文件
|-- .clang-format                       # 代码格式化配置文件
|-- .config                             # 工程配置文件，包含一些工程配置开关
```

## 2. 本地开发与调试

```bash
# 使用registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda11.3_cudnn8.2_trt7.2.3_py3.8镜像
docker run -it --runtime=nvidia --rm -v $(pwd):/grps_dev -w /grps_dev registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda11.3_cudnn8.2_trt7.2.3_py3.8 bash

# 下载模型并转换为trt格式
apt update && apt install libgl1-mesa-glx -y
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install onnx opencv-python -i https://pypi.mirrors.ustc.edu.cn/simple/
python3 download_and_to_trt.py

# 构建
grpst archive .

# 部署
grpst start ./server.mar

# 查看部署状态，可以看到端口（HTTP,RPC）、服务名、进程ID、部署路径
grpst ps
PORT(HTTP,RPC)      NAME                PID                 DEPLOY_PATH
8030,8031           my_grps             ***                 /root/.grps/my_grps

# 模拟请求
curl -X POST -T ./data/dog.jpg -H "Content-Type: application/octet-stream" http://127.0.0.1:8030/grps/v1/infer/predict
'''输出结果如下：
{
 "status": {
  "code": 200,
  "msg": "OK",
  "status": "SUCCESS"
 },
 "str_data": "Samoyed, Samoyede"
}
'''

# 退出
exit
```

## 3. docker部署服务

```bash
# 构建自定义工程docker镜像
# 注意可以修改Dockerfile中的基础镜像版本，选择自己所需的版本号，默认为grps_gpu:base镜像
docker build -t resnet50_trt_online:1.0.0 -f docker/Dockerfile .

# 启动docker容器
docker run -itd --runtime=nvidia --name="resnet50_trt_online" -p 8030:8030 -p 8031:8031 resnet50_trt_online:1.0.0

# 查看日志
docker logs -f resnet50_trt_online
```

## 4. 客户端请求

### 4.1 curl客户端

```bash
curl -X POST -T ./data/dog.jpg -H "Content-Type: application/octet-stream" http://127.0.0.1:8030/grps/v1/infer/predict
'''输出结果如下：
{
 "status": {
  "code": 200,
  "msg": "OK",
  "status": "SUCCESS"
 },
 "str_data": "Samoyed, Samoyede"
}
'''
```

### 4.2 python客户端

[下载grps_apis pip依赖](https://github.com/NetEase-Media/grps/blob/master/apis/grps_apis/python_gens)

```bash
# http python client，使用http端口
pip3 install requests
python3 client/python/http_client.py http://0.0.0.0:8030 ./data/dog.jpg
'''输出结果如下：
{'status': {'code': 200, 'msg': 'OK', 'status': 'SUCCESS'}, 'str_data': 'Samoyed, Samoyede'}
'''

# grpc python client，使用rpc端口
# 下载安装grps_apis依赖
pip3 install grps_apis-1.1.0-py3-none-any.whl
python3 client/python/grpc_client.py 0.0.0.0:8031 ./data/dog.jpg
'''输出结果如下：
status {
  code: 200
  msg: "OK"
}
str_data: "Samoyed, Samoyede"
'''
```

### 4.3 c++客户端

```bash
# 这里使用构建好的grps client容器环境，这里复用主机网络
docker run -it --rm -v $(pwd):/my_grps -w /my_grps --network=host registry.cn-hangzhou.aliyuncs.com/opengrps/client:1.1.0 bash

# 构建client
cd client/cpp
bash build.sh clean
bash build.sh

# 运行grpc c++ client，使用rpc端口
./build/RelWithDebInfo_install/bin/grpc_client --server=0.0.0.0:8031 --img_path=../../data/dog.jpg
'''输出结果如下：
I20231220 17:46:24.054205 16370 grpc_client.cc:52] Predict label: Samoyed, Samoyede, latency: 71906 us
'''

# 运行brpc c++ client，使用rpc端口（需要将conf/server.yml framework改为"http+brpc"并使用新的配置重启服务）
# ./build/RelWithDebInfo_install/bin/brpc_client --server=0.0.0.0:8031 --img_path=../../data/dog.jpg
'''输出结果如下：
I1220 17:49:12.433334 18215 brpc_client.cc:59] Predict label: Samoyed, Samoyede, latency: 71355 us
'''

# 清理并退出客户端容器
bash build.sh clean
exit
```

### 4.4 java客户端

```bash
# 使用构建好的grps client容器环境，可以指定复用主机网络
docker run -it --rm -v $(pwd):/my_grps -w /my_grps --network=host registry.cn-hangzhou.aliyuncs.com/opengrps/client:1.1.0 bash

# 构建client
cd client/java
mvn clean package
mvn dependency:copy-dependencies -DoutputDirectory=./maven-lib -DstripVersion=true

# 解决中文编码问题
export LC_ALL=zh_CN.UTF-8

# 运行, 使用rpc端口
java -classpath target/*:maven-lib/* com.netease.GrpsClient 127.0.0.1:8031 ../../data/dog.jpg
'''输出结果如下：
status {
  code: 200
  msg: "OK"
}
str_data: "Samoyed, Samoyede"
'''

# 退出客户端容器
rm -rf target maven-lib
exit
```

## 5. 关闭docker服务

```bash
docker rm -f resnet50_trt_online
```
