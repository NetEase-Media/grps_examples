# rnn-torch

简易的rnn文字生成模型部署样例（仅是样例，输出结果没有实际含义），支持streaming流式推理，输入提示词（固定为2个单词）获得推理结果。converter设定为none，使用no
converter模式，自定义inferer。

## 1. 工程结构

```text
|-- client                                      # 客户端样例
|-- conf                                        # 配置文件
|   |-- inference.yml                           # 推理配置
|   |-- server.yml                              # 服务配置
|-- data                                        # 数据文件
|-- docker                                      # docker镜像构建
|-- src                                         # 自定义源码
|   |-- customized_converter.py                 # 自定义前后处理转换器
|   |-- customized_inferer.py                   # 自定义推理器
|-- grps_framework-*-py3-none-any.whl           # grps框架依赖包，仅用于代码提示
|-- requirements.txt                            # 依赖包
|-- test.py                                     # 本地单元测试
```

## 2. 本地开发与调试

```bash
# 使用registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda10.1_cudnn7.6.5_tf2.3.0_torch1.8.1_py3.7镜像
docker run -it --runtime=nvidia --rm -v $(pwd):/grps_dev -w /grps_dev registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda10.1_cudnn7.6.5_tf2.3.0_torch1.8.1_py3.7 bash

# 构建
grpst archive .

# 部署
grpst start ./server.mar

# 查看部署状态，可以看到端口（HTTP,RPC）、服务名、进程ID、部署路径
grpst ps
PORT(HTTP,RPC)      NAME                PID                 DEPLOY_PATH
7080,7081           my_grps             ***                 /root/.grps/my_grps

# 模拟请求
curl -X POST -H "Content-Type:application/json" -d '{"str_data": "this process"}' http://0.0.0.0:7080/grps/v1/infer/predict
'''输出结果如下：
{
 "status": {
  "code": 200,
  "msg": "OK",
  "status": "SUCCESS"
 },
 "str_data": "this process gave physiology mountainous"
}
'''

# 退出
exit
```

## 3. docker部署服务

```bash
# 构建自定义工程docker镜像
# 注意可以修改Dockerfile中的基础镜像版本，选择自己所需的版本号，默认为grps_gpu:base镜像
docker build -t rnn_torch_online:1.0.0 -f docker/Dockerfile .

# 启动docker容器
docker run -itd --runtime=nvidia --name rnn_torch_online -p 7080:7080 -p 7081:7081 rnn_torch_online:1.0.0

# 查看日志
docker logs -f rnn_torch_online
```

## 4. 客户端请求

### 4.1 curl客户端

```bash
# curl命令请求，使用http端口
curl -X POST -H "Content-Type:application/json" -d '{"str_data": "this process"}' http://0.0.0.0:7080/grps/v1/infer/predict
'''输出结果如下：
{
 "status": {
  "code": 200,
  "msg": "OK",
  "status": "SUCCESS"
 },
 "str_data": "this process light breath reflection"
'''
# 带上"streaming:true" query-param，使用streaming模式请求
curl --no-buffer -X POST -H "Content-Type:application/json" -d '{"str_data": "this process"}' 'http://0.0.0.0:7080/grps/v1/infer/predict?streaming=true'
'''输出结果如下：
{
  "status": {
    "code": 200,
    "msg": "OK",
    "status": "SUCCESS"
  },
  "str_data": "this process"
}{
  "status": {
    "code": 200,
    "msg": "OK",
    "status": "SUCCESS"
  },
  "str_data": "who"
}{
  "status": {
    "code": 200,
    "msg": "OK",
    "status": "SUCCESS"
  },
  "str_data": "mein"
}{
  "status": {
    "code": 200,
    "msg": "OK",
    "status": "SUCCESS"
  },
  "str_data": "artist"
}
'''
```

### 4.2 python客户端

[下载grps_apis pip依赖](https://github.com/NetEase-Media/grps/blob/master/apis/grps_apis/python_gens)

```bash
# http python client，使用http端口
pip3 install requests
python3 client/python/http_client.py 0.0.0.0:7080
'''输出结果如下：
b'{\n  "status": {\n    "code": 200,\n    "msg": "OK",\n    "status": "SUCCESS"\n  },\n  "str_data": "this process"\n}'
b'{\n  "status": {\n    "code": 200,\n    "msg": "OK",\n    "status": "SUCCESS"\n  },\n  "str_data": "say"\n}'
b'{\n  "status": {\n    "code": 200,\n    "msg": "OK",\n    "status": "SUCCESS"\n  },\n  "str_data": "windward"\n}'
b'{\n  "status": {\n    "code": 200,\n    "msg": "OK",\n    "status": "SUCCESS"\n  },\n  "str_data": "afforded"\n}'
'''

# grpc python client，使用rpc端口
# 下载并安装grps_apis依赖
pip3 install grps_apis-1.1.0-py3-none-any.whl
python3 client/python/grpc_client.py 0.0.0.0:7081
'''输出结果如下：
status {
  code: 200
  msg: "OK"
}
str_data: "this process"

status {
  code: 200
  msg: "OK"
}
str_data: "everlasting"

status {
  code: 200
  msg: "OK"
}
str_data: "notice"

status {
  code: 200
  msg: "OK"
}
str_data: "afforded"
'''
```

### 4.3 c++客户端

```bash
# 这里使用已构建好的grps client容器环境，这里指定复用主机网络
docker run -it --rm -v $(pwd):/my_grps -w /my_grps --network=host registry.cn-hangzhou.aliyuncs.com/opengrps/client:1.1.0 bash

# 构建client
cd client/cpp
bash build.sh clean
bash build.sh

# 运行grpc c++ client，使用rpc端口
./build/RelWithDebInfo_install/bin/grpc_client --server=0.0.0.0:7081
'''输出结果如下：
I20231220 23:38:28.406960 45549 grpc_client.cc:40] Predict streaming response: status {
  code: 200
  msg: "OK"
}
str_data: "this process"
I20231220 23:38:28.425172 45549 grpc_client.cc:40] Predict streaming response: status {
  code: 200
  msg: "OK"
}
str_data: "forgotten"
I20231220 23:38:28.426054 45549 grpc_client.cc:40] Predict streaming response: status {
  code: 200
  msg: "OK"
}
str_data: "borne"
I20231220 23:38:28.426964 45549 grpc_client.cc:40] Predict streaming response: status {
  code: 200
  msg: "OK"
}
str_data: "meantime"
I20231220 23:38:28.427619 45549 grpc_client.cc:48] Predict streaming latency: 26898 us
'''

# 清理并退出客户端容器
bash build.sh clean
exit
```

## 5. 关闭docker部署

```bash
docker rm -f rnn_torch_online
```
