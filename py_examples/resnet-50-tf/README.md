## resnet-50-tf

tensorflow版本resnet-50图片分类服务。通过自定义converter（基于opencv开发），可以实现接口层支持图片的输入，直接返回label输出。支持打开dynamic
batching模式，需要使用grps1.1.0以上版本。

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
|-- grps_framework-*-py3-none-any.whl           # grps框架依赖包，仅用于代码提示
|-- requirements.txt                            # 依赖包
|-- test.py                                     # 本地单元测试
```

```bash
# 拷贝模型相关文件
cp -r ../../cpp_examples/resnet-50-tf/data/* ./data/
```

## 2. 本地开发与调试

```bash
# 使用registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda10.1_cudnn7.6.5_tf2.3.0_torch1.8.1_py3.7镜像
docker run -it --runtime=nvidia --rm -v $(pwd):/grps_dev -w /grps_dev registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda10.1_cudnn7.6.5_tf2.3.0_torch1.8.1_py3.7 bash
# 安装依赖
apt update && apt install libgl1-mesa-glx -y
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/

# 构建
grpst archive .

# 部署
grpst start ./server.mar

# 查看部署状态，可以看到端口（HTTP,RPC）、服务名、进程ID、部署路径
grpst ps
PORT(HTTP,RPC)      NAME                PID                 DEPLOY_PATH
8020,8021           my_grps             ***                 /root/.grps/my_grps

# 模拟请求
curl -X POST -T ./data/tabby.jpeg -H "Content-Type: application/octet-stream" http://127.0.0.1:8020/grps/v1/infer/predict
'''输出结果如下：
{
 "status": {
  "code": 200,
  "msg": "OK",
  "status": "SUCCESS"
 },
 "str_data": "tabby"
}
'''

# 退出
exit
```

## 3. docker部署服务

```bash
# 构建自定义工程docker镜像
# 注意可以修改Dockerfile中的基础镜像版本，选择自己所需的版本号，默认为grps_gpu:base镜像
docker build -t resnet50_tf_online:1.0.0 -f docker/Dockerfile .

# 启动docker容器
docker run -itd --runtime=nvidia --name="resnet50_tf_online" -p 8020:8020 -p 8021:8021 resnet50_tf_online:1.0.0

# 查看日志
docker logs -f resnet50_tf_online
```

## 4. 客户端请求

### 4.1 curl客户端

```bash
curl -X POST -T ./data/tabby.jpeg -H "Content-Type: application/octet-stream" http://127.0.0.1:8020/grps/v1/infer/predict
'''输出结果如下：
{
 "status": {
  "code": 200,
  "msg": "OK",
  "status": "SUCCESS"
 },
 "str_data": "tabby"
}
'''
```

### 4.2 python客户端

[下载grps_apis pip依赖](https://github.com/NetEase-Media/grps/blob/master/apis/grps_apis/python_gens)

```bash
# http python client，使用http端口
pip3 install requests
python3 client/python/http_client.py http://0.0.0.0:8020 ./data/tabby.jpeg
'''输出结果如下：
{'status': {'code': 200, 'msg': 'OK', 'status': 'SUCCESS'}, 'str_data': 'tabby'}
'''

# grpc python client，使用rpc端口
# 下载并安装grps_apis依赖
pip3 install grps_apis-1.1.0-py3-none-any.whl
python3 client/python/grpc_client.py 0.0.0.0:8021 ./data/tabby.jpeg
'''输出结果如下：
status {
  code: 200
  msg: "OK"
}
str_data: "tabby"
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
./build/RelWithDebInfo_install/bin/grpc_client --server=0.0.0.0:8021 --img_path=../../data/tabby.jpeg
'''输出结果如下：
I20231220 23:23:04.502463 44219 grpc_client.cc:52] Predict label: tabby, latency: 34881 us
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
java -classpath target/*:maven-lib/* com.netease.GrpsClient 127.0.0.1:8021 ../../data/tabby.jpeg
'''输出结果如下：
status {
  code: 200
  msg: "OK"
}
str_data: "tabby"
'''

# 清理并退出客户端容器
rm -rf target maven-lib
exit
```

## 5. 关闭docker服务

```bash
docker rm -f resnet50_tf_online
```
