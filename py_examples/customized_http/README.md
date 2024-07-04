# grps py template

自定义http请求和返回的样例，使用自定义json格式实现一个"c = a + b"的功能。

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
8010                my_grps             ***                 /root/.grps/my_grps

# 模拟请求
curl -X POST -H "Content-Type:application/json" -d '{"a": 1, "b": 2}' http://127.0.0.1:8010/custom_predict
# 输出结果如下：
# {"c":3.0}

# 退出
exit
```

## 3. docker部署服务

```bash
# 构建自定义工程docker镜像
# 注意可以修改Dockerfile中的基础镜像版本，选择自己所需的版本号，默认为grps_gpu:base镜像
docker build -t custom_http_online:1.0.0 -f docker/Dockerfile .

# 启动docker容器
docker run -itd --runtime=nvidia --name="custom_http_online" -p 8010:8010 custom_http_online:1.0.0

# 查看日志
docker logs -f custom_http_online
```

## 4. 客户端请求

### 4.1 curl客户端

```bash
curl -X POST -H "Content-Type:application/json" -d '{"a": 1, "b": 2}' http://127.0.0.1:8010/custom_predict
# 输出结果如下：
# {"c": 3.0}
```

## 5. 关闭docker服务

```bash
docker rm -f custom_http_online
```
