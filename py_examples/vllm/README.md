# vllm

使用```Vllm LLMEngine Api```实现vllm推理后端，采用自定义http格式进行服务部署。

## 1. 工程结构

```text
|-- client                                      # 客户端样例
|-- conf                                        # 配置文件
|   |-- inference.yml                           # 推理配置
|   |-- server.yml                              # 服务配置
|-- data                                        # 数据文件
|-- docker                                      # docker镜像构建
|-- src                                         # 自定义源码
|   |-- customized_inferer.py                   # 自定义推理器
|-- grps_framework-*-py3-none-any.whl           # grps框架依赖包，仅用于代码提示
|-- requirements.txt                            # 依赖包
|-- test.py                                     # 本地单元测试
```

## 2. 本地开发与调试

```bash
# 使用registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda11.8_cudnn8.6_vllm0.4.3_py3.10镜像
docker run -it --runtime=nvidia --rm -v $(pwd):/grps_dev -w /grps_dev registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda11.8_cudnn8.6_vllm0.4.3_py3.10 bash

# 修改conf/inference.yml中的vllm模型参数，如使用THUDM/chatglm3-6b模型
inferer_args: # more args of model inferer.
  model: "THUDM/chatglm3-6b"
  trust_remote_code: true
  dtype: float16
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
  device: auto

# 构建
grpst archive .

# 部署
grpst start ./server.mar

# 查看部署状态，可以看到端口（HTTP,RPC）、服务名、进程ID、部署路径
grpst ps
PORT(HTTP,RPC)      NAME                PID                 DEPLOY_PATH
7080                my_grps             ***                 /root/.grps/my_grps

# 非流式请求
curl -X POST -H "Content-Type:application/json" -d '{"prompt": "华盛顿是谁? ", "temperature": 0.1, "top_p": 0.5, "max_tokens": 4096}' 'http://127.0.0.1:7080/generate'
# 输出结果如下：
# 华盛顿(George Washington)是美国的一位政治家和将军,也是美国独立战争期间的指挥官,还是美国第一任总统。他出生于1732年,逝世于1799年。他被认为是美国历史上最伟大的领袖之一,因为他在美国革命战争期间的英勇行为和后来的政治领导能力。他建立了许多重要的政治制度和政策,对美国的发展和进步做出了巨大贡献。

# 流式请求
curl --no-buffer -X POST -H "Content-Type:application/json" -d '{"prompt": "华盛顿是谁? ", "temperature": 0.1, "top_p": 0.5, "max_tokens": 4096}' 'http://127.0.0.1:7080/generate?streaming=true'
# 输出结果如下：
# 华盛顿(George Washington)是美国的一位政治家和将军,也是美国独立战争期间的指挥官,还是美国第一任总统。他出生于1732年,逝世于1799年。他被认为是美国历史上最伟大的领袖之一,因为他在美国革命战争期间的英勇行为和后来的政治领导能力。他建立了许多重要的政治制度和政策,对美国的发展和进步做出了巨大贡献。

# 退出
exit
```

## 3. docker部署服务

```bash
# 构建自定义工程docker镜像
# 注意可以修改Dockerfile中的基础镜像版本，选择自己所需的版本号，默认为grps_gpu:base镜像
docker build -t vllm_online:1.0.0 -f docker/Dockerfile .

# 启动docker容器
docker run -itd --runtime=nvidia --name="vllm_online" -p 7080:7080 vllm_online:1.0.0

# 查看日志
docker logs -f vllm_online
```

## 4. 客户端请求

### 4.1 curl客户端

```bash
# 非流式请求
curl -X POST -H "Content-Type:application/json" -d '{"prompt": "华盛顿是谁? ", "temperature": 0.1, "top_p": 0.5, "max_tokens": 4096}' 'http://127.0.0.1:7080/generate'
# 输出结果如下：
# 华盛顿(George Washington)是美国的一位政治家和将军,也是美国独立战争期间的指挥官,还是美国第一任总统。他出生于1732年,逝世于1799年。他被认为是美国历史上最伟大的领袖之一,因为他在美国革命战争期间的英勇行为和后来的政治领导能力。他建立了许多重要的政治制度和政策,对美国的发展和进步做出了巨大贡献。

# 流式请求
curl --no-buffer -X POST -H "Content-Type:application/json" -d '{"prompt": "华盛顿是谁? ", "temperature": 0.1, "top_p": 0.5, "max_tokens": 4096}' 'http://127.0.0.1:7080/generate?streaming=true'
# 输出结果如下：
# 华盛顿(George Washington)是美国的一位政治家和将军,也是美国独立战争期间的指挥官,还是美国第一任总统。他出生于1732年,逝世于1799年。他被认为是美国历史上最伟大的领袖之一,因为他在美国革命战争期间的英勇行为和后来的政治领导能力。他建立了许多重要的政治制度和政策,对美国的发展和进步做出了巨大贡献。
```

## 5. 服务指标监控

登录http://ip:port/即可查看指标监控，在vllm后端中增加了tp的监控，如下：<br>
![tp_monitor.png](data/tp_monitor.png)<br>

以及grps内置GPU监控：<br>
![gpu_monitor.png](data/gpu_monitor.png)

## 6. 关闭docker服务

```bash
docker rm -f vllm_online
```