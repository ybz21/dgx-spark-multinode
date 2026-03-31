# DGX Spark 多节点推理部署

通过光缆 (ConnectX-7) 连接多台 NVIDIA DGX Spark，使用 SGLang Tensor Parallelism 实现跨节点大模型推理。

## 硬件要求

- 2+ 台 NVIDIA DGX Spark (GB10, 128GB 显存)
- 光缆连接 ConnectX-7 网口 (每台 4 口, 最高 200Gbps/口)
- 各节点可通过管理网 SSH 免密互通

## 快速开始

### 1. 配置

```bash
cp .env.example .env
vim .env   # 修改 NODE_LIST、MODEL_PATH 等
```

`NODE_LIST` 格式：`管理网IP,高速网IP,SSH用户名,主机名[,光口接口名]`

```bash
NODE_LIST=(
    "192.168.1.100,10.10.10.1,ai,spark-node1"       # 光口自动检测
    "192.168.1.101,10.10.10.2,ai,spark-node2"       # 光口自动检测
)
```

### 2. 配置高速网络

插上光缆后，一键配置所有节点的 ConnectX-7 网卡 IP 和 MTU：

```bash
bash deploy.sh network
# 输入 sudo 密码，自动检测光口并配置所有节点
```

### 3. 测试连接

```bash
# 在节点 1 上运行
bash test-connection.sh 1 2
```

预期结果：
- Ping 延迟 < 1ms
- Jumbo Frame (MTU 9000) 通过
- `iperf3` TCP ~40 Gbps
- `ib_write_bw` RDMA ~100 Gbps

### 4. 启动推理

```bash
bash deploy.sh start
# 等待 2-3 分钟模型加载完成

bash deploy.sh status   # 查看状态
bash deploy.sh test     # 发送测试请求
```

API 默认监听在 HEAD 节点的 30000 端口，兼容 OpenAI 格式：

```bash
curl http://<HEAD节点IP>:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/model","messages":[{"role":"user","content":"你好"}]}'
```

## 光口自动检测

DGX Spark 每台有 4 个 ConnectX-7 网口：

```
enp1s0f0np0   enp1s0f1np1      # 第 1 组 (2 口)
enP2p1s0f0np0 enP2p1s0f1np1    # 第 2 组 (2 口)
```

**光缆可以插在任意口上，两台机器不需要插同一编号的口。** 脚本会：

1. 自动扫描所有 ConnectX-7 口，找到 `carrier=1` (有光缆) 的口
2. 自动推导对应的 RDMA HCA 名 (如 `enp1s0f1np1` -> `rocep1s0f1`)
3. 在生成的 docker-compose 中设置正确的 `NCCL_SOCKET_IFNAME` 和 `NCCL_IB_HCA`

如果你需要强制指定某个口（比如插了多根光缆），可以在 `NODE_LIST` 第 5 个字段指定：

```bash
NODE_LIST=(
    "192.168.1.100,10.10.10.1,ai,spark-node1,enp1s0f1np1"    # 强制用这个口
    "192.168.1.101,10.10.10.2,ai,spark-node2,enp1s0f0np0"    # 强制用这个口
)
```

手动查看哪个口有光缆：

```bash
for dev in enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1; do
  carrier=$(cat /sys/class/net/$dev/carrier 2>/dev/null || echo "?")
  echo "$dev: carrier=$carrier"    # 1=有光缆  0=无光缆
done
```

## 管理命令

```bash
bash deploy.sh start       # 启动多节点推理
bash deploy.sh stop        # 停止
bash deploy.sh restart     # 重启
bash deploy.sh status      # 查看状态 + 健康检查
bash deploy.sh logs [N]    # 查看节点 N 的日志 (默认 1)
bash deploy.sh test        # 发送测试请求
bash deploy.sh network     # 配置高速网络 (需 sudo)
bash deploy.sh sync        # 同步配置到所有节点
```

## .env 配置说明

### 节点配置

| 字段 | 必填 | 说明 | 示例 |
|------|------|------|------|
| 管理网IP | 是 | SSH 连接用的 IP | `192.168.1.100` |
| 高速网IP | 是 | ConnectX-7 光缆互联 IP (自行分配) | `10.10.10.1` |
| SSH用户名 | 是 | 免密 SSH 的用户 | `ai` |
| 主机名 | 是 | 用于日志显示 | `spark-node1` |
| 光口接口名 | 否 | 省略则自动检测 | `enp1s0f0np0` |

- 第一个节点自动作为 HEAD (rank 0)
- 添加行即可增加 WORKER 节点
- `TP_SIZE` 需等于总节点数

### 换机器

修改 `NODE_LIST` 中的 IP，光口不用管（自动检测）：

```bash
bash deploy.sh network     # 配置新机器网络
bash deploy.sh restart     # 重启服务
```

### 换模型

```bash
MODEL_PATH=/home/ai/models/YOUR_MODEL
SGLANG_IMAGE=nvcr.io/nvidia/sglang:26.03-py3
TP_SIZE=2
QUANTIZATION=                # 留空=bf16, 或 modelopt_fp4
EXTRA_ARGS="--trust-remote-code"
```

### 自定义 Patch (可选)

支持挂载自定义脚本到容器中，用于 patch SGLang 行为。例如自定义 reasoning detector：

```bash
CUSTOM_SCRIPTS_DIR=/home/ai/scripts
CUSTOM_REASONING_DETECTOR=my_reasoning_detector.py
CUSTOM_ENTRYPOINT=my_entrypoint.sh
```

entrypoint 脚本会在 SGLang 启动前执行，可用于替换/patch SGLang 内部文件。

## 文件结构

```
dgx-spark-dual/
├── .env.example                      # 配置模板 (复制为 .env 使用)
├── .env                              # 实际配置 (git忽略)
├── .gitignore
├── deploy.sh                         # 管理脚本 (start/stop/status/...)
├── setup-network.sh                  # 网络配置 (由 deploy.sh network 调用)
├── test-connection.sh                # 连接测试
├── docker-compose.node1.example.yml  # HEAD 节点 compose 示例
├── docker-compose.node2.example.yml  # WORKER 节点 compose 示例
└── README.md
```

运行时 `deploy.sh start` 会根据 `.env` 和自动检测的光口生成实际的 `docker-compose.node0.yml`、`docker-compose.node1.yml` 等文件，不纳入版本管理。

## 注意事项

- 所有节点的模型路径 (`MODEL_PATH`) 必须一致，模型文件需提前拷贝到每台机器
- 节点间需免密 SSH，可用 `ssh-copy-id` 配置
- 光缆随便插哪个 ConnectX-7 口都行，脚本自动检测
- 单机能装下的模型 (如 35B)，多节点不会更快 (通信开销)；多节点的价值在于跑 128GB 以上的大模型
- NCCL 通信自动选择最优传输方式 (Socket / RoCE)

## 前置准备

### SSH 免密

```bash
# 在控制机上，对每台 DGX Spark 执行
ssh-copy-id user@node1-mgmt-ip
ssh-copy-id user@node2-mgmt-ip

# DGX Spark 之间也需要互通
ssh user@node1 "ssh-copy-id user@node2"
ssh user@node2 "ssh-copy-id user@node1"
```

### 光缆连接

DGX Spark 有 4 个 ConnectX-7 网口 (2 组各 2 口)。
用 QSFP112 光缆或 DAC 铜缆连接两台机器的任意网口即可，不要求两台插同一编号的口。
