# DGX Spark 多节点推理部署

通过光缆 (ConnectX-7) 连接多台 DGX Spark，使用 SGLang Tensor Parallelism 实现跨节点大模型推理。

## 硬件要求

- 2+ 台 NVIDIA DGX Spark (GB10, 128GB 显存)
- 光缆连接 ConnectX-7 网口 (每台 4 口, 最高 200Gbps/口)
- 各节点可通过管理网 SSH 免密互通

## 快速开始

### 1. 配置节点

编辑 `.env`，修改 `NODE_LIST` 为你的机器信息：

```bash
NODE_LIST=(
    "管理网IP,高速网IP,SSH用户名,主机名"
    "192.168.130.15,10.10.10.1,ai,spark-ccf1"    # 节点1 (HEAD)
    "192.168.130.16,10.10.10.2,ai,spark-bac4"    # 节点2 (WORKER)
)
```

### 2. 配置高速网络

插上光缆后，一键配置所有节点的 ConnectX-7 网卡 IP 和 MTU：

```bash
bash deploy.sh network
# 输入 sudo 密码，自动配置所有节点
```

### 3. 测试连接

```bash
# 在节点1上运行
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
curl http://HEAD节点IP:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/model","messages":[{"role":"user","content":"你好"}]}'
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

```bash
NODE_LIST=(
    "管理网IP,高速网IP,SSH用户名,主机名"
)
```

| 字段 | 说明 | 示例 |
|------|------|------|
| 管理网IP | SSH 连接用的 IP | `192.168.130.15` |
| 高速网IP | ConnectX-7 光缆互联 IP | `10.10.10.1` |
| SSH用户名 | 免密 SSH 的用户 | `ai` |
| 主机名 | 可选，用于日志显示 | `spark-ccf1` |

- 第一个节点自动作为 HEAD (rank 0)
- 添加行即可增加 WORKER 节点
- `TP_SIZE` 需等于总节点数

### 换机器

只需修改 `NODE_LIST` 中的 IP：

```bash
NODE_LIST=(
    "192.168.1.100,10.10.10.1,ai,new-spark-1"
    "192.168.1.101,10.10.10.2,ai,new-spark-2"
)
```

然后重新执行：

```bash
bash deploy.sh network     # 配置新机器网络
bash deploy.sh restart     # 重启服务
```

### 换模型

```bash
MODEL_PATH=/home/ai/models/DeepSeek-V3
SGLANG_IMAGE=nvcr.io/nvidia/sglang:26.03-py3
TP_SIZE=2
QUANTIZATION=                # 留空=bf16, 或 modelopt_fp4
EXTRA_ARGS=--trust-remote-code
```

### 网络参数

```bash
FAST_IFACE=enp1s0f0np0       # ConnectX-7 接口名
FAST_MTU=9000                # Jumbo Frame
RDMA_HCA=rocep1s0f0          # RDMA 设备名
```

查看可用接口: `ip -br link show`
查看 RDMA 设备: `ibv_devinfo`

## 文件结构

```
dgx-spark-multinode/
├── .env                  # 配置文件 (机器、模型、网络)
├── deploy.sh             # 管理脚本 (start/stop/status/...)
├── setup-network.sh      # 网络配置 (由 deploy.sh network 调用)
├── test-connection.sh    # 连接测试
├── docker-compose.node1.yml   # HEAD compose (自动生成)
├── docker-compose.node2.yml   # WORKER compose (自动生成)
└── README.md
```

## 注意事项

- 所有节点的模型路径 (`MODEL_PATH`) 必须一致，模型文件需提前拷贝到每台机器
- 节点间需免密 SSH，可用 `ssh-copy-id` 配置
- 当前 NCCL 通过 TCP Socket 通信；未来可配置 RDMA (RoCE) 以获得更高带宽
- Qwen3.5-35B-A3B 等单机可装下的模型，多节点不会更快（通信开销）；多节点的价值在于跑 128GB 以上的大模型
- `docker-compose.nodeN.yml` 在 `deploy.sh start` 时自动生成，手动修改会被覆盖

## 前置准备

### SSH 免密

```bash
# 在控制机上，对每台 DGX Spark 执行
ssh-copy-id ai@192.168.130.15
ssh-copy-id ai@192.168.130.16

# DGX Spark 之间也需要互通
ssh ai@192.168.130.15 "ssh-copy-id ai@192.168.130.16"
ssh ai@192.168.130.16 "ssh-copy-id ai@192.168.130.15"
```

### 光缆连接

DGX Spark 有 4 个 ConnectX-7 网口（2 组各 2 口），默认使用第一个口 `enp1s0f0np0`。
用 QSFP112 光缆/DAC 铜缆连接两台机器的对应网口即可。
