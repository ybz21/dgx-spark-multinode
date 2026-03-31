# DGX Spark 多节点推理部署

通过光缆 (ConnectX-7) 连接多台 NVIDIA DGX Spark，使用 SGLang Tensor Parallelism 实现跨节点大模型推理。

## 硬件要求

- 2+ 台 NVIDIA DGX Spark (GB10, 128GB 显存)
- 光缆连接 ConnectX-7 网口 (每台 4 口, 最高 200Gbps/口)
- 各节点可通过管理网 SSH 免密互通

## 快速开始

```bash
# 1. 配置
cp .env.example .env
vim .env                     # 修改 NODE_LIST 和 MODEL_PATH

# 2. 预检 (验证 SSH/Docker/模型/网络)
./deploy.sh check

# 3. 配置光缆网络 (首次)
./deploy.sh network

# 4. 启动推理
./deploy.sh start

# 5. 测试
./deploy.sh test
```

## 命令参考

```
./deploy.sh <命令> [选项]

命令:
  start            启动多节点推理
  stop    [-y]     停止推理 (-y 跳过确认)
  restart [-y]     重启推理
  status           查看状态和健康检查
  logs    [N] [-f] 查看节点 N 的日志 (-f 实时跟踪)
  test             发送推理测试请求
  check            预检: 验证 SSH/Docker/模型/网络
  network          配置高速网络 (需 sudo 密码)
  test-net         测试高速网络连通性和带宽
  pull             在所有节点上拉取 Docker 镜像
  sync             同步配置文件到所有节点

选项:
  -h, --help       显示帮助
  -v, --version    显示版本号
  -y, --yes        跳过确认提示
  -f, --follow     实时跟踪日志
  --dry-run        仅显示将要执行的操作
```

## .env 配置

### 节点列表

```bash
# 格式: 管理网IP,高速网IP,SSH用户名,主机名[,光口接口名]
NODE_LIST=(
    "192.168.1.100,10.10.10.1,ai,spark-node1"                # 光口自动检测
    "192.168.1.101,10.10.10.2,ai,spark-node2,enp1s0f1np1"    # 指定光口
)
```

| 字段 | 必填 | 说明 |
|------|:----:|------|
| 管理网IP | 是 | SSH 连接用的 IP |
| 高速网IP | 是 | 光缆互联 IP (自行分配) |
| SSH用户名 | 是 | 免密 SSH 用户 |
| 主机名 | 是 | 日志显示用 |
| 光口接口名 | 否 | 省略则自动检测有光缆的口 |

### 模型和推理

```bash
MODEL_PATH=/home/ai/models/YOUR_MODEL     # 每台机器上的模型路径 (需一致)
SGLANG_IMAGE=nvcr.io/nvidia/sglang:26.03-py3
TP_SIZE=2                                  # = 节点数
QUANTIZATION=                              # 留空=bf16, modelopt_fp4
EXTRA_ARGS="--trust-remote-code"
```

### 自定义 Patch (可选)

```bash
CUSTOM_SCRIPTS_DIR=/path/to/scripts
CUSTOM_REASONING_DETECTOR=my_detector.py
CUSTOM_ENTRYPOINT=my_entrypoint.sh
```

## 光口自动检测

DGX Spark 每台有 4 个 ConnectX-7 网口:

```
enp1s0f0np0   enp1s0f1np1      # 第 1 组
enP2p1s0f0np0 enP2p1s0f1np1    # 第 2 组
```

**光缆可以插在任意口上，两台机器不需要插同一编号的口。**

脚本自动处理:
1. 扫描所有 ConnectX-7 口，找到 `carrier=1` (有光缆) 的口
2. 推导对应的 RDMA HCA 名 (`enp1s0f1np1` → `rocep1s0f1`)
3. 生成正确的 NCCL/RDMA 配置

手动查看光口状态:

```bash
for dev in enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1; do
  echo "$dev: carrier=$(cat /sys/class/net/$dev/carrier 2>/dev/null || echo '?')"
done
```

## 文件结构

```
dgx-spark-dual/
├── deploy.sh                         # 唯一入口
├── .env.example                      # 配置模板
├── .env                              # 实际配置 (gitignore)
├── scripts/
│   ├── common.sh                     # 共享函数库
│   ├── setup-network.sh              # 网络配置 (deploy.sh network 调用)
│   └── test-connection.sh            # 连接测试 (deploy.sh test-net 调用)
├── docker-compose.node1.example.yml  # HEAD 节点示例
├── docker-compose.node2.example.yml  # WORKER 节点示例
└── README.md
```

## 前置准备

### SSH 免密

```bash
ssh-copy-id user@node1-mgmt-ip
ssh-copy-id user@node2-mgmt-ip
ssh user@node1 "ssh-copy-id user@node2"
ssh user@node2 "ssh-copy-id user@node1"
```

### 光缆连接

用 QSFP112 光缆或 DAC 铜缆连接两台机器的任意 ConnectX-7 网口。
不要求两台插同一编号的口，脚本自动检测。

## 注意事项

- 模型文件需提前拷贝到每台机器的相同路径
- 单机能装下的模型多节点不会更快 (通信开销)，多节点用于跑 >128GB 的大模型
- 首次使用先运行 `./deploy.sh check` 确认环境正常
