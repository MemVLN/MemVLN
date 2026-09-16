# PPU FIFO 训练

按用户提供、此前已跑通的 PPU 脚本修改。两份脚本独立运行：

| 脚本 | Nb | Dummy LR | GPU | 端口 |
|---|---|---|---|---|
| `train_ppu_fifo_nb50.sh` | 50 | on | 0–7 | 3012 |
| `train_ppu_fifo_nb100.sh` | 100 | off | 8–15 | 3013 |

```bash
bash /workspace/MemVLN/scripts/train_ppu_fifo_nb50.sh
bash /workspace/MemVLN/scripts/train_ppu_fifo_nb100.sh
```

保留旧脚本的 `python launch.py --mode accelerate`、8卡DDP、bf16混合精度、`NCCL_SOCKET_IFNAME=bond0`、`NCCL_SOCKET_FAMILY=AF_INET`、`NCCL_DEBUG=INFO`、offline变量、文件检查、nohup、PID与日志输出。Full模型的其他设置沿用 `configs/default.yaml`；显式设为1epoch，并添加FIFO与容量参数。

数据路径严格沿用用户实际运行脚本：`DATA_BASE=/workspace/data/leo2_data/vln`，输入检查为 `$DATA_BASE/vln/r2r/...` 和 `$DATA_BASE/vln/rxr/...`。当前配置会把该根目录解析为 `/workspace/data/leo2_data/vln/vln`；这是用户提供的目录布局，不按本地目录假设改写。

本版不再提供 `--dry-run` 或额外启动前检查；直接执行即按旧脚本后台启动。若需查看命令，直接打开脚本。先前版本的 `PPU_GPUS` 等环境变量覆盖接口也已移除，设备和端口直接写在各自脚本中。

输出与W&B目录仍位于 `/workspace/outputs/`，日志在 `/workspace/outputs/logs/`。不同实验使用带容量、dummy模式及时间戳的run名称。

Nb100 FIFO-off对照已有一epoch dummy-off Full reproduce；Nb50 FIFO-on对照匹配的原Nb50 FRME。两个容量的dummy模式不同，分别解释，不直接组成纯容量消融。

用户报告上一版Nb50在sampler首次all_gather时出现 `Bootstrap : no socket interface found`，未进入训练。此次恢复完整旧环境配置；已完成本地shell语法与配置解析验证，尚未在PPU验证重跑成功。
