# PPU：联合训练对照与 R2R-only 消融

这九份脚本都可单独复制运行，不依赖额外 shell helper。沿用此前跑通的 `nohup env ... python launch.py`、`NCCL_SOCKET_IFNAME=bond0`、`AF_INET` 和 PPU 路径。使用当前仓库已有的 task/model configs，没有新增模型或 sampler 实现。

## 共同配方

- Qwen3-VL-4B-Instruct 初始化，`pretrained_ckpt_path` 为空；不从已有 Full 续训。
- 默认 **1 epoch、dummy LR off、seed42、LR 2e-5**；8 卡、每卡 batch1、accumulation1。
- 使用**旧 episode sampler**，`sampling.mode=epoch`、`ratio_unit=sample`、ratio 全1；所有模型保留 episode 顺序，包括 Baseline。
- `training.max_updates=null`，沿用 epoch 模式的 warmup400。不是新 p50 采样实验，也没有给 R2R-only 设置172330 updates。
- 最近16帧，关闭 stop augmentation。Baseline仍保留Qwen视觉塔与最近帧上下文。
- 含 memory 的版本默认 **FIFO、Nb100、每帧100 tokens、readout300 tokens**；NAV视觉mask0，Full的summary视觉mask1。
- 每5000步保存，保留最近5个checkpoint，另按现有trainer行为保存。各run使用独立时间戳和PID命名，日志在 `/workspace/outputs/logs/`。

这里的“同1 epoch”表示各自所选数据的一轮训练；Full多出summary样本，实际optimizer updates和LR结束位置会与nav-only不同。记录启动日志中的 `Scheduler Init`，不要将该表称为严格等updates对照。R2R-only先采用这个较快、清晰的单域组件研究设置。

## 脚本清单

G = geometry；M = episodic memory；S = trajectory summarization。

| 训练数据 | 模型 | 脚本 | 默认GPU / 端口 |
|---|---|---|---|
| R2R+RxR | Baseline，无G/M/S | `train_ppu_joint_baseline.sh` | 8–15 / 3013 |
| R2R+RxR | Baseline+G，无M/S | `train_ppu_joint_geometry.sh` | 0–7 / 3012 |
| R2R+RxR | Baseline+M，无G/S | `train_ppu_joint_memory.sh` | 0–7 / 3012 |
| R2R+RxR | Baseline+G+M，无S | `train_ppu_joint_geometry_memory.sh` | 8–15 / 3013 |
| R2R-only | Baseline，无G/M/S | `train_ppu_r2r_baseline.sh` | 8–15 / 3013 |
| R2R-only | Baseline+G，无M/S | `train_ppu_r2r_geometry.sh` | 8–15 / 3013 |
| R2R-only | Baseline+M，无G/S | `train_ppu_r2r_memory.sh` | 8–15 / 3013 |
| R2R-only | Baseline+G+M，无S | `train_ppu_r2r_geometry_memory.sh` | 8–15 / 3013 |
| R2R-only | Full，G+M+S | `train_ppu_r2r_full.sh` | 8–15 / 3013 |

联合训练Full/FIFO100/off/1epoch已有结果，本批不重复提供其训练。R2R-only的Full需要新训：不能把已有联合训练Full作为单域组件表的最后一行。R2R-only的五行都应使用相同epoch、seed、采样设置，含memory的三行也应使用相同consolidation。

## 启动

将所需 `.sh` 复制到 PPU 的 `/workspace/MemVLN/scripts/`。如使用压缩包，可在 `/workspace/MemVLN` 解压，其内部路径是 `scripts/...`。每份脚本独立运行，不需要chmod。

仍建议先补联合训练中最可复用的两项，分别占两组8卡：

```bash
cd /workspace/MemVLN
bash scripts/train_ppu_joint_geometry.sh
bash scripts/train_ppu_joint_baseline.sh
```

它们的默认GPU和端口互不冲突。待一组卡空闲后，可启动R2R-only Full，先获得单域Full参照：

```bash
# 仅在8–15这组卡的上一项训练结束后执行。
bash scripts/train_ppu_r2r_full.sh
```

R2R-only其余四项按清单逐个运行即可；Full、Geometry-only、Geometry+Memory先形成memory/summary比较，再补Baseline与Memory-only。联合训练后续仍优先Geometry+Memory FIFO，再补Memory-only FIFO。按实际吞吐安排，脚本准备好不代表九项能保证在截止前完成。

**每份脚本都会立即放到后台并返回PID。返回提示不等于训练完成。所有R2R-only脚本默认复用8–15和3013，不能连续执行五条命令或用 `&&` 当作自动排队。** 请确认对应训练退出后再启动下一项；本批没有自动队列或自动占卡。

## 改卡号、策略或epoch

若空闲的是0–7，或独立8卡容器只枚举0–7：

```bash
GPU_IDS=0,1,2,3,4,5,6,7 DIST_PORT=3012 bash scripts/train_ppu_r2r_full.sh
```

将某一memory模型改为FRME：

```bash
CONSOLIDATION=frme bash scripts/train_ppu_r2r_full.sh
```

这个开关也支持 `*_memory.sh` 与 `*_geometry_memory.sh`。如果整套单域ablation采用FRME，M、GM、Full三项全部加 `CONSOLIDATION=frme`；相同预算的B/G可以复用，不含consolidation。

若之后要补对应FRME两epoch的联合训练消融，例如：

```bash
EPOCHS=2 CONSOLIDATION=frme bash scripts/train_ppu_joint_geometry_memory.sh
```

`EPOCHS=2`也适用于Baseline、Geometry-only与Memory-only脚本，但要分别从预训练初始化重训。同预算比较需要对应B2/G2/M2/GM2；不能把1epoch结果直接填进2epoch表，或把2epoch训练的中间checkpoint当作独立1epoch训练。

所有路径可通过同名环境变量覆盖：`CODE_DIR`、`QWEN3_PATH`、带geometry时的`STREAMVGGT_PATH`、`DATA_BASE`、`OUTPUT_BASE`；另支持`GPU_IDS`、`DIST_PORT`、`EPOCHS`、`SEED`。当前Python launcher会再次拼接shell文本，因此保持原PPU路径的简单格式，不使用含空格或shell特殊字符的路径。

默认 `DATA_BASE=/workspace/data/leo2_data/vln`，任务解析的annotation目录为 `$DATA_BASE/vln`，原样保留此前已跑通的路径。无geometry的脚本不要求StreamVGGT checkpoint；R2R-only不要求RxR annotation存在。

## 验证与结果记录

本地验证通过46个测试方法、73次shell脚本调用：9个默认组合和5个memory组合的FRME/2epochs/换卡与端口设置都经过实际launcher参数传递及Hydra合成；其余检查覆盖非法参数、缺失权重等提前退出路径。不加载真实模型、数据，不启动PPU训练。真实设备/数据与训练吞吐以PPU运行日志为准。

每个结果记录训练域、模型行、consolidation、epoch、dummy设置、seed、实际updates、checkpoint和完整评测范围。建议先统一比较最后一个checkpoint，邻近checkpoint只作稳定性诊断；R2R-only表与joint表分别展示。
