# Qwen2.5-VL-7B Baseline / Full MemVLN 单卡 4090 测试手册

本文件交给 4090 主机上的执行者或 Codex，按顺序完成测试。本轮只做
验证：不要修改代码、不要 commit/push，也不要在失败后静默改变模型合同。

## 1. 目标和判定

需要分别验证：

- Baseline：官方 Transformers 4.57 `Qwen2_5_VLForConditionalGeneration`，
  R2R+RxR navigation SFT。
- Full：同一 Qwen2.5 backbone，加 300 Memory tokens、冻结的 StreamVGGT、
  camera merger 和 trajectory-summary supervision。

固定合同：

- Transformers `4.57.0`，BF16，FlashAttention2，seed 42。
- Baseline/Full 的 LoRA 都是 rank 16、alpha 16、dropout 0，并使用相同的
  `q/k/v/o/gate/up/down_proj` 七个 language targets。
- hidden size 3584；Memory token 数 300；frame token 上限 100。
- 640×480 image 在 Qwen2.5 分支产生 391 tokens。
- StreamVGGT 保持 30×40 grid，即 420×560 geometry input。
- Full generation 只允许 greedy；Memory/VGGT 只在 prefill 更新。

每个阶段只能记为以下状态之一：

- `PASS`
- `ENV_BLOCKED`
- `DATA_BLOCKED`
- `CODE_FAIL`
- `CAPACITY_BLOCKED_24GB`

Full one-batch 如果仅因标准 CUDA allocator OOM 失败，可以记为
`CAPACITY_BLOCKED_24GB`，但不能写成 Full 全部通过。仍需在新进程中继续
完成真实权重 load 和 no-grad generation，以区分容量问题与代码问题。

## 2. Checkout、路径和日志

在 Bash 4+ 中执行。先把下面五个路径替换成 4090 主机上的真实路径；
不要照搬 Mac 路径。路径中不能含空格，因为当前 launcher 会重新拼接命令。

```bash
set -euo pipefail
test -n "${BASH_VERSION:-}"
(( BASH_VERSINFO[0] >= 4 ))

export MEMVLN_REPO=/实际路径/MemVLN
export PYTHON_BIN=/实际conda环境/bin/python
export QWEN25_VL_7B_PATH=/实际路径/Qwen2.5-VL-7B-Instruct
export STREAMVGGT_CKPT=/实际路径/StreamVGGT-1B.pth
# 指向包含 vln/ 子目录的根，而不是直接指向 vln/。
export ANNOTATION_BASE=/实际路径/leo2_data

# 输出必须位于 repo 外。
export MEMVLN_SMOKE_ROOT=/实际路径/outputs/qwen25_4090_smoke
export EXPECTED_BRANCH=codex/qwen25-baseline-full
export REQUIRED_IMPLEMENTATION_COMMIT=4d635df2168f91ff3f521efc2f69dac811b46e47

for value in \
  "$MEMVLN_REPO" "$PYTHON_BIN" "$QWEN25_VL_7B_PATH" \
  "$STREAMVGGT_CKPT" "$ANNOTATION_BASE" "$MEMVLN_SMOKE_ROOT"; do
  case "$value" in
    *" "*) echo "ERROR: path contains whitespace: $value"; exit 1 ;;
  esac
done

test -x "$PYTHON_BIN"
test -d "$MEMVLN_REPO"
test -d "$QWEN25_VL_7B_PATH"
test -f "$STREAMVGGT_CKPT"
mkdir -p "$MEMVLN_SMOKE_ROOT"

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export RUN_NOTE="qwen25_4090_$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_DIR="$MEMVLN_SMOKE_ROOT/$RUN_NOTE"
export LOG_DIR="$RUN_DIR/logs"
export EVIDENCE_DIR="$RUN_DIR/evidence"
export LOG_FILE="$LOG_DIR/00_session.log"
test ! -e "$RUN_DIR"
mkdir -p "$LOG_DIR" "$EVIDENCE_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "RUN_NOTE=$RUN_NOTE"
echo "PID=$$"
echo "LOG_FILE=$LOG_FILE"

cd "$MEMVLN_REPO"
test -z "$(git status --porcelain)"
git fetch origin "$EXPECTED_BRANCH"
if git show-ref --verify --quiet "refs/heads/$EXPECTED_BRANCH"; then
  git switch "$EXPECTED_BRANCH"
else
  git switch --track -c "$EXPECTED_BRANCH" "origin/$EXPECTED_BRANCH"
fi
git pull --ff-only origin "$EXPECTED_BRANCH"

test "$(git branch --show-current)" = "$EXPECTED_BRANCH"
git merge-base --is-ancestor "$REQUIRED_IMPLEMENTATION_COMMIT" HEAD
test -f docs/QWEN25_4090_SMOKE_TEST.md
test -f scripts/qwen25_generation_smoke.py
test -z "$(git status --porcelain)"

git rev-parse HEAD | tee "$EVIDENCE_DIR/git_commit.txt"
git status --short --branch | tee "$EVIDENCE_DIR/git_status.txt"
```

固定单卡、离线模型加载和可复现环境：

```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
```

## 3. 环境 gate

```bash
nvidia-smi -L
nvidia-smi \
  --query-gpu=name,uuid,driver_version,memory.total,memory.free \
  --format=csv | tee "$EVIDENCE_DIR/nvidia_smi.csv"
free -h | tee "$EVIDENCE_DIR/host_memory.txt"
df -h "$MEMVLN_SMOKE_ROOT" | tee "$EVIDENCE_DIR/disk.txt"
"$PYTHON_BIN" -m pip freeze > "$EVIDENCE_DIR/pip_freeze.txt"

"$PYTHON_BIN" - "$EVIDENCE_DIR/environment.json" <<'PY'
import json
import platform
import sys

import accelerate
import peft
import torch
import transformers
from flash_attn import flash_attn_func

assert transformers.__version__ == "4.57.0", transformers.__version__
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
assert "4090" in torch.cuda.get_device_name(0)

q = torch.randn(1, 16, 4, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, q, q, causal=True)
assert out.shape == q.shape
torch.cuda.synchronize()

payload = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "peft": peft.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "flashattention_smoke": "PASS",
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
print(json.dumps(payload, indent=2))
PY
```

以下任一情况记为 `ENV_BLOCKED` 并停止 GPU 阶段：Transformers 不是
4.57.0、CUDA/BF16 不可用、FlashAttention2 import/CUDA smoke 失败，或
GPU 上还有其他大显存进程。禁止静默回退到 SDPA/eager。

## 4. 权重和数据 gate

已验证的 artifact hashes：

```bash
export EXPECTED_QWEN25_MANIFEST_SHA256=41c166cc8218dcc80d6d1248cd75b9b7fa30b3592c75457f1c02168c673aa369
export EXPECTED_STREAMVGGT_SHA256=3d3d914d09609fdb485e16947c4885cca0e5ea4e5557598a3b3ffb1cfe0a2f59

"$PYTHON_BIN" - \
  "$QWEN25_VL_7B_PATH" \
  "$STREAMVGGT_CKPT" \
  "$EVIDENCE_DIR/weight_hashes.json" <<'PY'
import json
import os
import sys

from common.reproducibility import hash_artifact_file, hash_local_hf_checkpoint

qwen = hash_local_hf_checkpoint(sys.argv[1])
vggt = hash_artifact_file(sys.argv[2])
assert qwen["manifest_sha256"] == os.environ["EXPECTED_QWEN25_MANIFEST_SHA256"]
assert qwen["shard_count"] == 5
assert vggt["sha256"] == os.environ["EXPECTED_STREAMVGGT_SHA256"]
payload = {"qwen2_5": qwen, "streamvggt": vggt}
with open(sys.argv[3], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
print(json.dumps(payload, indent=2))
PY
```

训练数据结构：

```bash
export DATA_READY=1
if ! test -d "$ANNOTATION_BASE/vln"; then DATA_READY=0; fi
if ! test -f "$ANNOTATION_BASE/vln/r2r/train/train_sub.json" \
  && ! test -f "$ANNOTATION_BASE/vln/r2r/train/train.json.gz"; then
  DATA_READY=0
fi
if ! test -f "$ANNOTATION_BASE/vln/r2r/train/train_gt.json.gz"; then
  DATA_READY=0
fi
if ! test -f "$ANNOTATION_BASE/vln/rxr/train_v2/annotations.json"; then
  DATA_READY=0
fi
echo "DATA_READY=$DATA_READY" | tee "$EVIDENCE_DIR/data_gate.txt"
```

如果权重 hash 不一致，停止并报告发现的 hash，不能称为相同 revision。
如果仅数据缺失，记为 `DATA_BLOCKED`，但仍继续 processor、load 和 generation
等不依赖训练数据的阶段。

## 5. 静态测试和配置 identity

`QWEN25_VL_7B_PATH` 已设置，因此真实 processor/index 测试不应在 Linux
上因为 Mac 默认路径而 skip。

```bash
PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m compileall -q \
  common data evaluator model realworld scripts trainer tests run.py launch.py

PYTHONDONTWRITEBYTECODE=1 \
QWEN25_VL_7B_PATH="$QWEN25_VL_7B_PATH" \
"$PYTHON_BIN" -m pytest -q | tee "$LOG_DIR/05_pytest.log"
```

必须没有 failed/error。记录 passed、skipped 和 subtests 的实际数量，不要只
比较总数。

```bash
"$PYTHON_BIN" - "$EVIDENCE_DIR/config_contract.json" <<'PY'
import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

with initialize_config_dir(version_base=None, config_dir=str(Path.cwd() / "configs")):
    baseline = compose(config_name="qwen2_5_baseline")
    full = compose(config_name="qwen2_5_full")
    qwen3 = compose(config_name="default")

targets = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
assert baseline.llm.model_family == full.llm.model_family == "qwen2_5_vl"
assert baseline.llm.lora.rank == full.llm.lora.rank == 16
assert baseline.llm.lora.alpha == full.llm.lora.alpha == 16
assert list(baseline.llm.lora.target_modules) == targets
assert list(full.llm.lora.target_modules) == targets
assert baseline.llm.hidden_dim == full.llm.hidden_dim == 3584
assert full.llm.mem_module.memory_token_num == 300
assert full.llm.mem_module.frame_max_tokens_num == 100
assert list(full.vision3d.input_grid_hw) == [30, 40]
assert baseline.rng_seed == full.rng_seed == 42
assert baseline.training.max_train_batches is None
assert full.training.max_train_batches is None
assert baseline.training.gradient_checkpointing is False
assert full.training.gradient_checkpointing is False
assert baseline.eval.run_after_train is True
assert full.eval.run_after_train is True
assert qwen3.llm.lora.rank == 128
assert qwen3.llm.lora.alpha == 256

payload = {
    "status": "PASS",
    "qwen25_lora": {"rank": 16, "alpha": 16, "targets": targets},
    "qwen3_lora": {
        "rank": int(qwen3.llm.lora.rank),
        "alpha": int(qwen3.llm.lora.alpha),
    },
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
```

## 6. 真实 Full base checkpoint load

本阶段在 CPU 上读取全部 Qwen shards，验证 thin Full class 没有 base-key
漂移。至少需要约 20GB 可用 host RAM。

```bash
"$PYTHON_BIN" - \
  "$QWEN25_VL_7B_PATH" \
  "$EVIDENCE_DIR/full_base_load.json" <<'PY'
import json
import sys
from pathlib import Path

import torch
from model.qwen.modeling_qwen2_5_vl_mem_vggt import (
    Qwen2_5_VLForConditionalGeneration_Mem_VGGT,
)

model, info = Qwen2_5_VLForConditionalGeneration_Mem_VGGT.from_pretrained(
    sys.argv[1],
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    local_files_only=True,
    low_cpu_mem_usage=True,
    output_loading_info=True,
)
for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
    assert not info.get(key), (key, info.get(key))
payload = {
    "status": "PASS",
    "class": type(model).__name__,
    "parameters": sum(parameter.numel() for parameter in model.parameters()),
    "embedding_shape": list(model.get_input_embeddings().weight.shape),
}
assert payload["parameters"] == 8292166656
assert payload["embedding_shape"] == [152064, 3584]
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
```

## 7. Baseline：一个真实 batch、save、resume/reload

smoke-only overrides 只有：单 batch 上限、non-reentrant gradient
checkpointing、关闭训练结束后的完整 Habitat validation，以及 `debug=true`
避免外部 W&B。模型、数据 mixture、像素预算和优化器不变。

若 `DATA_READY=0`，本节和下一节记为 `DATA_BLOCKED` 并跳过，直接执行
第 9 节；不要伪造或临时替换数据路径。

```bash
export SMOKE_PORT=29517
export BASELINE_ROOT="$RUN_DIR/training/baseline"
export BASELINE_EXP="$BASELINE_ROOT/Qwen2.5-VLNCE_qwen2_5_7b_r2r_rxr_v2_baseline"

run_baseline() {
  "$PYTHON_BIN" launch.py \
    --mode accelerate \
    --config configs/qwen2_5_baseline.yaml \
    --num_nodes 1 \
    --gpu_per_node 1 \
    --strategy ddp \
    --mixed_precision bf16 \
    --port "$SMOKE_PORT" \
    data.annotation_base="$ANNOTATION_BASE" \
    base_dir="$BASELINE_ROOT" \
    debug=true \
    training.max_train_batches=1 \
    training.gradient_checkpointing=true \
    training.checkpoint_total_limit=1 \
    eval.run_after_train=false
}

nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
  --format=csv -l 1 > "$LOG_DIR/07_baseline_gpu.csv" &
MONITOR_PID=$!
set +e
run_baseline 2>&1 | tee "$LOG_DIR/07_baseline_train.log"
BASELINE_STATUS=${PIPESTATUS[0]}
set -e
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
test "$BASELINE_STATUS" -eq 0

test -f "$BASELINE_EXP/config.yaml"
test -f "$BASELINE_EXP/reproducibility.json"
export BASELINE_CKPT="$(
  find "$BASELINE_EXP/checkpoints" -maxdepth 1 -type d -name 'checkpoint_*' \
    | sort -V | tail -n 1
)"
test -n "$BASELINE_CKPT"
test -f "$BASELINE_CKPT/model.safetensors" \
  || test -f "$BASELINE_CKPT/pytorch_model.bin"
find "$BASELINE_CKPT" -maxdepth 1 -type f -printf '%f %s\n' \
  | sort | tee "$EVIDENCE_DIR/baseline_checkpoint_files.txt"
```

`reproducibility.json` 应报告 Baseline raw/effective trainable parameters 都是
`40,370,176`。

第二次执行完全相同的命令，验证 full-state reload：

```bash
set +e
run_baseline 2>&1 | tee "$LOG_DIR/07_baseline_resume.log"
BASELINE_RESUME_STATUS=${PIPESTATUS[0]}
set -e
test "$BASELINE_RESUME_STATUS" -eq 0
grep -F "Train: resume from" "$LOG_DIR/07_baseline_resume.log"
grep -F "Successfully loaded from" "$LOG_DIR/07_baseline_resume.log"
if grep -F "Start training epoch" "$LOG_DIR/07_baseline_resume.log"; then
  echo "ERROR: resume unexpectedly trained another batch"
  exit 1
fi
```

由于 checkpoint 中的 tracker epoch 已是 1，而总 epoch 也是 1，第二次运行
应该只完成 load，不再训练。

## 8. Full：一个真实 batch、save、resume/reload

```bash
export FULL_ROOT="$RUN_DIR/training/full"
export FULL_EXP="$FULL_ROOT/Qwen2.5-VLNCE_qwen2_5_7b_full_memvln_r2r_rxr_v2"

run_full() {
  "$PYTHON_BIN" launch.py \
    --mode accelerate \
    --config configs/qwen2_5_full.yaml \
    --num_nodes 1 \
    --gpu_per_node 1 \
    --strategy ddp \
    --mixed_precision bf16 \
    --port "$SMOKE_PORT" \
    data.annotation_base="$ANNOTATION_BASE" \
    base_dir="$FULL_ROOT" \
    debug=true \
    training.max_train_batches=1 \
    training.gradient_checkpointing=true \
    training.checkpoint_total_limit=1 \
    eval.run_after_train=false
}

nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
  --format=csv -l 1 > "$LOG_DIR/08_full_gpu.csv" &
MONITOR_PID=$!
set +e
run_full 2>&1 | tee "$LOG_DIR/08_full_train.log"
FULL_STATUS=${PIPESTATUS[0]}
set -e
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
echo "$FULL_STATUS" | tee "$EVIDENCE_DIR/full_exit_code.txt"
```

如果成功：

```bash
export FULL_CKPT="$(
  find "$FULL_EXP/checkpoints" -maxdepth 1 -type d -name 'checkpoint_*' \
    | sort -V | tail -n 1
)"
test -n "$FULL_CKPT"
test -f "$FULL_CKPT/model.safetensors" \
  || test -f "$FULL_CKPT/pytorch_model.bin"

set +e
run_full 2>&1 | tee "$LOG_DIR/08_full_resume.log"
FULL_RESUME_STATUS=${PIPESTATUS[0]}
set -e
test "$FULL_RESUME_STATUS" -eq 0
grep -F "Train: resume from" "$LOG_DIR/08_full_resume.log"
grep -F "Successfully loaded from" "$LOG_DIR/08_full_resume.log"
if grep -F "Start training epoch" "$LOG_DIR/08_full_resume.log"; then
  echo "ERROR: Full resume unexpectedly trained another batch"
  exit 1
fi
```

Full 成功时的参考计数：total `9,918,522,496`，raw trainable
`1,262,240,896`，effective trainable `717,254,272`，三个 Memory
embedding rows 的 effective count 为 `10,752`。

只有同时满足以下条件才可把失败记为 `CAPACITY_BLOCKED_24GB`：

1. traceback 明确是 `torch.cuda.OutOfMemoryError` 或 CUDA allocator OOM；
2. 开始前没有其他 GPU 进程占用；
3. 已使用 BF16、FlashAttention2、rank 16、batch 1 和 non-reentrant
   gradient checkpointing；
4. 没有减少 300 Memory tokens、hidden size、视觉预算或 VGGT grid；
5. 静态测试和真实 base load 已通过；
6. 下一节在新进程中单独执行 Full no-grad generation。

非 OOM 的 shape/key/cache/device 错误一律记为 `CODE_FAIL`。禁止用 4/8-bit、
QLoRA、SDPA、删除 VGGT 或缩小 Memory 配置后宣布正式 PASS。

## 9. 两步短 generation 与 episode reset

仓库脚本使用两张确定性的 640×480 synthetic observations。它不评价动作
质量，但会真实加载 endpoint、在 GPU 上连续生成两个 clip，并让 Full
经过 Memory/VGGT cache 和 episode reset。生成长度临时设为 8。

Baseline 在训练成功时使用刚保存的 checkpoint；若训练阶段因数据阻塞未执行，
则不传 checkpoint，报告会明确标记为 base-only：

```bash
BASELINE_CHECKPOINT_ARGS=()
if test -n "${BASELINE_CKPT:-}"; then
  BASELINE_CHECKPOINT_ARGS=(--checkpoint "$BASELINE_CKPT")
fi

"$PYTHON_BIN" scripts/qwen25_generation_smoke.py \
  --endpoint baseline \
  --model-path "$QWEN25_VL_7B_PATH" \
  "${BASELINE_CHECKPOINT_ARGS[@]}" \
  --device cuda:0 \
  --max-new-tokens 8 \
  --report "$EVIDENCE_DIR/baseline_generation.json" \
  2>&1 | tee "$LOG_DIR/09_baseline_generation.log"
```

Full training checkpoint 成功时加载它；如果 Full one-batch 是容量阻塞，则
不传 checkpoint，报告会明确写
`base_only_untrained_project_modules`，不能把它描述成训练后质量测试。

```bash
FULL_CHECKPOINT_ARGS=()
if test -n "${FULL_CKPT:-}"; then
  FULL_CHECKPOINT_ARGS=(--checkpoint "$FULL_CKPT")
fi

"$PYTHON_BIN" scripts/qwen25_generation_smoke.py \
  --endpoint full \
  --model-path "$QWEN25_VL_7B_PATH" \
  --streamvggt-checkpoint "$STREAMVGGT_CKPT" \
  "${FULL_CHECKPOINT_ARGS[@]}" \
  --device cuda:0 \
  --max-new-tokens 8 \
  --report "$EVIDENCE_DIR/full_generation.json" \
  2>&1 | tee "$LOG_DIR/09_full_generation.log"
```

校验 JSON：

```bash
"$PYTHON_BIN" - \
  "$EVIDENCE_DIR/baseline_generation.json" \
  "$EVIDENCE_DIR/full_generation.json" <<'PY'
import json
import sys

baseline = json.load(open(sys.argv[1], encoding="utf-8"))
full = json.load(open(sys.argv[2], encoding="utf-8"))
for report in (baseline, full):
    assert report["status"] == "passed"
    assert report["transformers_version"] == "4.57.0"
    assert report["lora"]["rank"] == 16
    assert report["lora"]["alpha"] == 16
    assert len(report["steps"]) == 2
    assert all(step["visual_token_count"] == 391 for step in report["steps"])
    assert all(step["image_grid_thw"] == [[1, 34, 46]] for step in report["steps"])
    if report["checkpoint"]["status"] == "loaded":
        assert report["checkpoint"]["missing_trainable_keys"] == []

assert all(step["memory_token_count"] == 0 for step in baseline["steps"])
assert all(step["memory_token_count"] == 300 for step in full["steps"])
assert all(step["vggt_input_shape"] == [1, 3, 420, 560] for step in full["steps"])
assert full["episode_reset_counts"]["memory_episodes"] > 0
assert full["episode_reset_counts"]["vggt_past_key_values"] > 0
print("generation contracts: PASS")
PY
```

如果 Full 仅在把整个 endpoint 移到 4090 或 no-grad generation 时 OOM，记为
inference `CAPACITY_BLOCKED_24GB`；不要误判为 cache 实现错误。若已进入 forward
后出现 token、shape、key、prefill/cache 异常，则记为 `CODE_FAIL`。

## 10. 最终报告

不要只回复“跑通/没跑通”。提交以下表格和 first-failure 证据：

```markdown
# Qwen2.5 4090 Smoke Report

- Branch / commit:
- GPU / driver:
- Python / Torch / CUDA / FlashAttention:
- Transformers / PEFT / Accelerate:
- Qwen manifest SHA256:
- StreamVGGT SHA256:
- Annotation root:

| Stage | Baseline | Full | Evidence file |
|---|---|---|---|
| Static tests | | | |
| Config/processor contract | | | |
| Real Full base-key load | N/A | | |
| One batch forward/backward/step | | | |
| Checkpoint save | | | |
| Full-state resume | | | |
| Two-step greedy generation | | | |
| Episode cache reset | N/A | | |

## First failure

- Exact command:
- Exit code:
- Failure class:
- First traceback:
- Stage: load / forward / backward / optimizer step / save / reload / generation
- GPU free memory before start:
- Peak allocated memory:
- Other GPU processes:
- Any semantic config changes:
- Evidence files:

## Conclusion

- Baseline:
- Full:
- Is Full training blocked only by 24GB capacity:
- Remaining test required on a larger GPU/distributed setup:
```

若 Full 是 `CAPACITY_BLOCKED_24GB`，结论必须明确写：真实 Qwen2.5 与
StreamVGGT load 是否通过、no-grad generation 是否通过，以及正式 optimizer
step/save/reload 尚未满足，需要更大显存或分布式环境复验。`Full - Baseline`
仍只能解释为完整 MemVLN package 的增益，不能写成纯 Memory 增益。
