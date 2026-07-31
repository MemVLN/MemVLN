# MemVLN 4090 快速 R2R Inference Parity 与性能测试手册

> 本文件给另一台 RTX 4090 主机上的 Codex 直接执行。
>
> 纯 Qwen-LoRA baseline 已完成：优化后与原始 LEO-R 输出一致，120 episodes 约提速 30%。该结果现在是冻结的历史证据，**不要重跑 baseline**。
>
> 本轮只测试真正的 MemVLN checkpoint，并把 parity、逐步 trace、最终指标和性能统计合并到同一次 rollout 中。默认成功路径只有两次完整 rollout：
>
> 1. `A1 = 原始 LEO-R qwen3-nav`
> 2. `B1 = 当前 MemVLN，全部已确认 inference 优化开启`
>
> 当前已知 SHA256 为 `27086cf4...1ec` 的 `pytorch_model.bin` 是纯 Qwen-LoRA baseline，**不能用于本轮**。如果尚未提供另一份真正的 MemVLN checkpoint，本轮状态应为 `BLOCKED (matching MemVLN checkpoint not supplied)`，不得猜测或退回 baseline 合同。
>
> 2026-07-30 的首次 MemVLN 快速测试已完成 P0/P1 和 checkpoint/warmup gates，但 B_SMOKE 因 test-only harness 重复设置首个 measured episode 而停止。本版默认复用那些已验证证据，只修复 harness并从全新的 B_SMOKE 继续；不得重跑已通过的重型前置测试。

## 0. 最短成功路径

严格按以下顺序执行：

| 阶段 | 内容 | 大模型完整 rollout 次数 |
|---|---|---:|
| P0 | 复用已验证的环境、Hydra、单测和 Habitat/EGL evidence；仅做 identity gate | 0 |
| P1 | 复用已验证的 32-episode/11-scene manifest 和 3-episode smoke manifest | 0 |
| P2 | 在 B 上跑 3-episode fail-fast smoke | 1 个很小的 smoke |
| P3 | A1：原始 LEO-R，在固定 manifest 上同时收集 parity 与 timing | 1 |
| P4 | B1：当前 MemVLN，在相同 manifest 上同时收集 parity 与 timing | 1 |
| P5 | exact comparator；parity 通过后读取同批 timing | 0 |

默认到 P5 即停止。不要预先运行以下旧矩阵：

- 不跑独立 `REF-B`；
- 不跑 `CACHE`、`ATTN`、`HABITAT` 三个完整 attribution rollout；
- 不另跑一套 performance shard；
- 不另跑同步式 profiler；
- 不默认跑 RxR；
- 不默认跑视频；
- 不默认跑 120 episodes；
- 不重跑已经接受的纯 Qwen baseline。

只有第 11 节的预定义触发条件成立，才增加 run。不要因为“机器已经开着”而扩展矩阵。

### 0.1 本次 stopped-run 的结论

已确认以下部分不是失败原因：

- MemVLN checkpoint gate PASS：571 tensors，0 active unexpected keys，0 missing active trainable keys；
- StreamVGGT checkpoint 1210/1210 model keys loaded；
- Hydra compose、R2R evaluator、数据路径和 EGL 均正常；
- 完整 manifest PASS：32 episodes、11 scenes；
- warmup PASS：2 次 generation、4 个 physical actions，Memory/grounded/StreamVGGT state 清理完成；
- 第一轮 measured generation 已实际写入 grounded cache、StreamVGGT cache，并执行 `need_weights=False`；
- 无 OOM、EGL error 或 CUDA error。

唯一直接失败原因是：

```text
首个 measured episode 已在 timer 前完成：
env.current_episode = episode
observations = env.reset()

随后 rollout_episode(initial_observations=observations) 又执行：
env.current_episode = episode

但没有第二次 reset，导致第一次 env.step() 触发 Habitat episode-changed invariant。
```

因此本轮不得重跑 P0/P1 来“试试看”；必须先执行第 6.5 节的精确 harness 修复。

## 1. 本轮要回答的两个问题

### 1.1 严格等价

同一 MemVLN checkpoint、同一 R2R episode、同一 seed 下，A1 与 B1 必须逐项一致：

- episode 的有序 `(scene_id, episode_id)` manifest；
- 每个 physical step 的原始 RGB shape、dtype、SHA256；
- 每次 generation 的最终 `input_ids` shape、dtype、SHA256；
- 原始生成文本；
- parser 的结构化结果；
- merged action 的 repeat count；
- 每一个实际传给 `env.step()` 的 physical action；
- STOP 所在 physical step；
- episode steps；
- SR、SPL、NE、OS。

先按 exact equality 检查浮点指标。只有确认差异纯粹来自 JSON 数值表示时，才可以额外报告误差；不得直接改 tolerance 后宣布 PASS。

### 1.2 端到端提速

只有 1.1 全部 PASS 后，才计算：

```text
speedup = A1.core_rollout_seconds / B1.core_rollout_seconds
gain_percent = (speedup - 1) * 100
```

这里比较的是：

- A1：原始 LEO-R evaluator，包括旧的逐-step `empty_cache()`、inline debug measures 和旧 Memory eval path；
- B1：当前 MemVLN 默认优化路径，包括 grounded Memory cache、`need_weights=False`、Fast Habitat、episode-end state reset 等。

不得用当前 MemVLN 内部 `REF/ALL` 的比值冒充完整 LEO-R before/after。

## 2. 不可改变的模型合同

- 不训练，不更新权重，不 commit，不 push，不修改 README。
- A1 与 B1 必须使用同一份 MemVLN navigation checkpoint。
- 两边必须使用同一 Qwen base、同一 StreamVGGT checkpoint、BF16 和 `flash_attention_2`。
- 保持：
  - `task=sft_r2r_rxr_v2_w_traj_sum_w_mem`
  - `llm=qwen3-4b-mem`
  - `vision3d=streamvggt`
  - `data.vln.prompt_format=v2`
  - `data.vln.memory_prompt=true`
  - `data.vln.max_num_frames=16`
  - `data.vln.obs_hist_sample=nearest`
  - `data.vln.merge_action=true`
  - `data.vln.max_act_hist=0`
  - `data.vln.skip_merge_step_obs=false`
  - `data.vln.vggt_cache_feat=true`
  - `rng_seed=42`
- 不改变 prompt、history frame、dense StreamVGGT 输入、Memory token 数、生成参数、action parser 或 max context/output length。
- 不允许 FlashAttention 失败后静默回退到 SDPA/eager。
- 单卡只运行一个模型进程。
- 每个 run 使用全新的输出目录；禁止 resume partial result。
- `save_video=0`。Trace logger 必须与视频彻底解耦。
- A1 只允许加入不改变行为的 test-only manifest filter、trace/hash logger 和低干扰 timer。
- B1 也只能加入相同 test-only instrumentation。
- MemVLN 使用 `memory_prompt=true`，因此纯 baseline 的 `False` prompt bug 不会被触发；仍应检查最终 prompt/input 中没有字面量 `FalseCurrent observation`。

## 3. 路径和固定 artifact

以下代码使用 Bash array，必须在 Bash 4+ 中执行：

```bash
set -euo pipefail
test -n "${BASH_VERSION:-}"
(( BASH_VERSINFO[0] >= 4 ))

export MEMVLN_SOURCE_REPO=/home/wangyan/bigai/codes/MemVLN
export LEOR_SOURCE_REPO=/home/wangyan/bigai/codes/LEO-R
export PYTHON_BIN="${PYTHON_BIN:-/home/wangyan/anaconda3/envs/qwen3-nav/bin/python}"
export QWEN_DIR=/mnt/fillipo/pretrained_weights/Qwen3-VL-4B-Instruct
export STREAMVGGT_CKPT=/mnt/fillipo/pretrained_weights/StreamVGGT-1B.pth
export VLNCE_ROOT=/mnt/fillipo/Datasets/VLN-CE

# 已由 stopped-run 验证，包含 Memory/StreamVGGT merger/LoRA 参数。
export MEMVLN_NAV_CKPT=/home/wangyan/bigai/ckpts/vln/memvln.safetensors
export MEMVLN_NAV_SHA256=6d5592fbfb97f086e628511ae6cbc94b305aca71cb2e6dc068e37ff610b4f606
export BASELINE_NAV_SHA256=27086cf4c1f6dbcbc2d85ac7848291eaf98794d83160236352a2d3f3028611ec
export QWEN_DIR_SHA256=9d092c5dc90c6565762262360098cf17e45b6bbd9a12c47fb19214893cb88d5c
export QWEN_CONFIG_SHA256=edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d
export STREAMVGGT_SHA256=3d3d914d09609fdb485e16947c4885cca0e5ea4e5557598a3b3ffb1cfe0a2f59
export R2R_DATA_SHA256=1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3

# 原始 LEO-R snapshot。
export EXPECTED_LEOR_SHA=0bbcfad8b168dd26f4297414e41dab1e87a8f7b2

# 已完成 baseline 的结果目录只作 provenance；可以留空，但绝不重跑。
export ACCEPTED_BASELINE_DIR="${ACCEPTED_BASELINE_DIR:-}"

export TEST_BATCH="$(date +%Y%m%d_%H%M%S)"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/memvln_4090_fast_${TEST_BATCH}}"
export PREVIOUS_RUN_ROOT="${PREVIOUS_RUN_ROOT:-/tmp/memvln_4090_fast_20260730_103345_stopped}"
export REUSE_VERIFIED_PREFIX="${REUSE_VERIFIED_PREFIX:-1}"
export MEMVLN_REPO="$OUTPUT_ROOT/worktrees/MemVLN"
export LEOR_REPO="$OUTPUT_ROOT/worktrees/LEO-R"
export MANIFEST_SIZE="${MANIFEST_SIZE:-32}"
export MANIFEST_SEED=42
export R2R_HABITAT_TEMPLATE="$MEMVLN_REPO/configs/habitat/r2r.yaml"
export R2R_HABITAT_CONFIG="$OUTPUT_ROOT/manifests/r2r_resolved.yaml"

unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
unset PYTHONOPTIMIZE
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONHASHSEED=42
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

test -x "$PYTHON_BIN"
test -d "$MEMVLN_SOURCE_REPO"
test -d "$LEOR_SOURCE_REPO"

if [[ "$REUSE_VERIFIED_PREFIX" == 1 \
      && ! -d "$PREVIOUS_RUN_ROOT" \
      && -d /tmp/memvln_4090_fast_20260730_103345 ]]; then
  export PREVIOUS_RUN_ROOT=/tmp/memvln_4090_fast_20260730_103345
fi

# 必须在首次 mkdir/cp/write 前执行。新目录不得已存在，也不得与旧目录重叠。
test ! -e "$OUTPUT_ROOT"
"$PYTHON_BIN" - "$OUTPUT_ROOT" "$PREVIOUS_RUN_ROOT" <<'PY'
import sys
from pathlib import Path

output = Path(sys.argv[1]).resolve()
previous = Path(sys.argv[2]).resolve()
if output == previous or output in previous.parents or previous in output.parents:
    raise RuntimeError(
        f"OUTPUT_ROOT and PREVIOUS_RUN_ROOT overlap: {output} vs {previous}"
    )
print("Fresh output isolation: PASS", output)
PY

mkdir -p "$OUTPUT_ROOT"/{logs,evidence,manifests,source,worktrees}

prepare_execution_copies() {
  test ! -e "$MEMVLN_REPO"
  test ! -e "$LEOR_REPO"
  mkdir "$MEMVLN_REPO"
  cp -a "$MEMVLN_SOURCE_REPO/." "$MEMVLN_REPO/"
  git clone --quiet --no-hardlinks "$LEOR_SOURCE_REPO" "$LEOR_REPO"
  git -C "$LEOR_REPO" checkout --quiet --detach "$EXPECTED_LEOR_SHA"
}

# 默认复用这次已经通过的 P0/P1。旧目录只读；新 run 仍使用全新 OUTPUT_ROOT。
if [[ "$REUSE_VERIFIED_PREFIX" == 1 ]]; then
  test -d "$PREVIOUS_RUN_ROOT"

  "$PYTHON_BIN" - "$PREVIOUS_RUN_ROOT" "$OUTPUT_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

previous = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
expected = {
    "memvln_test_rollout.py": "22eaf8cc48a3e08bf78a9f6cb0cdf8b56444c72079a51c2c32f64cb82e46482a",
    "logs/p0_environment.log": "26dc40dd871586ca15bffa25474f409702a6a0d43a6983d2d1d73aad8b7b2eab",
    "logs/p0_flashattention_cuda_smoke.log": "a27851bb24d73a441ab08ba9e01e702218694df12231584cd8a7b2a6200305b2",
    "logs/p0_memvln_contract_tests.log": "d592038d0d1995b0dc231693d147289bc3863655d077e8206b0391aa4100d89f",
    "logs/p0_habitat_egl_smoke.log": "295f20dc564f103c0d9791f9e96a7bfecd647537eee4a22f46145792df29a285",
    "logs/p1_build_r2r_manifest.log": "d45a0675c964fb15ca9b95f3b4731e13f79651815cc5dc9e757a53559e52d341",
    "logs/p1_validate_r2r_manifest.log": "d25be33daffff1f1322b7cbe6a1a4733a9365d6e2dc7c96b57ccc4be6e605b6a",
    "logs/p2_build_smoke_manifest.log": "215c2736ef8eef1a7e4908a9f47a863fd64a9673bec1eb9e9c618acac0ceeb35",
    "logs/p2_memvln_contract_tests.log": "39db750b90432248ce519c1048a4d1083ec54cd37c76e93db9e27e2f5871dfa8",
    "source/pip_freeze.txt": "07772dc1401e983ff923fb7328b98c8f64044f1dec0b1e05cc0d22fe6778cff4",
    "source/memvln_original_source_before_relative.sha256": "89a0ca938b8a524b7157bec8d226332d88849bfde3ff8dbaf5140823b0c944ff",
    "source/leor_snapshot_source_before.sha256": "59a7943ffa010267818039f65f2819fff137723170217932e8cc4638e8b4dfc6",
    "manifests/r2r_resolved.yaml": "1757d9bd85c845a588f59aacb691081daf8064419ff03adfc32a5dced2c75aca",
    "manifests/r2r_32.json": "73eada2e9da61602c873c5a86c6be82f271a9f8dc76a8cd87d99f30c8bee9484",
    "manifests/r2r_smoke_3.json": "b6f63b495d23c80a3337403758537fe26a40bb82b34ff1c2b0eafbfc910efca2",
    "evidence/P0_A_resolved_config.yaml": "a5ac0bf1a82f2e4e62a185d6e39f87905c40970dd38d5ff19544be1a06b451a7",
    "evidence/P0_B_resolved_config.yaml": "ce6a9837962f6d99e285d9d4e0c9e4f95d770f9362bcfbb14fc0261b4ad2f61b",
    "evidence/B_SMOKE/checkpoint_load_report.json": "a6b6e0a9489e6f4b060395d8db7953cca00309bf109ec17a421f5b867595d372",
    "STOP_REPORT.md": "38aee884b2936622effcc8c00589c0bb0cc429e2a6cce2eb6ea2c4839181c461",
}

actual = {}
for relative, expected_sha in expected.items():
    path = previous / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha:
        raise RuntimeError(
            f"previous evidence changed: {relative}: {digest} != {expected_sha}"
        )
    actual[relative] = digest

checkpoint = json.loads(
    (previous / "evidence/B_SMOKE/checkpoint_load_report.json").read_text()
)
assert checkpoint["status"] == "PASS", checkpoint
assert checkpoint["checkpoint_actual_sha256"] == (
    "6d5592fbfb97f086e628511ae6cbc94b305aca71cb2e6dc068e37ff610b4f606"
)
assert checkpoint["active_missing_trainable_keys"] == []
assert checkpoint["active_unexpected_keys"] == []

manifest = json.loads((previous / "manifests/r2r_32.json").read_text())
assert len(manifest["episodes"]) == 32
assert len({str(x["scene_id"]) for x in manifest["episodes"]}) == 11

report = {
    "status": "PASS",
    "previous_run_root": str(previous),
    "reused_files": actual,
    "skipped_stages": [
        "P0 environment and FlashAttention",
        "P0 production contract tests",
        "P0 sequential Habitat/EGL smoke",
        "P1 full manifest build and reconstruction",
        "P2 smoke manifest build",
        "standalone checkpoint preflight",
    ],
    "must_rerun": [
        "test-only helper regression tests",
        "source/helper hash freeze",
        "warmup",
        "B_SMOKE",
        "A1",
        "B1",
    ],
}
(output / "source/reused_prefix_verification.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("Verified reusable P0/P1 prefix: PASS")
PY

  # 只做廉价的 runtime identity gate，不重跑 FlashAttention、生产单测或 Habitat。
  "$PYTHON_BIN" -m pip freeze > "$OUTPUT_ROOT/source/pip_freeze_current.txt"
  test "$(sha256sum "$OUTPUT_ROOT/source/pip_freeze_current.txt" | awk '{print $1}')" = \
    07772dc1401e983ff923fb7328b98c8f64044f1dec0b1e05cc0d22fe6778cff4
  "$PYTHON_BIN" - <<'PY'
import torch

assert torch.cuda.is_available()
assert "4090" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
assert torch.cuda.get_device_capability(0) == (8, 9)
assert torch.cuda.is_bf16_supported()
print("Reusable runtime identity: PASS", torch.__version__, torch.version.cuda)
PY

  # P0/P1 只能在 production code 未变化时复用。执行副本与 production 隔离；
  # MemVLN 仅忽略本测试手册及其备份，其他文件必须逐文件完全一致。
  prepare_execution_copies
  "$PYTHON_BIN" - \
    "$MEMVLN_REPO" \
    "$PREVIOUS_RUN_ROOT/source/memvln_original_source_before_relative.sha256" \
    "$LEOR_REPO" \
    "$PREVIOUS_RUN_ROOT/source/leor_snapshot_source_before.sha256" \
    "$OUTPUT_ROOT/source/current_source_identity.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

memvln_root = Path(sys.argv[1]).resolve()
memvln_manifest = Path(sys.argv[2]).resolve()
leor_root = Path(sys.argv[3]).resolve()
leor_manifest = Path(sys.argv[4]).resolve()
report_path = Path(sys.argv[5]).resolve()

EXCLUDED_PARTS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}

def ignored(relative: str, *, allow_runbook_change: bool) -> bool:
    parts = Path(relative).parts
    if any(part in EXCLUDED_PARTS for part in parts):
        return True
    name = Path(relative).name
    if Path(relative).suffix in EXCLUDED_SUFFIXES:
        return True
    if name == "core" or name.startswith("core."):
        return True
    return allow_runbook_change and name.startswith(
        "CODEX_4090_INFERENCE_TEST"
    ) and name.endswith(".md")

def load_manifest(path: Path, *, allow_runbook_change: bool) -> dict[str, str]:
    expected = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.removeprefix("./")
        if ignored(relative, allow_runbook_change=allow_runbook_change):
            continue
        expected[relative] = digest
    return expected

def verify_tree(
    root: Path,
    manifest: Path,
    *,
    allow_runbook_change: bool,
) -> dict:
    expected = load_manifest(
        manifest, allow_runbook_change=allow_runbook_change
    )
    actual_paths = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if ignored(relative, allow_runbook_change=allow_runbook_change):
            continue
        actual_paths[relative] = path
    if set(actual_paths) != set(expected):
        raise RuntimeError(
            f"source file set changed under {root}: "
            f"missing={sorted(set(expected) - set(actual_paths))[:20]}, "
            f"extra={sorted(set(actual_paths) - set(expected))[:20]}"
        )
    mismatches = {}
    for relative, path in actual_paths.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected[relative]:
            mismatches[relative] = {
                "expected": expected[relative],
                "actual": actual,
            }
    if mismatches:
        raise RuntimeError(f"source content changed under {root}: {mismatches}")
    return {
        "root": str(root),
        "manifest": str(manifest),
        "verified_file_count": len(expected),
        "status": "PASS",
    }

payload = {
    "status": "PASS",
    "memvln": verify_tree(
        memvln_root, memvln_manifest, allow_runbook_change=True
    ),
    "leor": verify_tree(
        leor_root, leor_manifest, allow_runbook_change=False
    ),
}
report_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("Current production source identity: PASS")
PY

  mkdir -p "$OUTPUT_ROOT/reused_evidence/logs"
  # 只把旧 helper 作为待修复起点复制；第 6.5 节要求其 SHA 必须改变。
  cp "$PREVIOUS_RUN_ROOT/memvln_test_rollout.py" \
    "$OUTPUT_ROOT/memvln_test_rollout.py"
  cp "$PREVIOUS_RUN_ROOT/manifests/r2r_resolved.yaml" "$R2R_HABITAT_CONFIG"
  cp "$PREVIOUS_RUN_ROOT/manifests/r2r_32.json" "$OUTPUT_ROOT/manifests/r2r_32.json"
  cp "$PREVIOUS_RUN_ROOT/manifests/r2r_smoke_3.json" "$OUTPUT_ROOT/manifests/r2r_smoke_3.json"
  for passed_log in \
    p0_environment.log \
    p0_flashattention_cuda_smoke.log \
    p0_memvln_contract_tests.log \
    p0_habitat_egl_smoke.log \
    p1_build_r2r_manifest.log \
    p1_validate_r2r_manifest.log \
    p2_build_smoke_manifest.log \
    p2_memvln_contract_tests.log
  do
    cp "$PREVIOUS_RUN_ROOT/logs/$passed_log" \
      "$OUTPUT_ROOT/reused_evidence/logs/$passed_log"
  done
  cp "$PREVIOUS_RUN_ROOT/evidence/P0_A_resolved_config.yaml" "$OUTPUT_ROOT/reused_evidence/"
  cp "$PREVIOUS_RUN_ROOT/evidence/P0_B_resolved_config.yaml" "$OUTPUT_ROOT/reused_evidence/"
  cp "$PREVIOUS_RUN_ROOT/evidence/B_SMOKE/checkpoint_load_report.json" \
    "$OUTPUT_ROOT/reused_evidence/checkpoint_load_report_previous.json"
  cp "$PREVIOUS_RUN_ROOT/source/memvln_original_source_before_relative.sha256" \
    "$OUTPUT_ROOT/source/"
  cp "$PREVIOUS_RUN_ROOT/source/leor_snapshot_source_before.sha256" \
    "$OUTPUT_ROOT/source/"
  cp "$PREVIOUS_RUN_ROOT/STOP_REPORT.md" "$OUTPUT_ROOT/reused_evidence/"
else
  prepare_execution_copies
  # Fresh fallback：两个 repo 必须共用这一份 repo 外 test-only Habitat config。
  # 原 YAML 硬编码了 /share/generalvision/...；只设置 VLNCE_ROOT 并不会覆盖它。
  "$PYTHON_BIN" - "$R2R_HABITAT_TEMPLATE" "$R2R_HABITAT_CONFIG" "$VLNCE_ROOT" <<'PY'
import sys
from pathlib import Path
import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
root = Path(sys.argv[3]).resolve()
payload = yaml.safe_load(source.read_text(encoding="utf-8"))
dataset = payload["habitat"]["dataset"]
dataset["scenes_dir"] = str(root)
dataset["data_path"] = str(
    root / "R2R_VLNCE_v1-3_preprocessed" / "{split}" / "{split}.json.gz"
)
target.write_text(
    yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
print(target)
PY
fi
```

检查路径和 hash：

```bash
test -d "$MEMVLN_REPO"
test -d "$LEOR_REPO"
test -d "$QWEN_DIR"
test -f "$STREAMVGGT_CKPT"
test -f "$MEMVLN_NAV_CKPT"
test -f "$R2R_HABITAT_TEMPLATE"
test -f "$R2R_HABITAT_CONFIG"
test "$(sha256sum "$MEMVLN_NAV_CKPT" | awk '{print $1}')" = "$MEMVLN_NAV_SHA256"
test "$MEMVLN_NAV_SHA256" != "$BASELINE_NAV_SHA256"
test "$(sha256sum "$QWEN_DIR/config.json" | awk '{print $1}')" = "$QWEN_CONFIG_SHA256"
test "$(sha256sum "$STREAMVGGT_CKPT" | awk '{print $1}')" = "$STREAMVGGT_SHA256"
test "$(git -C "$LEOR_REPO" rev-parse HEAD)" = "$EXPECTED_LEOR_SHA"
test -f "$VLNCE_ROOT/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
test "$(sha256sum "$VLNCE_ROOT/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz" | awk '{print $1}')" = "$R2R_DATA_SHA256"

"$PYTHON_BIN" - "$R2R_HABITAT_CONFIG" "$VLNCE_ROOT" <<'PY'
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = str(Path(sys.argv[2]).resolve())
dataset = config["habitat"]["dataset"]
assert dataset["scenes_dir"] == root, dataset
assert dataset["data_path"] == (
    f"{root}/R2R_VLNCE_v1-3_preprocessed/{{split}}/{{split}}.json.gz"
), dataset
print("Resolved R2R Habitat paths: PASS")
PY

if [[ -n "$ACCEPTED_BASELINE_DIR" ]]; then
  test -d "$ACCEPTED_BASELINE_DIR"
  find "$ACCEPTED_BASELINE_DIR" -type f -print0 \
    | sort -z \
    | xargs -0 -r sha256sum \
    > "$OUTPUT_ROOT/source/accepted_baseline_files.sha256"
fi
```

保存 source 状态。不要清理用户已有改动：

```bash
for repo_name in leor memvln; do
  if [[ "$repo_name" == leor ]]; then repo="$LEOR_REPO"; else repo="$MEMVLN_REPO"; fi
  git -C "$repo" rev-parse HEAD > "$OUTPUT_ROOT/source/${repo_name}_head.txt" 2>&1 || true
  git -C "$repo" branch --show-current > "$OUTPUT_ROOT/source/${repo_name}_branch.txt" 2>&1 || true
  git -C "$repo" status --short > "$OUTPUT_ROOT/source/${repo_name}_status_before.txt" 2>&1 || true
done

find "$MEMVLN_REPO" \
  -type f \
  ! -path '*/.git/*' \
  ! -path '*/__pycache__/*' \
  ! -name '*.pyc' \
  -print0 \
  | sort -z \
  | xargs -0 -r sha256sum \
  > "$OUTPUT_ROOT/source/memvln_source_before.sha256"
```

## 4. 零 rollout preflight

当且仅当第 3 节生成的 `source/reused_prefix_verification.json` 为 `status=PASS` 时：

- **跳过** 4.1 的完整环境/FlashAttention smoke；
- **跳过** 4.3 的 66 个生产合同测试、AST/YAML/shell 全量检查；
- **跳过** 4.4 的 Habitat/EGL smoke；
- **跳过**单独的 checkpoint preflight。

仍必须执行第 3 节的当前 artifact identity checks、第 6.5 节新增的两个 harness regression tests，并在新的 B_SMOKE 进程内重新执行 loader checkpoint gate。旧 B_SMOKE 的 warmup和 partial trace不能复用。

如果 reuse verification 没有 PASS，才从 4.1 开始走完整 fresh path。

### 4.1 GPU、runtime 和 FlashAttention

```bash
"$PYTHON_BIN" --version
nvidia-smi

set +e
"$PYTHON_BIN" -m pip check 2>&1 | tee "$OUTPUT_ROOT/logs/p0_pip_check.log"
pip_check_status=${PIPESTATUS[0]}
set -e
if (( pip_check_status != 0 )); then
  # 本机唯一已知的 metadata warning。运行时 import 已在通过的 P0 中验证。
  remaining_pip_errors="$(
    sed '/^decord 0.6.0 is not supported on this platform$/d;/^[[:space:]]*$/d' \
      "$OUTPUT_ROOT/logs/p0_pip_check.log"
  )"
  if [[ -n "$remaining_pip_errors" ]]; then
    echo "Unexpected pip check failure:" >&2
    printf '%s\n' "$remaining_pip_errors" >&2
    exit 1
  fi
fi

"$PYTHON_BIN" - <<'PY'
from importlib.metadata import version
import sys
import torch

assert sys.flags.optimize == 0, "Do not use python -O"
assert torch.cuda.is_available()
assert "4090" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
assert torch.cuda.get_device_capability(0) == (8, 9)
assert torch.cuda.is_bf16_supported()

for package in (
    "torch", "torchvision", "transformers", "accelerate", "peft",
    "flash-attn", "habitat-sim", "habitat-lab", "habitat-baselines",
):
    try:
        print(package, version(package))
    except Exception as error:
        raise RuntimeError(f"{package}: {error}") from error

print("GPU:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
PY
```

再运行一次真实 FlashAttention CUDA kernel smoke。失败即停止；不允许改 backend。

保存环境：

```bash
"$PYTHON_BIN" -m pip freeze > "$OUTPUT_ROOT/source/pip_freeze.txt"
nvidia-smi -q > "$OUTPUT_ROOT/source/nvidia_smi_q_before.txt"
```

### 4.2 Checkpoint fail-fast

远端 Codex必须在两个 runner 的 `load_state_dict(strict=False)` 后加入 test-only fail-fast 检查，并保存完整 key report：

- LoRA 参数存在且实际加载；
- Memory bank、Memory embeddings/retrieval 参数存在且实际加载；
- VGGT embedding merger 参数存在且实际加载；
- 两边相同的 trainable inference 参数都被 checkpoint 覆盖；
- 当前 MemVLN 只允许旧 checkpoint 中的 `memory_repa_adapter.*` 被报告为 deprecated unexpected keys；
- 除 RePA 以外的 unexpected key 一律 FAIL；
- 除原始 LEO-R 中 eval-only、从不参与 forward 的 RePA adapter 差异外，missing trainable key 一律 FAIL；
- 禁止“打印 missing/unexpected 后继续跑”。

纯 Qwen-LoRA baseline checkpoint 没有 Memory/merger 参数，必须在这里 fail-fast，不能进入 rollout。

### 4.3 当前 MemVLN 合同测试

```bash
cd "$MEMVLN_REPO"

"$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py'

"$PYTHON_BIN" - <<'PY'
import ast
from pathlib import Path
for path in Path(".").rglob("*.py"):
    if ".git" not in path.parts:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("Python AST: PASS")
PY

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import yaml
for path in Path("configs").rglob("*.yaml"):
    yaml.safe_load(path.read_text(encoding="utf-8"))
print("YAML parse: PASS")
PY

for script in split_eval/*.sh; do bash -n "$script"; done
```

Hydra compose 必须确认最终 resolved config 是 MemVLN 合同，并且 R2R 是唯一 evaluator。不要为了测试 R2R 而构造 RxR evaluator。

### 4.4 Habitat/EGL smoke

不加载大模型，**在同一进程中严格串行**创建、step并关闭：

1. 原始 LEO-R R2R env；
2. 当前 MemVLN `fast_habitat=true` R2R env。

两边对同一 episode 执行 `reset()` 和一个固定 action，确认：

- RGB 存在，shape/dtype/value range 相同；
- scene 和 episode id 相同；
- EGL 可用；
- B 的主 env 不含 depth/GPS/compass、TopDownMap、fog、collision；
- `save_video=0` 时 B 不创建 debug env。

禁止并发创建 A/B Habitat renderer。首次 stopped-run 的并发尝试出现过 `GL::Context::current(): no current context`；串行关闭 A 后再创建 B 已 PASS。

## 5. 一次生成固定 R2R manifest

不要再使用 `REF-A` 的结果集合隐式定义 manifest，也不要用 `split_num=16/512` 猜 episode 数。

远端 Codex在 `$OUTPUT_ROOT/build_r2r_manifest.py` 创建临时脚本。脚本只加载 R2R `val_unseen` dataset/config，不加载模型：

1. 枚举全部 episodes；
2. 使用 evaluator 中的 canonical scene id；
3. 按 scene 分组；
4. scene 名排序；
5. scene 内按 `sha256("42|scene_id|episode_id")` 排序；
6. scene round-robin **选择** episode，直到得到 `MANIFEST_SIZE=32`；
7. 将已选 episodes 重新按 `scene_id` 分组执行，scene 内保持稳定 hash 顺序，避免 round-robin 执行导致反复加载 scene；
8. 另选一个不在 measured manifest 中的 warmup episode；
9. 保存有序 JSON，不依赖 Python set/dict 的偶然顺序。

Schema：

```json
{
  "dataset": "R2R",
  "split": "val_unseen",
  "seed": 42,
  "episodes": [
    {
      "order": 0,
      "selection_rank": 0,
      "scene_id": "canonical_scene_id",
      "episode_id": "episode_id",
      "instruction_sha256": "..."
    }
  ],
  "warmup": {
    "scene_id": "canonical_scene_id",
    "episode_id": "episode_id"
  }
}
```

生成并检查：

```bash
export TEST_MANIFEST="$OUTPUT_ROOT/manifests/r2r_${MANIFEST_SIZE}.json"

if [[ "$REUSE_VERIFIED_PREFIX" == 1 ]]; then
  test "$MANIFEST_SIZE" = 32
  test "$(sha256sum "$TEST_MANIFEST" | awk '{print $1}')" = \
    73eada2e9da61602c873c5a86c6be82f271a9f8dc76a8cd87d99f30c8bee9484
  test "$(sha256sum "$OUTPUT_ROOT/manifests/r2r_smoke_3.json" | awk '{print $1}')" = \
    b6f63b495d23c80a3337403758537fe26a40bb82b34ff1c2b0eafbfc910efca2
  sha256sum "$OUTPUT_ROOT/manifests/r2r_smoke_3.json" \
    > "$OUTPUT_ROOT/manifests/r2r_smoke_3.json.sha256"
  echo "P1 manifest build/reconstruction: REUSED VERIFIED PASS"
else
  "$PYTHON_BIN" "$OUTPUT_ROOT/build_r2r_manifest.py" \
    --config "$R2R_HABITAT_CONFIG" \
    --split val_unseen \
    --size "$MANIFEST_SIZE" \
    --seed "$MANIFEST_SEED" \
    --output "$TEST_MANIFEST"

  "$PYTHON_BIN" - "$TEST_MANIFEST" "$MANIFEST_SIZE" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
expected_size = int(sys.argv[2])
episodes = payload["episodes"]
keys = [(str(x["scene_id"]), str(x["episode_id"])) for x in episodes]
assert len(episodes) == expected_size, (len(episodes), expected_size)
assert len(keys) == len(set(keys))
assert len({x[0] for x in keys}) >= 8
assert (str(payload["warmup"]["scene_id"]), str(payload["warmup"]["episode_id"])) not in set(keys)
assert [x["order"] for x in episodes] == list(range(len(episodes)))
print("Manifest: PASS", len(episodes), "episodes,", len({x[0] for x in keys}), "scenes")
PY
fi

sha256sum "$TEST_MANIFEST" > "${TEST_MANIFEST}.sha256"
```

如果实际 R2R 数据无法满足 32 episodes 或至少 8 scenes，停止并报告数据问题；不要悄悄缩小门槛。

## 6. 合并 parity 与 timing 的 test-only instrumentation

远端 Codex应在两个 repo 中加入功能等价的临时 instrumentation。保存精确 diff/文件 hash，但不提交。

### 6.1 Manifest filter

使用环境变量：

```text
MEMVLN_TEST_MANIFEST=/absolute/path/to/r2r_32.json
```

在 evaluator 已加载 episodes、但尚未开始 scene/episode loop 时：

- 按 manifest 的有序 keys 精确过滤；
- 强制按 manifest order 执行；
- 缺少、重复或多出 episode 都 fail-fast；
- 使用显式 manifest 后，runner 固定 `--split-id 0 --split-num 1`；
- 禁止 resume/skip。

### 6.2 Warmup

模型、processor 和 Habitat env 全部 ready 后，先在 manifest 的 `warmup` episode 完成一次不计时 rollout：

- 不写入正式 result/trace；
- 不改变 A 在 measured episodes 中的原始 reset 行为；
- B 使用现有 `reset_episode_state(warmup_episode_id)`；A 的 test harness 只删除 warmup id 对应的 Memory bank 和 StreamVGGT `past_key_values/past_clip_ids/past_feats` entries，使 measured timer 开始前恢复到冷 episode-state；
- 不得把 B 的 episode-end reset 反向移植到 A 的 measured loop；
- HF generation KV 只是单次 `generate()` 的局部对象，Qwen `rope_deltas` 也不是 episode-keyed；释放 warmup 的局部 input/output 引用即可，**不得为了满足测试文字新增 LLM episode cache/reset 逻辑**；
- 检查 Memory 和 StreamVGGT 中没有 warmup episode 的 live entry；
- 不调用额外的 `torch.cuda.empty_cache()`；
- 然后执行：

```python
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
```

如果完整 warmup episode异常长，可以只执行两个真实 generation decisions，但必须随后可靠 reset env、Memory、StreamVGGT past KV 和 test logger；不能让 warmup 状态进入 measured manifest。

### 6.3 Evidence logger

Trace 必须和视频解耦，且覆盖 manifest 中所有 episode。逐步在内存中记录：

```text
scene_id
episode_id
instruction_sha256
physical_step
rgb_shape / rgb_dtype / rgb_sha256
generation_step
input_ids_shape / input_ids_dtype / input_ids_sha256
generated_text
parsed_actions
action_index
repeat_count
physical_action
stop_step
final metrics
```

约束：

- RGB 在任何 resize/processor 前 hash；
- `input_ids` 在 H2D 前对 contiguous CPU bytes hash，避免为证据引入 GPU sync；
- logger 不调用模型、不调用 env、不调用 RNG；
- 不在每个 step 写盘；
- 不依赖 `save_video`；
- timer 停止后一次性原子写出：

```text
evidence/<A1|B1>/
├── resolved_config.yaml
├── trace.json
├── input_hashes.json
├── results.json
├── profile.json
└── checkpoint_load_report.json
```

原生 `results_split_0.json` 也要保留。

### 6.4 同批低干扰 profiling

不再运行独立 profiler。Measured rollout 内记录：

- `rollout_wall_seconds`；
- `evidence_overhead_seconds`；
- `evidence_artifact_write_seconds`；
- `env_step_seconds`；
- `finalize_metrics_seconds`；
- 每次 generation 的 CPU end-to-end latency；
- 每次 generation 一对 CUDA Events；
- ToMe 调用次数和处理 frame 数；
- CUDA peak allocated/reserved；
- episode、generation decision 和 physical action 数；
- nvidia-smi 采样文件路径。

计时合同：

1. warmup 和 reset 完成；
2. `torch.cuda.synchronize()`；
3. 开始 `time.perf_counter()`；
4. measured rollout 中禁止 per-decision `torch.cuda.synchronize()`；
5. CUDA Events 只 record，不立即解析；
6. manifest 结束；
7. `torch.cuda.synchronize()` 一次；
8. 停止 wall timer；
9. 解析 deferred events；
10. timer 外写 artifact。

定义：

```text
core_rollout_seconds =
    rollout_wall_seconds
    - evidence_overhead_seconds
    - evidence_artifact_write_inside_rollout_seconds
```

通常 `evidence_artifact_write_inside_rollout_seconds` 应为 0。原生 evaluator 每 episode 的 result 写入仍属于实际 rollout，不得扣除。也不能扣除模型、Memory、StreamVGGT、Habitat step 或最终指标计算。

这组 timing 是工程筛选结果，不是论文级 benchmark；但它和 strict parity 来自同一批 rollout，不再重复计算导航轨迹。

### 6.5 本次必须修复的 harness reset 顺序

这是 stopped-run 唯一直接根因。`rollout_episode()` 必须遵守：

```python
if initial_observations is None:
    # 普通 episode：这里完成唯一一次 assignment + reset。
    env.current_episode = episode
    observations = env.reset()
    evaluator.reset()
else:
    # 首个 measured episode 已在 timer 前 assignment + reset。
    # 这里禁止再次赋值 env.current_episode，也禁止再次 reset。
    current_key = _episode_key(env.current_episode)
    expected_key = _episode_key(episode)
    if current_key != expected_key:
        raise RuntimeError(
            f"preloaded observation episode mismatch: "
            f"{current_key} != {expected_key}"
        )
    observations = initial_observations
```

外层继续保持：

```python
first_measured_episode = measured_episodes[0]
env.current_episode = first_measured_episode
first_measured_observations = env.reset()
evaluator.reset()
recorder.begin_measured()
```

硬合同：

- 每个 episode 在第一次 `env.step()` 前只能发生一次 `current_episode` assignment 和一次 `env.reset()`；
- 预加载 observation 分支只能验证 episode key，不能产生任何 Habitat state mutation；
- 不得通过捕获/忽略 Habitat assertion 继续运行。

同时修复即将出现的下一处 smoke 误判。Fast Habitat 的合同是：

- simulator sensors 精确只有 `rgb_sensor`；
- task observations 必须包含 `rgb`；
- task observations 不得包含 `depth`、`gps` 或 `compass`；
- Habitat 必需的 `instruction` observation 合法，不应要求 `observations.keys() == ["rgb"]`。

将旧检查：

```python
record["keys"] == ["rgb"]
```

改为等价的集合检查：

```python
keys = set(record["keys"])
observation_ok = (
    "rgb" in keys
    and not (keys & {"depth", "gps", "compass"})
)
```

在任何新模型进程前，给 test-only helper 增加并运行两个小回归测试：

1. `initial_observations` 分支不会再次 assignment/reset，随后第一步 `env.step()` 合法；
2. `{"rgb", "instruction"}` 通过 fast-observation 检查，而任一 depth/GPS/compass key 失败。

```bash
"$PYTHON_BIN" -m py_compile "$OUTPUT_ROOT/memvln_test_rollout.py"
"$PYTHON_BIN" "$OUTPUT_ROOT/test_memvln_test_rollout_regressions.py"
```

完成所有 test-only instrumentation、通过上述测试后，立即冻结 helper 和两个执行 source tree。之后不得再编辑 helper 或任一执行 tree：

```bash
"$PYTHON_BIN" - \
  "$OUTPUT_ROOT/memvln_test_rollout.py" \
  "$MEMVLN_REPO" \
  "$LEOR_REPO" \
  "$OUTPUT_ROOT/source/test_harness_hashes.env" <<'PY'
import hashlib
import os
import re
import sys
from pathlib import Path

helper_path = Path(sys.argv[1]).resolve()
memvln_root = Path(sys.argv[2]).resolve()
leor_root = Path(sys.argv[3]).resolve()
output_path = Path(sys.argv[4]).resolve()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def source_tree_digest(root: Path) -> str:
    excluded_parts = {
        ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    }
    excluded_suffixes = {".pyc", ".pyo"}
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in excluded_parts for part in relative.parts):
            continue
        if path.is_symlink():
            files.append((relative.as_posix(), path))
            continue
        if not path.is_file():
            continue
        if path.suffix in excluded_suffixes or (
            path.name == "core" or path.name.startswith("core.")
        ):
            continue
        files.append((relative.as_posix(), path))
    files.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    for relative, path in files:
        relative_bytes = relative.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        if path.is_symlink():
            target = os.readlink(path).encode("utf-8")
            digest.update(b"L")
            digest.update(len(target).to_bytes(8, "big"))
            digest.update(target)
            continue
        digest.update(b"F")
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()

values = {
    "MEMVLN_TEST_HELPER_SHA256": sha256_file(helper_path),
    "MEMVLN_TEST_MEMVLN_SOURCE_SHA256": source_tree_digest(memvln_root),
    "MEMVLN_TEST_LEOR_SOURCE_SHA256": source_tree_digest(leor_root),
}
assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in values.values())
output_path.write_text(
    "".join(f"export {key}={value}\n" for key, value in values.items()),
    encoding="utf-8",
)
print(output_path)
PY

source "$OUTPUT_ROOT/source/test_harness_hashes.env"
export STOPPED_HELPER_SHA256=22eaf8cc48a3e08bf78a9f6cb0cdf8b56444c72079a51c2c32f64cb82e46482a
test "$MEMVLN_TEST_HELPER_SHA256" != "$STOPPED_HELPER_SHA256"
```

这里的 source-tree 算法必须与 helper 内部 `_source_tree_digest()` 保持逐字等价；runner 会在模型构造前、rollout 结束后各验证一次。不得复用 stopped-run 的旧 helper SHA：

```text
22eaf8cc48a3e08bf78a9f6cb0cdf8b56444c72079a51c2c32f64cb82e46482a
```

## 7. 准备一个 3-episode B smoke

从正式 manifest 的前三个不同 scene 各取一个 episode，生成：

```text
$OUTPUT_ROOT/manifests/r2r_smoke_3.json
```

只在当前 MemVLN B 路径运行一次，确认：

- checkpoint gate 通过；
- StreamVGGT checkpoint 成功加载；
- clip id 连续；
- grounded cache 被写入和 episode-end 清空；
- `need_weights=False` 路径实际执行；
- simulator sensor 只有 RGB；task observation 包含 RGB、允许 `instruction`、不含 depth/GPS/compass；
- evidence/profile 五类 artifact 完整；
- 没有 OOM；
- 没有 video/debug env。

Smoke 的 timing 不进入最终性能报告。先生成 manifest；定义完第 8 节 runner 后再按第 8 节命令执行。Smoke 失败时不要启动 A1/B1。

## 8. A1/B1 runner

先启动一份低频 GPU telemetry；不要用 profiler 干扰每个 decision：

```bash
nvidia-smi dmon -s pucmt -d 5 -o DT > "$OUTPUT_ROOT/logs/nvidia_smi_dmon.txt" &
export NVIDIA_DMON_PID=$!
trap 'kill "$NVIDIA_DMON_PID" 2>/dev/null || true' EXIT
```

在 Bash 中定义公共 overrides：

```bash
test -f "$OUTPUT_ROOT/source/test_harness_hashes.env"
source "$OUTPUT_ROOT/source/test_harness_hashes.env"
test "$MEMVLN_TEST_HELPER_SHA256" != \
  22eaf8cc48a3e08bf78a9f6cb0cdf8b56444c72079a51c2c32f64cb82e46482a

COMMON_OVERRIDES=(
  "task=sft_r2r_rxr_v2_w_traj_sum_w_mem"
  "llm=qwen3-4b-mem"
  "vision3d=streamvggt"
  "llm.cfg_path=$QWEN_DIR"
  "vision3d.ckpt_path=$STREAMVGGT_CKPT"
  "llm.torch_dtype=bf16"
  "llm.attn_implementation=flash_attention_2"
  "data.vln.prompt_format=v2"
  "data.vln.memory_prompt=true"
  "data.vln.max_num_frames=16"
  "data.vln.obs_hist_sample=nearest"
  "data.vln.merge_action=true"
  "data.vln.max_act_hist=0"
  "data.vln.skip_merge_step_obs=false"
  "data.vln.vggt_cache_feat=true"
  "~task.vlnce.RxR"
  "task.vlnce.R2R.args.habitat_config_path=$R2R_HABITAT_CONFIG"
  "task.vlnce.R2R.args.split=val_unseen"
  "task.vlnce.R2R.args.save_video=0"
  "mode=eval"
  "num_gpu=1"
  "rng_seed=42"
)
```

定义 runner：

```bash
run_pair_member() {
  local variant="$1"  # B_SMOKE | A1 | B1 | B2 | A2
  local repo
  local source_sha
  local -a inference_overrides

  case "$variant" in
    A1|A2)
      repo="$LEOR_REPO"
      source_sha="$MEMVLN_TEST_LEOR_SOURCE_SHA256"
      inference_overrides=()
      ;;
    B_SMOKE|B1|B2)
      repo="$MEMVLN_REPO"
      source_sha="$MEMVLN_TEST_MEMVLN_SOURCE_SHA256"
      inference_overrides=(
        "inference.enabled=true"
        "inference.cache_grounded_memory=true"
        "inference.skip_memory_attention_weights=true"
        "inference.fast_habitat=true"
        "inference.empty_cache_on_episode_end=false"
        "inference.video_selection_seed=42"
      )
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      return 2
      ;;
  esac

  local tag="${TEST_BATCH}_memvln_r2r_${variant,,}"
  local evidence_dir="$OUTPUT_ROOT/evidence/$variant"
  local native_dir="$OUTPUT_ROOT/Qwen3-VLNCE_${tag}"
  local manifest_sha
  local full_manifest_sha
  manifest_sha="$(sha256sum "$TEST_MANIFEST" | awk '{print $1}')"
  full_manifest_sha="$(sha256sum "$FULL_TEST_MANIFEST" | awk '{print $1}')"
  [[ "$source_sha" =~ ^[0-9a-f]{64}$ ]]
  [[ "$MEMVLN_TEST_HELPER_SHA256" =~ ^[0-9a-f]{64}$ ]]
  test ! -e "$evidence_dir"
  test ! -e "$native_dir"
  mkdir -p "$evidence_dir"

  (
    cd "$repo"
    MEMVLN_TEST_MANIFEST="$TEST_MANIFEST" \
    MEMVLN_TEST_FULL_MANIFEST="$FULL_TEST_MANIFEST" \
    MEMVLN_TEST_EVIDENCE_DIR="$evidence_dir" \
    MEMVLN_TEST_VARIANT="$variant" \
    MEMVLN_TEST_PROFILE=1 \
    MEMVLN_TEST_HELPER_ROOT="$OUTPUT_ROOT" \
    MEMVLN_TEST_TELEMETRY_PATH="$OUTPUT_ROOT/logs/nvidia_smi_dmon.txt" \
    MEMVLN_NAV_SHA256="$MEMVLN_NAV_SHA256" \
    QWEN_DIR="$QWEN_DIR" \
    QWEN_DIR_SHA256="$QWEN_DIR_SHA256" \
    QWEN_CONFIG_SHA256="$QWEN_CONFIG_SHA256" \
    STREAMVGGT_CKPT="$STREAMVGGT_CKPT" \
    STREAMVGGT_SHA256="$STREAMVGGT_SHA256" \
    MEMVLN_TEST_SOURCE_SHA256="$source_sha" \
    MEMVLN_TEST_HELPER_SHA256="$MEMVLN_TEST_HELPER_SHA256" \
    MEMVLN_TEST_MANIFEST_SHA256="$manifest_sha" \
    MEMVLN_TEST_FULL_MANIFEST_SHA256="$full_manifest_sha" \
    PYTHONPATH="$repo" \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONHASHSEED=42 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    "$PYTHON_BIN" split_eval/run_eval_split.py \
      --config-name default \
      --model-path "$MEMVLN_NAV_CKPT" \
      --split-id 0 \
      --split-num 1 \
      --epoch 0 \
      "base_dir=$OUTPUT_ROOT" \
      "note=$tag" \
      "${COMMON_OVERRIDES[@]}" \
      "${inference_overrides[@]}"
  ) 2>&1 | tee "$OUTPUT_ROOT/logs/${variant}.log"
}
```

注意：

- A1/A2 不得接收当前 MemVLN 的 `inference.*` 参数。
- A1/A2 保留原始 evaluator 行为；只加入第 6 节 instrumentation。
- B1/B2 必须开启当前全部已接受优化。
- 原始 LEO-R 的 RePA 代码只在 training 中消费；不得为了本轮 inference 临时启用 RePA loss。
- 两边 loader 必须写 `checkpoint_load_report.json` 并通过第 4.2 节 gate。
- runner 打印的 resolved config 必须保存到 evidence 目录。

先执行 fail-fast smoke，再恢复正式 manifest并执行默认成功路径：

```bash
test "$(realpath "$OUTPUT_ROOT")" != "$(realpath "$PREVIOUS_RUN_ROOT")"
export FULL_TEST_MANIFEST="$TEST_MANIFEST"
export TEST_MANIFEST="$OUTPUT_ROOT/manifests/r2r_smoke_3.json"
run_pair_member B_SMOKE

"$PYTHON_BIN" - "$OUTPUT_ROOT/evidence/B_SMOKE/profile.json" <<'PY'
import json
import sys
from pathlib import Path

profile = json.loads(Path(sys.argv[1]).read_text())
assert profile["status"] == "complete", profile.get("status")
assert profile["counts"]["episodes"] == 3, profile["counts"]
assert all(profile["validation_checks"].values()), profile["validation_checks"]
assert all(profile["smoke_checks"].values()), profile["smoke_checks"]
assert profile["runtime"] == {
    "oom": False,
    "egl_errors": False,
    "cuda_errors": False,
    "recovery_count": 0,
    "partial_output": False,
    "validation_failed": False,
}, profile["runtime"]
print("Fresh B_SMOKE: PASS")
PY

export TEST_MANIFEST="$FULL_TEST_MANIFEST"
run_pair_member A1
run_pair_member B1
```

旧失败目录及其 `evidence/B_SMOKE/{trace,input_hashes,results,profile}.json` 均为 partial/failed，禁止复制到新的 active evidence、禁止 resume、禁止覆盖。不要在 A1/B1 之间修改 source、checkpoint、manifest、环境变量或 GPU power limit。

## 9. 一次 comparator 同时判 parity 和性能

远端 Codex在 `$OUTPUT_ROOT/compare_joint_runs.py` 创建临时 comparator。

Comparator 必须先做 artifact 完整性，再做 exact parity，最后才输出 timing：

1. source/checkpoint/config identity；
2. manifest SHA 和有序 episode keys；
3. result/trace/input/profile 的 episode key set；
4. 每个 episode 的 RGB physical-step coverage；
5. 每个 generation 的 `input_ids` coverage；
6. RGB hash；
7. `input_ids` hash；
8. generated text；
9. parser result和 repeat count；
10. physical action sequence；
11. STOP step；
12. steps；
13. SR、SPL、NE、OS；
14. 只有以上全 PASS，才计算 timing。

执行：

```bash
"$PYTHON_BIN" "$OUTPUT_ROOT/compare_joint_runs.py" \
  --reference "$OUTPUT_ROOT/evidence/A1" \
  --candidate "$OUTPUT_ROOT/evidence/B1" \
  --manifest "$TEST_MANIFEST" \
  --output "$OUTPUT_ROOT/joint_comparison_A1_B1.json"
```

最终控制台必须明确输出：

```text
STRICT_PARITY: PASS | FAIL
FIRST_MISMATCH: ...
A1_CORE_ROLLOUT_SECONDS: ...
B1_CORE_ROLLOUT_SECONDS: ...
SPEEDUP: ...
GAIN_PERCENT: ...
```

如果 parity FAIL，performance 状态必须是：

```text
INVALID (behavior mismatch)
```

不能因为最终 SR/SPL 相同就忽略中间 text/action mismatch。

## 10. 默认通过门槛

默认两-run 流程在以下条件同时满足时结束：

- A1/B1 strict parity 全 PASS；
- manifest 正好 `MANIFEST_SIZE` 个 episodes；默认 32，且至少 8 scenes；
- 至少 200 次 generation decisions；若不足则将 manifest 扩到 48，只重做 A1/B1；
- 没有 OOM、EGL recovery、partial output 或后台 GPU 任务；
- A1/B1 都没有 thermal throttling；
- CUDA clocks、power limit 和显存状态可比较；
- `gain_percent >= 10%`，或虽低于 10%但用户只要求 parity、不要求稳定速度结论。

满足时不要运行 REF-B、CACHE、ATTN、HABITAT、RxR、视频、120 episodes 或三次重复。

报告中的速度措辞必须是：

```text
single-pair engineering estimate on a fixed 32-episode multi-scene manifest
```

不要把一次 pair 写成统计稳健的 median。

## 11. 只有触发时才增加 run

### 11.1 Parity mismatch

如果 A1/B1 mismatch：

1. 先只跑 `A2`；
2. 比较 A1/A2；
3. 若 A1/A2 mismatch，判定 reference nondeterministic，停止性能结论；
4. 若 A1/A2 完全一致，说明 B1 行为改变；
5. 只在第一处失败 episode 所在的 3–8 episode diagnostic manifest 上依次定位：
   - current MemVLN reference：`enabled=false, fast_habitat=false`
   - cache-only
   - no-attention-weights-only
   - fast-Habitat-only
   - ALL
6. 找到第一个引入 mismatch 的开关后停止，不跑完整 32 episodes attribution。

单项矩阵只用于定位，不进入最终吞吐报告。

### 11.2 速度较小或可能受顺序影响

出现任一条件时，将 `A1 → B1` 扩为 ABBA：

- `gain_percent < 10%` 且需要明确是否真实提速；
- GPU clocks/temperature/power 显示 A1 与 B1 状态明显不同；
- decision p95 的变化与总 wall time方向冲突且差异超过 10%；
- 有明显 page-cache/order concern；
- 用户要求论文级或更正式的速度数字。

继续：

```bash
run_pair_member B2
run_pair_member A2
```

要求：

- A1/A2 strict parity；
- B1/B2 strict parity；
- A1/B1 与 A2/B2 strict parity；
- 使用配对几何均值：

```text
r1 = A1.core / B1.core
r2 = A2.core / B2.core
paired_speedup = sqrt(r1 * r2)
```

若 `max(r1, r2) / min(r1, r2) > 1.05`，报告环境噪声，不要自动无限追加 run。只有用户明确要求更严格 benchmark 时才扩展到更多 pairs。

## 12. 默认延后项目

### 12.1 RxR / nDTW

默认不运行 RxR。R2R PASS 不能证明 RxR nDTW 正确。

只有以下情况才运行一个 2–3 episode、多 scene 的 RxR A/B 专项：

- 本轮修改了 `finalize_lazy_measures()`、nDTW 或 RxR evaluator；
- baseline archive 中没有任何成功的 lazy-nDTW parity，而本次发布要求声明 RxR 已验证；
- 用户明确要求。

该专项只验证 text/action/metric/nDTW exact parity，不做性能统计，不扩大到 120 episodes。

### 12.2 视频

默认 `save_video=0`。Parity 通过后如果需要人工查看，只从已经保存的 B1 physical-action trace 中选一个 episode，在独立 debug env 重放；不得再次运行模型。

### 12.3 120 episodes

只有以下情况才扩展到 120：

- 32 episodes generation decisions 不足 200；
- 需要对外发布稳定吞吐数字；
- ABBA 的两个 paired ratio 相差超过 5%，且已排除 thermal/background load；
- 用户明确要求。

扩展时复用同一 manifest 生成算法和 joint evidence logger，不能恢复“parity 一套、timing 另一套”的重复流程。

## 13. 必须停止的情况

任何一项发生都停止并保存现有证据：

- checkpoint SHA 不符；
- 任一 reused P0/P1 artifact hash 与第 3 节固定值不符；
- reset-order 或 fast-observation regression test 失败；
- 使用了纯 Qwen-LoRA baseline checkpoint；
- Memory/merger/LoRA key coverage 不完整；
- 除 deprecated RePA 以外有 unexpected key；
- 两边 resolved config 的模型合同不同；
- FlashAttention 没有实际运行 CUDA kernel；
- manifest episode 缺失、重复、顺序不同或 resume；
- A/B 的原始 RGB 已经不同；
- prompt/input 中出现意外的 `FalseCurrent observation`；
- OOM、EGL error、CUDA error；
- evidence key set 不完整；
- text/action/STOP/metric mismatch；
- timing 包含模型加载、视频 replay，或使用 per-decision synchronize；
- 测试期间 source/checkpoint/manifest 被修改。

不得通过减小模型、改变 dtype、改输入帧、降低 Memory token、放宽 comparator 或随机重跑到“刚好相同”来绕过。

## 14. 最终报告

在 `$OUTPUT_ROOT/REPORT.md` 写：

```markdown
# MemVLN 4090 Fast Inference Validation

## Status

- Baseline historical result: REUSED / NOT PROVIDED
- Stopped-run P0/P1 prefix: REUSED VERIFIED PASS / RERUN
- Harness reset-order regressions: PASS / FAIL
- Fresh B_SMOKE: PASS / FAIL
- MemVLN checkpoint gate: PASS / FAIL
- A1/B1 strict parity: PASS / FAIL
- Performance validity: VALID / INVALID
- Overall: PASS / FAIL / BLOCKED

## Frozen inputs

- LEO-R source SHA:
- MemVLN source manifest SHA:
- MemVLN checkpoint path and SHA256:
- Qwen path/hash:
- StreamVGGT path/hash:
- R2R config/data/scene hashes:
- Manifest path/hash:
- Episode count:
- Scene count:
- Generation decision count:

## Exact parity

| Evidence | A1 vs B1 |
|---|---|
| Ordered episodes | |
| RGB hashes | |
| Input ID hashes | |
| Generated text | |
| Parser/repeat count | |
| Physical actions | |
| STOP step | |
| Steps | |
| SR/SPL/NE/OS | |

First mismatch:

## Joint timing

| Run | Episodes | Decisions | Physical actions | Core rollout s | Episode/hour | Decision CPU p50/p95 | Decision CUDA p50/p95 | Env step s | Finalize s | Evidence overhead s | Peak allocated/reserved |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| A1 original LEO-R | | | | | | | | | | | |
| B1 current MemVLN ALL | | | | | | | | | | | |

- Speedup:
- Gain percent:
- Timing class: single-pair engineering estimate / ABBA paired estimate
- Thermal/background-load notes:

## Conditional runs

- A2:
- B2:
- Diagnostic variants:
- RxR:
- Video replay:

## Test-only changes

- LEO-R instrumentation diff:
- MemVLN instrumentation diff:
- Comparator and manifest-builder hashes:
- Previous stopped-run root and reuse-verification hash:
- New helper hash after reset-order fix:
- Confirmed no commit/push:

## Limitations

- R2R parity does not verify RxR nDTW unless the optional专项 was run.
- Single-pair timing is an engineering estimate, not publication-grade statistics.
```

最后保存：

```bash
git -C "$LEOR_REPO" status --short > "$OUTPUT_ROOT/source/leor_status_after.txt" 2>&1 || true
git -C "$MEMVLN_REPO" status --short > "$OUTPUT_ROOT/source/memvln_status_after.txt" 2>&1 || true
nvidia-smi -q > "$OUTPUT_ROOT/source/nvidia_smi_q_after.txt"

find "$OUTPUT_ROOT" -type f \
  ! -name 'ALL_ARTIFACTS.sha256' \
  -print0 \
  | sort -z \
  | xargs -0 -r sha256sum \
  > "$OUTPUT_ROOT/ALL_ARTIFACTS.sha256"
```

不要 commit，不要 push。不要删除 partial failure artifact。最终把 `OUTPUT_ROOT` 复制到持久存储，并向用户报告绝对路径。
