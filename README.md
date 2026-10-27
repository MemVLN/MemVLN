<div align="center">

# MemVLN: Streaming Episodic Memory for Vision-Language Navigation

**Persistent, compact navigation memory for monocular streaming video**

**[Project](https://memvln.github.io/) · [Checkpoint](https://huggingface.co/Yannnnnnnn/MemVLN) · [Data](https://huggingface.co/datasets/Yannnnnnnn/MemVLN-data) · [Installation](#installation) · [Evaluation](#evaluation)**

</div>

<p align="center">
  <img src="assets/readme/teaser.png" width="100%" alt="MemVLN overview">
</p>

MemVLN is a streaming episodic memory framework for vision-language
navigation. It augments a VLM navigator with a bounded memory bank that stores
compressed perceptual features together with temporal and odometric context.
A memory perceiver retrieves a fixed number of memory tokens, allowing the
policy to combine long-horizon route history with the current observation and
local visual context.

The current implementation uses Qwen3-VL-4B as the VLM, StreamVGGT as the
geometry encoder, and a trainable memory module with trajectory-summary
supervision.

## Highlights

- **Streaming episodic memory.** Navigation observations are progressively
  written to a persistent episode-level memory bank.
- **Bounded history.** Similar consecutive frames and spatial tokens are
  consolidated to keep memory cost controlled over long trajectories.
- **Spatio-temporal retrieval.** Fourier position features, temporal
  embeddings, and cross-attention produce 300 compact memory tokens.
- **Memory-focused supervision.** Navigation training is combined with masked,
  multi-granularity trajectory summarization derived from the same R2R and RxR
  demonstrations.
- **Simulation and real-world evaluation.** The manuscript evaluates MemVLN on
  R2R/RxR VLN-CE and on a Unitree G1 robot.

## Method

<p align="center">
  <img src="assets/readme/method.png" width="100%" alt="MemVLN method">
</p>

At every navigation step, MemVLN:

1. encodes the current RGB observation and geometry-aware StreamVGGT features;
2. appends the observation feature, timestep, and accumulated odometry to the
   episode memory;
3. consolidates the memory when its capacity is reached; and
4. retrieves memory tokens that condition autoregressive action generation.

The default action space is `STOP`, `MOVE_FORWARD`, `TURN_LEFT`, and
`TURN_RIGHT`. Consecutive actions are merged into short text commands during
training and expanded back into atomic Habitat actions during evaluation.

## Results

The paper reports the following results. NE is measured in meters; the other
metrics are percentages.

| Dataset | Split | NE ↓ | OSR ↑ | SR ↑ | SPL ↑ | nDTW ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| R2R VLN-CE | Val-Unseen | 5.19 | 61.2 | 55.0 | 50.5 | — |
| RxR VLN-CE | Val-Unseen | 5.59 | — | 55.4 | 48.4 | 64.6 |

The navigation model is trained on the official R2R and RxR training
trajectories: 10,819 R2R demonstrations and 19,996 RxR demonstrations.
R2R sub-trajectory supervision additionally uses the
[FGR2R fine-grained annotations](https://github.com/YicongHong/Fine-Grained-R2R),
which align sub-instructions with intermediate waypoints on the original R2R
routes. Training starts from pretrained Qwen3-VL and StreamVGGT weights.

For the real-world study, each variant is evaluated on six held-out
instructions with five runs per instruction:

| Method | Runs | Successes | SR ↑ | Collision rate ↓ |
| --- | ---: | ---: | ---: | ---: |
| Base VLM | 30 | 5 | 16.7 | 56.7 |
| + Memory | 30 | 9 | 30.0 | 66.7 |
| MemVLN | 30 | 12 | 40.0 | 46.7 |
| MemVLN-PT (24 demonstrations) | 30 | 14 | 46.7 | 46.7 |

<p align="center">
  <img src="assets/readme/analysis.png" width="100%" alt="MemVLN analysis">
</p>

## Release status

| Artifact | Status |
| --- | --- |
| Training and evaluation code | Available in this repository |
| Real-world G1/Go2 deployment code | Available under `realworld/` |
| Project page, paper, and supplement | [Project page](https://memvln.github.io/) |
| Navigation checkpoint | [Hugging Face](https://huggingface.co/Yannnnnnnn/MemVLN), 2.45 GiB |
| R2R/RxR data, processed FGR2R annotations, and real-world recordings | [Hugging Face](https://huggingface.co/datasets/Yannnnnnnn/MemVLN-data), 71.77 GiB |

The checkpoint is publicly downloadable. Dataset downloads require accepting
the Matterport3D agreement on the dataset page and logging in with
`hf auth login`; access is automatically granted after completing the form.
File revisions and checksums are recorded in
[release/artifacts.json](release/artifacts.json).

## Installation

Use Python 3.9 and Habitat 0.2.4. Direct Python dependencies are pinned in
`requirements.txt`. Install a PyTorch build compatible with the CUDA driver
on your machine before compiling FlashAttention.

```bash
git clone https://github.com/MemVLN/MemVLN.git
cd MemVLN

conda create -n memvln python=3.9 ffmpeg=4.4 -c conda-forge
conda activate memvln

# PyTorch reference setup; change the CUDA wheel if required by your machine.
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118

# Habitat-Sim and Habitat-Lab 0.2.4.
conda install habitat-sim==0.2.4 headless withbullet \
  -c conda-forge -c aihabitat
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
pip install -e habitat-lab/habitat-lab
pip install -e habitat-lab/habitat-baselines

pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

The commands above provide a reference installation. The exact historical
PyTorch/CUDA/FlashAttention build is unavailable; record these versions with
the resolved config when running on a new machine.

## Pretrained models

The navigation checkpoint contains trainable MemVLN parameters and the token
embedding matrix. Download the frozen Qwen and StreamVGGT weights separately.
Run the following commands from the MemVLN repository root:

```bash
# Compatible with the Transformers version in this repository.
pip install 'huggingface_hub[cli]>=0.34,<1.0'
hf auth login

hf download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --local-dir ./pretrained/Qwen3-VL-4B-Instruct

hf download lch01/StreamVGGT checkpoints.pth \
  --revision f9ba55b1955bc4f34337c51142158a9aa2862c7f \
  --local-dir ./pretrained/StreamVGGT

hf download Yannnnnnnn/MemVLN memvln.safetensors \
  --revision 9d6c106ba1737aa4d06bfb939a0f4c20a15ccecd \
  --local-dir ./checkpoints/MemVLN

shasum -a 256 -c release/MODEL_SHA256SUMS
```

These immutable revisions pin the complete Qwen processor/configuration and
weights, and the official StreamVGGT PyTorch state dict. The upstream weight
hashes have remained unchanged since their first uploads. The reference
evaluation names its local StreamVGGT file `StreamVGGT-1B.pth`; the official
download is named `checkpoints.pth`. Use the latter path directly, without
conversion. See [base-model provenance](release/BASE_MODELS.md) for the file
hashes and the boundary of the historical-file verification.

## Data

MemVLN uses [R2R](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R),
[RxR](https://github.com/google-research-datasets/RxR), and
[Matterport3D](https://github.com/niessner/Matterport). The processed archive
includes the training observations/annotations, validation episodes, RxR GT,
and 24 real-world demonstration episodes. Download and extract it from the
repository root after obtaining access to the dataset repository:

```bash
hf download Yannnnnnnn/MemVLN-data vlnce_data.zip \
  --repo-type dataset \
  --revision 2a97637c79aeef17526e8dde96334f44c4adbc65 \
  --local-dir ./downloads/MemVLN-data
shasum -a 256 -c release/DATA_SHA256SUMS
mkdir -p ./datasets
unzip ./downloads/MemVLN-data/vlnce_data.zip -d ./datasets
```

The ZIP expands into `datasets/vln/`; allow about 80 GiB free for extraction
in addition to the 71.77 GiB download. Matterport3D scene meshes are obtained
separately from the official source under its access terms and are required
for Habitat evaluation. See the [dataset card](https://huggingface.co/datasets/Yannnnnnnn/MemVLN-data)
for the full archive inventory, attribution, and component-specific terms.

The cached training observations and annotations are read from
`${data.annotation_base}/vln`:

```text
${annotation_base}/vln/
├── r2r/
│   └── train/
│       ├── train_sub.json       # R2R + FGR2R sub-trajectory alignment
│       ├── train.json.gz        # fallback without sub-trajectory supervision
│       ├── train_gt.json.gz
│       └── traj/
│           └── <trajectory_id>/
│               └── <frame_id>.png
└── rxr/
    └── train_v2/
        ├── annotations.json
        └── <trajectory_id>/
            └── rgb/
                └── <frame_id>.jpg
```

`train_v2` is the inherited processed-cache directory name. The dataset,
configuration keys, and Python interfaces use the unversioned name `RxR`.

Set the root at launch time:

```text
data.annotation_base=/path/to/annotations
```

Alternatively, set `MEMVLN_DATA_ROOT`; the local fallback is `./datasets`.

Real-world recordings use a separate root, configured through
`data.vln.realworld.root_dirs`. Each `episode_*` directory contains
`trajectory_log.txt` (JSONL) and the images named by its `image_filename`
fields. Preserve these relative paths when extracting the data:

```text
data.vln.realworld.root_dirs=[/path/to/realworld]
```

The real-world post-training tasks are
`posttrain_realworld_weighted_bc` and
`posttrain_realworld_reward_weighted_bc`. The loader uses the recorded
executed actions, instructions, and poses; it expects each episode to end
with a stop action. See the [recording guide](realworld/readme.md#runtime-interaction)
for the robot clients' save flow.

Habitat evaluation additionally requires the Matterport3D scenes, R2R/RxR
VLN-CE episode files, and the RxR reference trajectories used for nDTW.
Configure their locations through these environment variables:

| Variable | Contents under this directory |
| --- | --- |
| `MEMVLN_SCENES_DIR` | Scene files addressed by the episode `scene_id` values |
| `MEMVLN_R2R_DATA_ROOT` | `{split}/{split}.json.gz` |
| `MEMVLN_RXR_DATA_ROOT` | `{split}/{split}_guide_en.json.gz` |
| `MEMVLN_RXR_GT_ROOT` | `{split}/{split}_guide_gt.json.gz` |

The RxR episode and GT roots can be different directories. Defaults use
`./datasets/VLN-CE` and its `R2R_VLNCE_v1-3_preprocessed` or `RxR` subdirectory.
For a different file layout, edit `habitat.dataset.data_path` in
`configs/habitat/r2r.yaml` or `configs/habitat/rxr.yaml`, and
`habitat.task.measurements.ndtw.gt_path` in the RxR config. The nDTW path accepts
an exact filename or a `{split}` template; the evaluator supplies the selected
split.

The complete training-cache generation pipeline is not included. Training
requires the annotations and observation layout shown above. Matterport3D
scenes and derived data remain subject to the
[Matterport3D terms](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf), which
require an agreement link with published derived material and a click-through
agreement for distribution of a substantial portion of the data. Obtain scenes
through the official access process; any released observation cache must retain
the applicable access conditions.

Included inspection and conversion utilities:

```bash
# Convert an RxR guide split to the R2R-compatible English evaluation schema.
python habitat_extensions/rxr_conversion.py \
  /path/to/train_guide.json.gz \
  /path/to/train_guide_en.json.gz \
  --language en

# Inspect action and sequence lengths before changing context limits.
python habitat_extensions/stat_vln_action_lengths.py \
  --anno-dir /path/to/annotations/vln \
  --split train \
  --datasets r2r rxr
```

Run each utility with `--help` for its complete contract. Existing outputs are
not overwritten unless `--overwrite` is explicitly passed.

## Training

`configs/default.yaml` selects joint R2R/RxR training with trajectory
summarization and FRME memory. The main settings are:

| Setting | Value |
| --- | --- |
| Task mixture | R2R + RxR navigation and trajectory summarization |
| GPUs | 8 |
| Per-GPU batch size | 1 |
| Gradient accumulation | 1 |
| Epochs | 1 |
| Learning rate | `2e-5` |
| Dummy LR decay | enabled (`task.training.dummy_lr_decay=true`) |
| Memory consolidation | FRME, 100 frames, 100 tokens per frame |
| Memory tokens | 300 |
| Observation history | nearest 16 frames |
| Prompt structure | memory + labeled historical/current observations |
| StreamVGGT cached features | enabled |

Additional defaults:

| Setting | Default |
| --- | --- |
| Memory-module parameter dtype | BF16 |
| Checkpoints retained | 5 |
| Fast Habitat pipeline | enabled in `default`; disabled in `eval_release` |

The memory dtype and checkpoint-retention controls are explicit settings in
this codebase; they were not separate fields in the original training config.

Launch training from the repository root. The command disables W&B; to log
online, set `WANDB_MODE=online` and explicitly add
`logger.entity=<your-entity>`.

```bash
RUN_NOTE="memvln_train_$(date +%Y%m%d_%H%M%S)_$$"

WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
python launch.py \
  --mode accelerate \
  --strategy ddp \
  --gpu_per_node 8 \
  --cpu_per_task 16 \
  --port 3012 \
  '~task.vlnce' \
  llm.cfg_path="$PWD/pretrained/Qwen3-VL-4B-Instruct" \
  vision3d.ckpt_path="$PWD/pretrained/StreamVGGT/checkpoints.pth" \
  data.annotation_base="$PWD/datasets" \
  base_dir=/path/to/outputs \
  note="$RUN_NOTE"
```

`~task.vlnce` disables Habitat rollout during training, which is useful on
machines without an NVIDIA EGL/OpenGL evaluation environment. It does not
change the model or training loss.

`data.annotation_base` must point to the parent of the `vln/` directory. For
example, when the data lives at `/workspace/data/memvln_data/vln`, set
`data.annotation_base=/workspace/data/memvln_data` because the config appends
`/vln`.

Optional experimental launchers are documented in
[PPU component ablations](scripts/README_ppu_ablations.md),
[FIFO experiments](scripts/README_ppu_fifo.md), and
[probabilistic episode sampling](docs/probabilistic_episode_sampling.md).
These recipes have their own training settings; use `eval_release` for the
paper checkpoint.

### Memory-module parameter dtype

The memory-module parameter and floating-buffer dtype is independently
configurable:

```text
llm.mem_module.param_dtype=bf16   # current reproduction default
llm.mem_module.param_dtype=fp32   # explicit FP32 ablation
llm.mem_module.param_dtype=fp16   # explicit FP16 ablation
```

Only memory-module parameters and floating-point buffers are converted. LoRA
and StreamVGGT merger parameters retain their own construction dtypes.

### Checkpoint retention

`training.checkpoint_total_limit` controls Accelerate's automatic checkpoint
retention and defaults to 5. Checkpoint directories use the full numeric
suffix, for example `checkpoint_9`, `checkpoint_10`, and `checkpoint_33`.

The main trainer accepts an exact model file, a checkpoint directory
containing exactly one of the files below, an Accelerate `checkpoints/`
directory, or an experiment root containing `checkpoints/`:

```text
model.safetensors
pytorch_model.bin
```

For the latter two forms it selects the largest full numeric checkpoint
suffix. For example, if the available directories are `checkpoint_9`,
`checkpoint_10`, and `checkpoint_33`, it selects `checkpoint_33`.

Full-state mid-epoch resume with the custom streaming sampler is not claimed
to be bit-exact because sampler progress and episode memory are not fully
contained in the checkpoint. Use released checkpoints for evaluation or
model-only initialization unless a run was saved at an epoch boundary.

## Evaluation

Use `configs/eval_release.yaml` for the paper's FRME checkpoint. This preset
matches the server evaluation settings: BF16, FlashAttention 2, nearest 16
observations, FRME with a 100-frame bank and 300 memory tokens, grounded-memory
caching, skipped attention weights, `fast_habitat=false`, and no video output.
It evaluates both R2R and RxR Val-Unseen with seed 42 on eight shards.

Use the same navigation checkpoint, Qwen weights, StreamVGGT weights, scene
data, episode files, and RxR GT as the reference evaluation. Choose a new
output name for each run because the evaluator resumes existing result
files by skipping completed episodes.

```bash
EVAL_NOTE="release_frme_$(date +%Y%m%d_%H%M%S)_$$"

export MEMVLN_SCENES_DIR=/path/to/scene_datasets
export MEMVLN_R2R_DATA_ROOT="$PWD/datasets/vln/r2r"
export MEMVLN_RXR_DATA_ROOT="$PWD/datasets/vln/rxr"
export MEMVLN_RXR_GT_ROOT="$PWD/datasets/vln/rxr"

bash split_eval/launch_eval_split.sh \
  --gpus 8 \
  --config eval_release \
  --model "$PWD/checkpoints/MemVLN/memvln.safetensors" \
  --epoch 0 \
  base_dir="$PWD/outputs" \
  note="$EVAL_NOTE" \
  llm.cfg_path="$PWD/pretrained/Qwen3-VL-4B-Instruct" \
  vision3d.ckpt_path="$PWD/pretrained/StreamVGGT/checkpoints.pth"
```

For R2R alone, append `'~task.vlnce.RxR'`; for RxR alone, append
`'~task.vlnce.R2R'`.

The general `default` config enables `fast_habitat=true`. Use the release
preset when comparing against the reference checkpoint results. FIFO remains
available for memory ablations; the paper evaluation preset selects FRME.

Merge all eight shards and check the complete episode counts:

```bash
python split_eval/merge_results.py \
  --dir "$PWD/outputs/Qwen3-VLNCE_${EVAL_NOTE}/eval_results/R2R/epoch_0" \
  --expected-splits 8 --expected-episodes 1839
python split_eval/merge_results.py \
  --dir "$PWD/outputs/Qwen3-VLNCE_${EVAL_NOTE}/eval_results/RxR/epoch_0" \
  --expected-splits 8 --expected-episodes 3669
```

The release evaluates the English RxR guide episodes. Its GT file includes
all guide languages; the evaluator selects GT entries by episode ID.

The split evaluator currently expects the exact checkpoint file path rather
than a checkpoint directory. It loads non-strictly for legacy compatibility,
so inspect the reported missing/unexpected keys against the checkpoint's
documented contents. A checkpoint containing only trainable parameters also
requires the matching frozen Qwen and StreamVGGT weights.

## Real-world deployment

The manuscript evaluates MemVLN on Unitree G1. The repository also contains an
experimental Go2 client. See [realworld/readme.md](realworld/readme.md) for the
inference server, camera/odometry interfaces, and robot-side commands.

Real-robot execution requires an operator-controlled emergency stop and a
clear operating area.

## Repository structure

```text
configs/                  Hydra training, model, data, and Habitat configs
data/                     R2R/RxR datasets, collation, and streaming sampler
model/memory/             Episodic memory bank, retrieval, and consolidation
model/streamvggt/         Streaming geometry encoder integration
model/qwen/               Qwen3-VL navigation and memory integration
trainer/                  SFT training and checkpoint management
evaluator/                Habitat VLN-CE evaluators
split_eval/               Multi-GPU sharded evaluation and result merging
realworld/                G1/Go2 server and robot clients
tests/                    Lightweight regression and contract tests
```

## Tests

The test suite does not require Matterport3D or a full Habitat rollout:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python -m unittest discover -s tests -p 'test_*.py'
```

These tests cover configuration, data handling, and model-independent
contracts; benchmark metrics require the model and Habitat environment.

## Limitations

- Similarity-based memory consolidation can discard observations that later
  become relevant.
- The current memory supervision is centered on trajectory summarization;
  broader memory-centric objectives remain open.
- Real-world collision avoidance and fine-grained landmark grounding remain
  significant limitations.
- The exact historical software build was not fully preserved, so new runs
  should record PyTorch, CUDA, FlashAttention, base-model revisions, data
  hashes, and the resolved Hydra config.

## Citation

The arXiv identifier will be added after the preprint is available.

```bibtex
@misc{wang2026memvln,
  title  = {MemVLN: Streaming Episodic Memory for Vision-Language Navigation},
  author = {Wang, Yan and Huang, Jiangyong and Huang, Siyuan and Jia, Baoxiong},
  year   = {2026}
}
```

## Acknowledgements

This codebase builds on
[LEO](https://github.com/embodied-generalist/embodied-generalist),
[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct),
[StreamVGGT](https://github.com/wzzheng/StreamVGGT),
[Habitat-Lab](https://github.com/facebookresearch/habitat-lab),
[StreamVLN](https://github.com/InternRobotics/StreamVLN), R2R, FGR2R,
and RxR.
Please follow the licenses and terms of each upstream project and dataset.

## License

The authors' original checkpoint, real-world data, and processing
contributions use [CC BY-NC 4.0](release/LICENSE-MODEL-DATA.md). The model and
dataset cards retain the component-specific upstream terms.

See [LICENSE](LICENSE) for the inherited LEO MIT license and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for included components and
their license texts. StreamVGGT-derived code is subject to CC BY-NC-SA 4.0,
and the ToMe utility to CC BY-NC 4.0. Model weights and datasets retain their
own licenses and access terms.
