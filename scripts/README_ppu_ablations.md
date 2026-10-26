# PPU component ablations

These standalone scripts provide optional joint R2R/RxR and R2R-only training
experiments. They use PPU-specific network and filesystem defaults. The paper
checkpoint's evaluation settings are in
[the release recipe](../README.md#evaluation).

## Training settings

- Initialize from Qwen3-VL-4B-Instruct and, for geometry variants, StreamVGGT;
  `pretrained_ckpt_path` is empty.
- One epoch, dummy LR decay off, seed 42, learning rate `2e-5`, eight GPUs,
  per-GPU batch size 1, gradient accumulation 1.
- Epoch sampling, sample ratios all 1, `training.max_updates=null`, and the
  epoch sampler's `warmup_steps=400` setting.
- Nearest 16 observations and stop augmentation disabled. The baseline keeps
  the Qwen visual tower and local observation history.
- Memory variants default to FIFO, a 100-frame bank, 100 tokens per frame,
  and 300 retrieved tokens. Navigation visual masking is 0; the Full model's
  summary visual masking is 1.
- Save every 5,000 training steps and retain five automatic checkpoints.
  Each run has a distinct timestamp/PID-based name. Logs default to
  `/workspace/outputs/logs/`.

An epoch covers the selected training data. Full includes additional summary
items, so one epoch does not imply equal optimizer-update budgets across model
variants. Record the actual update count and LR schedule when comparing runs.

## Scripts

G = geometry, M = episodic memory, S = trajectory summarization.

| Data | Model | Script | Default GPUs / port |
| --- | --- | --- | --- |
| R2R+RxR | Baseline | `train_ppu_joint_baseline.sh` | 8–15 / 3013 |
| R2R+RxR | Baseline+G | `train_ppu_joint_geometry.sh` | 0–7 / 3012 |
| R2R+RxR | Baseline+M | `train_ppu_joint_memory.sh` | 0–7 / 3012 |
| R2R+RxR | Baseline+G+M | `train_ppu_joint_geometry_memory.sh` | 8–15 / 3013 |
| R2R-only | Baseline | `train_ppu_r2r_baseline.sh` | 8–15 / 3013 |
| R2R-only | Baseline+G | `train_ppu_r2r_geometry.sh` | 8–15 / 3013 |
| R2R-only | Baseline+M | `train_ppu_r2r_memory.sh` | 8–15 / 3013 |
| R2R-only | Baseline+G+M | `train_ppu_r2r_geometry_memory.sh` | 8–15 / 3013 |
| R2R-only | Full (G+M+S) | `train_ppu_r2r_full.sh` | 8–15 / 3013 |

The joint Full/FIFO100 configuration is available through
[`train_ppu_fifo_nb100.sh`](train_ppu_fifo_nb100.sh); its settings are described
in the [FIFO guide](README_ppu_fifo.md).

## Launch and overrides

From the repository root, launch one selected experiment:

```bash
bash scripts/train_ppu_r2r_full.sh
```

Each script immediately starts a background process and returns its PID.
Scripts sharing GPUs or a port must run sequentially after the previous
training process exits. Chaining these background launchers with `&&` does not
queue the training jobs.

Select another GPU group or port:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 DIST_PORT=3012 bash scripts/train_ppu_r2r_full.sh
```

Select FRME or a different epoch budget:

```bash
CONSOLIDATION=frme bash scripts/train_ppu_r2r_full.sh
EPOCHS=2 CONSOLIDATION=frme bash scripts/train_ppu_joint_geometry_memory.sh
```

`CONSOLIDATION` is supported by the memory, geometry-memory, and Full scripts.
Use the same consolidation for all memory variants in a component comparison.
`EPOCHS` is also supported by the baseline and geometry-only scripts. Compare
independently trained runs with matching schedules: an intermediate checkpoint
from a two-epoch schedule is not equivalent to training for one epoch.

The nine scripts support `CODE_DIR`, `QWEN3_PATH`, `STREAMVGGT_PATH` (geometry
variants), `DATA_BASE`, `OUTPUT_BASE`, `GPU_IDS`, `DIST_PORT`, `EPOCHS`, and
`SEED` environment overrides. The Python launcher assembles a shell command,
so use paths without spaces or shell metacharacters.

`DATA_BASE` must be the parent of `vln/`; annotation paths resolve under
`$DATA_BASE/vln`. The supplied default `/workspace/data/leo2_data/vln` therefore
expects a second `vln/` directory. Set `DATA_BASE` to match your layout.
Geometry-free variants do not require a StreamVGGT checkpoint; R2R-only
variants do not require RxR annotations. Set `NCCL_SOCKET_IFNAME` in the script
to an available interface if the container does not expose `bond0`.

## Result records

Record the training domain, model components, consolidation, frame capacity,
epoch budget, dummy LR setting, seed, actual optimizer updates, checkpoint,
and evaluation split. Use the same checkpoint-selection rule within each
comparison and report joint-data and R2R-only results separately.
