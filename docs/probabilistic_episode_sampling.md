# Rank-local probabilistic episode sampling

This is an optional training recipe. The default training config uses epoch
sampling; the paper checkpoint is evaluated with
[`eval_release`](../configs/eval_release.yaml).

The opt-in `probabilistic_episode` sampler maintains one independent training
stream per GPU/rank. Each time its current episode finishes, that rank:

1. Selects R2R or RxR with the configured probabilities (default: 0.5 each).
2. Samples one episode uniformly **with replacement** from the selected dataset.
3. Consumes all of that episode's navigation and trajectory-summary items in
   temporal order, with navigation before summary at a shared clip ID.
4. Repeats until the fixed optimizer-update budget is reached.

There is no fixed allocation of ranks to datasets, no cross-rank episode
partition, and no epoch-level shuffle queue. Different ranks may draw the same
source episode; these are independent draws. Within each rank, repeated draws
have distinct occurrence IDs so both memory and StreamVGGT start fresh. Within
one draw, nav and summary share the same occurrence ID and history.

The 0.5 probabilities describe **episode starts**, not steps, tokens, or losses.
Longer episodes contribute more training items. Summary is attached to the
selected navigation episode, not sampled as another dataset.

## Enable with the existing launcher

The PPU launch script is `scripts/train_ppu_probabilistic_sampling.sh`. It uses
the established 8-GPU DDP/bf16 launcher, `bond0`/IPv4 socket settings, offline
variables and `/workspace/outputs/logs/` output layout. It trains Full with
FRME/Nb100, dummy LR off, and 0.5/0.5 episode probabilities from the pretrained
backbone. Run it in the PPU container with:

```bash
bash /workspace/MemVLN/scripts/train_ppu_probabilistic_sampling.sh
```

The script defaults to `MAX_UPDATES=172330`. With eight ranks, batch size 1,
and accumulation 1, this corresponds to 1,378,640 global item consumptions.
Set this budget from the actual optimizer-update count of the epoch-based
run you want to compare against: whole-episode rank assignment and trimming
can make that count differ from the raw dataset size divided by world size.
Equal consumption budgets do not guarantee equal episode coverage. The script
sets `WARMUP_UPDATES=50` to express an eight-rank, 400-tick epoch-sampler warmup
in optimizer-update units; see the LR section.

Keep the existing model, paths and launch options. Add these overrides to the
full R2R+RxR task (replace `YOUR_UPDATE_COUNT` with the actual positive integer
optimizer-update budget):

```text
task=sft_r2r_rxr_w_traj_sum
task.dataset_mix.VLNDatasetMix.sampling.mode=probabilistic_episode
task.dataset_mix.VLNDatasetMix.sampling.dataset_probabilities.r2r=0.5
task.dataset_mix.VLNDatasetMix.sampling.dataset_probabilities.rxr=0.5
task.dataset_mix.VLNDatasetMix.ratio_unit=episode
task.dataset_mix.VLNDatasetMix.ratio=[1,1,1,1]
task.training.dummy_lr_decay=false
training.max_updates=YOUR_UPDATE_COUNT
dataloader.train.batchsize=1
```

`sft_r2r_rxr` without summary uses two ratio entries, `[1,1]`. The single-domain
R2R tasks also support this sampler, with R2R probability 1.0. Defaults remain
`sampling.mode=epoch` and `training.max_updates=null`; existing training recipes
keep their previous behavior.

The new mode requires a positive integer `max_updates`, one sample per rank per
microbatch, positive gradient accumulation, all dataset multipliers equal to 1,
and dummy LR disabled. Probabilities must be finite, nonnegative, sum to one,
and name exactly the loaded nonempty domains. It supports ordinary DDP and
single-process training with bf16 or no mixed precision. Unsupported combinations
fail explicitly rather than silently using the old sampler.

## Update budget and learning rate

With `T=training.max_updates` and `G=gradient_accumulation_steps`, every rank
consumes exactly `T * G` microbatches. `task.training.epochs` does not determine
the duration in this mode. The scheduler advances once per successful optimizer
update; `training.schedule.args.warmup_steps` is also measured in **optimizer
updates**. The warmup must be an integer from zero through `T-1`.

The old custom-sampler path retains its historical Accelerate scheduler
behavior: the scheduler can advance once per world rank at each update.
For example, under that 8-rank DDP behavior, old `warmup_steps=400` spans roughly
50 optimizer updates; new `warmup_steps=400` spans 400 updates. To compare against
an old run, recover its actual update count and LR trace and express the warmup
in the new units (50 in this example). Do not infer the update count from a
checkpoint filename or assume unchanged numeric warmup means the same LR curve.

`training.save_interval` retains its existing microbatch unit. Checkpoints are
written only after a complete optimizer update. In-training evaluation is
deferred until the fixed budget finishes because evaluation would overwrite
the currently active navigation memory/KV caches.

Only the final budget boundary may truncate an episode. All ranks stop at the
same update count; finishing their last episodes separately would give different
DDP lengths. No extra updates are added to finish the final episode or summary.

## Checkpoints and consumption accounting

DataLoader workers may prefetch future samples. The sampler keeps dispatch
state separate from **trainer-confirmed consumption**, so prefetched episodes
are never recorded as trained. At each full-state checkpoint, every rank writes
`sampler_rank_<rank>.pt` into that exact checkpoint directory. It records its
random-generator state, active episode, consumed item count, and per-domain
started/completed episode and item counts. All rank files are required for resume;
run identity, probability, data-index manifest, budget, LR settings and world size
must match. Model-only checkpoints remain usable as initialization, not as an
exact training-state resume.

The online memory and StreamVGGT KV caches are not part of model weights.
Therefore **a resumed unfinished episode restarts at its first item** to rebuild
history. That replay uses the remaining update budget; previous updates and LR
progress are not rolled back. Its replayed-item count is recorded. This is a
documented restart policy, not bitwise-equivalent mid-episode continuation.
An episode-boundary resume retains the future episode draw sequence.

Consumption metrics are gathered at checkpoints and training end. The main-rank
experiment logger writes both `train/sampling/rank_<rank>/...` counters and
`train/sampling/global/...` sums, including items and started/completed episodes
per domain. Sampler sidecars also retain every rank's counts. A rank's final
`unfinished_episode=1` indicates the budget cut an active episode short.

## Tests

```bash
python -m unittest discover -s tests -p 'test_probabilistic_episode*.py' -v
python -m unittest discover -s tests -p test_fixed_update_training.py -v
python -m unittest discover -s tests -p test_episode_dataset_mix.py -v
```

For the real two-process CPU DDP test, enable local Gloo networking:

```bash
MEMVLN_RUN_DDP_TESTS=1 python -m unittest discover -s tests -p test_probabilistic_episode_sampler.py -v
```

On macOS, `GLOO_SOCKET_IFNAME=lo0` may be needed for the local loopback test.
This test setting is unrelated to the PPU training network configuration.
These tests use small CPU models and synthetic episodes; they do not run a full
Qwen/StreamVGGT navigation training job or validate PPU/NCCL behavior.
