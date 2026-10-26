# PPU FIFO experiments

These optional eight-GPU scripts train the joint R2R/RxR model with trajectory
summarization and FIFO memory. The paper checkpoint uses FRME; see the
[release evaluation recipe](../README.md#evaluation).

| Script | Frame capacity | Dummy LR decay | GPUs | Port |
| --- | ---: | --- | --- | ---: |
| `train_ppu_fifo_nb50.sh` | 50 | on | 0–7 | 3012 |
| `train_ppu_fifo_nb100.sh` | 100 | off | 8–15 | 3013 |

Both use one epoch, seed 42, BF16, and the model defaults in
`configs/default.yaml`, with Habitat evaluation disabled during training.
Capacity and dummy LR decay both differ between these scripts. Compare each
against an FRME run with the same capacity, schedule, and checkpoint-selection
rule; the two FIFO runs alone do not isolate a capacity effect.

## Configure and launch

Edit the path, device, port, and network settings at the top of each script
for your PPU environment. These two scripts use literal settings rather than
environment-variable overrides. In particular:

- `CODE_DIR` points to the repository root.
- `QWEN3_PATH` and `STREAMVGGT_PATH` point to the pretrained model files.
- `DATA_BASE` is the parent of the `vln/` directory. The scripts look for
  `$DATA_BASE/vln/r2r/...` and `$DATA_BASE/vln/rxr/...`. Their supplied
  `DATA_BASE=/workspace/data/leo2_data/vln` therefore resolves to a second
  `vln/` level; adjust it to your actual data layout.
- `OUTPUT_BASE` defaults to `/workspace/outputs`; logs are in its `logs/`
  subdirectory.
- `NCCL_SOCKET_IFNAME=bond0` must name an interface available in the container.

From the repository root:

```bash
bash scripts/train_ppu_fifo_nb50.sh
# Or, on its configured GPU group:
bash scripts/train_ppu_fifo_nb100.sh
```

Each command starts a background `nohup` process and prints its PID and log
path. Confirm that its GPUs and port are available before launching, then use
the log to check training progress and completion. W&B and model access are
configured for offline operation.
