# Base-model provenance

The release downloads the official Qwen3-VL-4B-Instruct and StreamVGGT weights
at immutable Hub commits. Full file metadata is in [artifacts.json](artifacts.json).

| Dependency | Pinned revision | Weight format |
| --- | --- | --- |
| [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17) | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | Two safetensors shards, with the matching tokenizer, processor and configuration |
| [lch01/StreamVGGT](https://huggingface.co/lch01/StreamVGGT/tree/f9ba55b1955bc4f34337c51142158a9aa2862c7f) | `f9ba55b1955bc4f34337c51142158a9aa2862c7f` | `checkpoints.pth`, a PyTorch state dictionary |

## Verification on 2026-10-26

- All twelve Qwen runtime files match their first complete upload on
  2025-10-11, across all eight subsequent-or-initial populated revisions.
- StreamVGGT `checkpoints.pth` has the same SHA256 as its first upload on
  2025-07-17. Later repository changes added a safetensors serialization and
  updated the model card; the PyTorch checkpoint was not replaced.
- The StreamVGGT file header contains a plain parameter dictionary with
  `aggregator.*` keys, compatible with the release loader's expected format.
  Its published filename is `checkpoints.pth`; the reference evaluation used
  the local name `StreamVGGT-1B.pth`. The release uses the published filename.
- Weight-file SHA256 values come from the Hub's LFS metadata; processor and
  configuration files are pinned by revision and recorded Git blob hashes.
  `MODEL_SHA256SUMS` verifies the downloaded large files from the source root.

The reference script identifies these model releases by name. The historical
server copies were not accessible during release preparation, so their local
bytes were not independently hashed. This provenance check did not run a new
navigation evaluation.

## Upstream terms

- Qwen weights: Apache 2.0, as declared by the pinned model card.
- StreamVGGT weights: CC BY-NC 4.0, as declared by the pinned model card.
- StreamVGGT source code: CC BY-NC-SA 4.0. The code license and model-card
  license apply to their respective artifacts; keep both notices.
