# Third-party notices

The root [LICENSE](LICENSE) preserves the MIT license of the inherited LEO
codebase. Included third-party source files retain their upstream copyright
notices and licenses. The root license does not replace those terms or license
the separately distributed model weights and datasets.

## Included source

| Local component | Upstream | License text |
| --- | --- | --- |
| LEO-derived training and model infrastructure | [LEO](https://github.com/embodied-generalist/embodied-generalist) | [MIT](LICENSE), copyright 2023 embodied-generalist |
| StreamVGGT integration in `model/streamvggt/` | [StreamVGGT](https://github.com/wzzheng/StreamVGGT) | [CC BY-NC-SA 4.0](third_party/licenses/StreamVGGT.txt) |
| VGGT-derived geometry modules within `model/streamvggt/` | [VGGT](https://github.com/facebookresearch/vggt) | [VGGT License and Acceptable Use Policy](third_party/licenses/VGGT.txt); retain individual file notices |
| Qwen3-VL modeling and processing files in `model/qwen/` | [Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_vl) | [Apache 2.0](third_party/licenses/Transformers.txt), with Qwen/Hugging Face copyright notices in the files |
| Files explicitly marked Apache 2.0 in `model/streamvggt/layers/` | Meta/DINOv2-derived layers included through StreamVGGT | [Apache 2.0 text](third_party/licenses/Transformers.txt); retain Meta copyright notices in the files |
| Token-merging utility in `model/memory/tome/merge.py` | [ToMe](https://github.com/facebookresearch/ToMe) | [CC BY-NC 4.0](third_party/licenses/ToMe.txt), copyright Meta Platforms, Inc. and affiliates |

MemVLN modifies the Qwen integration for navigation and episodic-memory
conditioning, and the StreamVGGT integration for streaming navigation inputs,
cached features, optional prediction heads, and model dtype handling. Existing
upstream notices remain in the source. StreamVGGT-derived adaptations retain
its noncommercial and share-alike terms; ToMe code retains its noncommercial
terms. This repository should not be described as wholly MIT-licensed.

The license copies above were retrieved from these upstream files on
2026-10-26:

- [StreamVGGT LICENSE.txt](https://github.com/wzzheng/StreamVGGT/blob/main/LICENSE.txt)
- [VGGT LICENSE.txt](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt)
- [ToMe LICENSE](https://github.com/facebookresearch/ToMe/blob/main/LICENSE)
- [Transformers LICENSE](https://github.com/huggingface/transformers/blob/main/LICENSE)

The original vendoring commits are not recorded. These copies document the
listed upstream terms; they do not assert that every included file came from
the current upstream revision.

## Model weights and datasets

Qwen3-VL and StreamVGGT weights are downloaded separately. Their model cards
and licenses apply to those artifacts. The pinned Qwen model card specifies
Apache 2.0; the pinned StreamVGGT model card specifies CC BY-NC 4.0 for its
weights, while its source code above uses CC BY-NC-SA 4.0. See
[release/BASE_MODELS.md](release/BASE_MODELS.md) and
[release/artifacts.json](release/artifacts.json) for the exact revisions and
file hashes.

R2R, FGR2R, RxR, and Matterport3D retain their own licenses and access terms.
In particular, the [Matterport3D agreement](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf)
covers derived information, including trained models. It requires an agreement
copy or link with published derived material. Distribution containing a
substantial portion of Matterport3D data also requires recipients to accept
the agreement before access and a record of that acceptance. Include the
agreement in the data and checkpoint release documentation.
