# [DEPRECATED] DeepSeek-V4-Flash on 2× NVIDIA GB10 (DGX Spark)

> [!WARNING]
> **DO NOT USE THIS REPOSITORY TO DEPLOY DEEPSEEK-V4-FLASH.** Use the actively maintained
> [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) project and
> its
> [`deepseek-v4-flash-0731` recipe](https://github.com/eugr/spark-vllm-docker/blob/main/recipes/deepseek-v4-flash-0731.yaml).
> That is also the stack we use now. This repository remains online only as a
> historical record of an earlier working setup.

For a new deployment, follow eugr's current networking and setup documentation.
At the time this repository was deprecated, the upstream quick start was:

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
./run-recipe.sh recipes/deepseek-v4-flash-0731.yaml --setup
```

The recipe uses eugr's tested B12X image and is where ongoing DGX Spark support,
fixes, and recipe updates live. Do not treat the image, fork, or pinned versions
below as current guidance.

An archived reproduction record for serving **DeepSeek-V4-Flash** across **two GB10 / DGX
Spark** boxes (compute capability **sm_121**, consumer Blackwell) with vLLM —
fast and reliably — at **384K context**, tensor-parallel + expert-parallel over
a RoCE link, with DSpark speculative decoding.

This was the config + runbook that took a dual-Spark setup from "crashes /
wedges / ~12 tok/s" to "stable, 384K, ~40–60 tok/s single-stream, ~1.7k tok/s
prefill." It is retained for historical and troubleshooting reference.

> Not affiliated with vLLM or DeepSeek. Built **on top of the `jasl/vllm` fork**,
> which carries the SM12x DeepSeek-V4 enablement. Stock vLLM does **not** run this
> model on sm_120/121 yet — see `docs/BUILD.md`. Apache-2.0.

**Historical tested configuration:** `deepseek-ai/DeepSeek-V4-Flash-0731` (GA) on fork tag
`sm120-pr-41834-stable-preview-20260727d` (`d64074e6f`), vLLM
`0.1.dev19369+gd64074e6f`. See [What changed for GA](#what-changed-for-ga-2026-07-31)
if you set this up from the pre-GA version of this repo.

## Archived contents

The scripts, patches, validation tools, and old image metadata remain for
historical comparison only. They are not an installation path. Do not pull or
base new work on `hazyumps/deepseek-v4-flash-gb10`; use eugr's repository and
current recipe linked above.

## Layout
```
scripts/start_head.sh, start_worker.sh # the tuned launch (TP=2 + EP, DSpark n=5, 384K/0.80, NCCL 2.30.4)
env.example                            # copy -> env.sh, set your IPs/NICs
verify/                                # boot-watch, prefill/decode probe, patch-correctness gate
docs/BUILD.md                          # the image (jasl/vllm fork, CUDA 13, arch 12.1a, NCCL 2.30.4)
docs/NETWORK.md                        # RoCE + RDMA passthrough + NCCL 2.30.4 (the reliability layer)
docs/TUNING.md                         # every flag explained + root causes + perf envelope + wedge calibration
docs/VALIDATION.md                     # how to confirm you're at the same spot
patches/                               # HISTORICAL indexer fix -- superseded, see patches/README.md
```

## What's tuned (and why) — short version
- **NCCL 2.30.4 via LD_PRELOAD** — kills the `shm_broadcast` deadlock (the wedge).
- **`--device=/dev/infiniband` + caps** — makes NCCL use RDMA, not TCP (~12→~30+ tok/s).
- **DSpark speculative decoding, `num_speculative_tokens: 5`** — GA replaces the
  single MTP head with 3 DSpark draft groups + a markov head. **n must be ≥
  `dspark_block_size` (5)** or the engine refuses to start; see `docs/TUNING.md`.
- **384K @ 0.80 mem-util, EP, fp8 KV, FULL_AND_PIECEWISE cudagraph** —
  see `docs/TUNING.md`.
- **No patch bind-mount anymore.** The indexer fix this repo shipped is now
  native in the fork; `patches/` is kept for the historical record only.

## Performance to expect (it's a bandwidth-bound box, not a cloud GPU)
Measured on the config above (GA, DSpark n=5, thinking off, unique prompts,
`ignore_eos`, counted from `usage.completion_tokens` — see `docs/VALIDATION.md`
for why each of those matters):

| metric | value |
|---|---|
| single-stream decode | **~47–60 tok/s** on easy content, **~40 tok/s** on real mixed content |
| prefill | **~1.6–1.8k tok/s**, roughly linear (6k ≈ 3.6s, 24k ≈ 13s, 45k ≈ 27s) |
| KV cache @ 0.80 util | 1,187,206 tokens → **~3.0x** concurrency at 393,216 |

Single-stream decode is **content-dependent and variable** — DSpark draft
acceptance depends on how predictable the output is, so run any comparison 3×
and report a range. Prefill is the stable, trustworthy headline number.

> These are ~4–5x the prefill and ~1.5x the decode of the numbers this repo
> published in 2026-05 (~330–430 tok/s prefill, ~31–34 tok/s decode). The gain
> is the fork's native fused DeepGEMM top-k path plus DSpark — not a config
> change on our side.

## Status / upstream
- **DeepGEMM #324** (sm120 kernels) — **merged** 2026-06-24 into `nv_dev`.
- **vLLM #41834** (SM12x DeepSeek-V4 support) — still **open**.
- **vLLM #41063** (tracking: DeepGEMM SM12.x coverage gaps) — still **open**.

Until #41834 lands, the `jasl/vllm` fork is still the way to run this on
consumer Blackwell.

## What changed for GA (2026-07-31)
GA (`-0731`) was **not** a drop-in over the beta, despite identical quantization
(fp8 e4m3 / ue8m0 / block 128) and the same 43-layer base:

- **Speculative decoding changed shape.** Beta had one MTP head; GA has **3
  DSpark draft groups + a markov head**. `--speculative-config` moves from
  `{"method":"deepseek_mtp","num_speculative_tokens":2}` to
  `{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}`.
- **Bigger checkpoint:** 48 shards / ~167 GB (vs 46 / ~160 GB).
- **A rebuild is mandatory** — the older image cannot load the DSpark draft.
- **Concurrency headroom dropped** from ~5.5x to ~3.0x at 384K (bigger weights,
  same 0.80 util). Still fine for `--max-num-seqs 4`.
- **`--max-num-seqs` went 2 → 4**, which the 2026-06 fork rebase made stall-free.
- The indexer patch is **gone from the launch** — its fixes are native now.

## Credits
The SM12x DeepSeek-V4 enablement is **jasl**'s work (`jasl/vllm`,
`jasl/vllm-ds4-sm120-harness`). This repo adds GB10-specific tuning + a
reproducible historical runbook on top. The current deployment and maintained
recipe are eugr's work. Model: DeepSeek. Engine: vLLM (Apache-2.0).

## Licensing of the published image

**Apache-2.0 covers this repository — the scripts, docs and config. It does not
cover the container image.**

The published image is a derived container built on NVIDIA's `nvidia/cuda`
sbsa base, and is distributed under the
[NVIDIA Deep Learning Container License](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf).
By pulling it you accept those terms. It also bundles vLLM (Apache-2.0), the
`jasl/vllm` fork's changes, and CUDA 13.2 components under NVIDIA's respective
licenses. It contains **no model weights** — pull those from DeepSeek yourself.

> This software contains source code provided by NVIDIA Corporation.
