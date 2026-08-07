# Building the vLLM image for GB10 (sm_121)

## Don't build it if you don't have to

The exact image this repo's numbers come from is published:

```
docker pull hazyumps/deepseek-v4-flash-gb10:sm121-cu130-20260727d
```

**`linux/arm64` only** (GB10/aarch64 — it will not run on x86; there is no
amd64 manifest, so on an x86 host it pulls and then fails at runtime).
~11 GB compressed. Contains the pinned fork build below; nothing else in this
repo changes. Building from source takes ~50 minutes, most of it WAN, and needs
~107 GB free RAM — so pull unless you're changing the fork pin.

To pin by digest instead of tag:

```
docker pull hazyumps/deepseek-v4-flash-gb10@sha256:08241111a99c5c1d15e14d11cb04f9b897fc3cd31d978f31e38641fed8a1cdb8
```

The rest of this document is for when you *are* changing it.

> **Honesty up front:** as of 2026-08-03, *stock* `vllm-project/vllm` does **not**
> run DeepSeek-V4-Flash on consumer Blackwell (sm_120/121 / GB10). Its fused
> DeepSeek-V4 indexer + sparse-MLA kernels are sm_90/sm_100 only and DeepGEMM's
> sm_120 support (merged in DeepGEMM #324, 2026-06-24) hasn't reached a vLLM
> release with the model path wired up — so stock crashes at load. The working
> base is the **`jasl/vllm` fork**, which adds the SM12x DeepSeek-V4 path.
> Tracking: vLLM **#41834** (open) and **#41063** (open).

## Base image

Build from the `jasl/vllm` fork (the SM12x DeepSeek-V4 effort):

- Fork: https://github.com/jasl/vllm — the PR #41834 line, tagged as
  `sm120-pr-41834-stable-preview-<date>`.
- Canonical bring-up reference for bare-metal dual-Spark:
  **https://github.com/jasl/vllm-ds4-sm120-harness** —
  see `docs/dgx_spark_bare_metal_cluster.md` and `docs/sm120_optimization_notes.md`.

**Current pin (validated by this repo):**

| | |
|---|---|
| tag | `sm120-pr-41834-stable-preview-20260727d` |
| commit | `d64074e6f07250f6cd072861aa3c389a929befb9` |
| reports as | `vLLM 0.1.dev19369+gd64074e6f` |
| model | `deepseek-ai/DeepSeek-V4-Flash-0731` (GA) |

Newer `stable-preview` tags appear regularly and generally work; re-run
`docs/VALIDATION.md` after any bump. The **GA weights need a GA-era build** —
the DSpark draft (3 draft groups + markov head) will not load in a pre-GA image.
Keep the old image around until the new one serves; that is your rollback.

Previously validated pins, for the record: `73e99c16` (`sm12x-20260617`, fixed
the streaming tool-call crash and the long-prefill wedge), `8725eb97`
(`sm12x-20260605`), `dda4668b5` (the original 2026-05 build).

## Toolchain that worked (GB10 / aarch64)

- CUDA **13.x** (image tagged `cu130` here), aarch64
- `TORCH_CUDA_ARCH_LIST=12.1a`, `FLASHINFER_CUDA_ARCH_LIST=12.1a`
  (the default `"12.0+PTX"` produces non-native sm_12x cubins — use `12.1a`)
- **NCCL 2.30.4** present in the image (`libnccl2=2.30.4-1+cuda13.2`) — see NETWORK.md
- FlashInfer with sm_120/121 (`compute_120f`) support

## Outline (adapt to your build)

```dockerfile
# Base: NVIDIA CUDA 13.x aarch64 devel image with the GB10 driver stack.
# 1. apt install libnccl2=2.30.4-1+cuda13.2 libnccl-dev=2.30.4-1+cuda13.2
# 2. git clone https://github.com/jasl/vllm && git checkout <pinned-tag>
# 3. git tag -l | xargs -r git tag -d      # see gotcha 1 below
# 4. export TORCH_CUDA_ARCH_LIST=12.1a ; build the fork (uv build / pip install -e .)
# 5. install FlashInfer with compute_120f
# Tag it and point IMAGE= at it (env.example defaults to the published image).
```

## Build gotchas

1. **`setuptools_scm` chokes on the fork's tags** — version resolution fails and
   the build dies. Delete the tags in the build stage before building:
   `git tag -l | xargs -r git tag -d`. (You lose nothing; the version string is
   derived from the commit either way.)
2. **The long pole is WAN, not CPU.** `FetchContent` submodule clones (notably
   `ROCm/aiter`, ~482 MB) dominate; a full build ran ~50 min end to end with
   only ~3 min of that on the compiler. Don't diagnose a "stall" with
   `pgrep -c -f cicc|nvcc` — that matches cmake's own
   `-DCMAKE_CUDA_COMPILER=` argv. Use `pgrep -c -x`.
3. **Free the RAM first.** The build needs ~107 GB free; stop the serving
   cluster before building on a node that is also serving.
4. **Torch version drift** (hit on the 2026-06-05 v0.22.0 rebase, not since): the
   wheel install pulled an aarch64 *CPU* torch over the CUDA one →
   `libtorch_cuda.so not found`. Re-check `python -c "import torch;
   print(torch.version.cuda)"` inside the built image before shipping it.

## About `patches/`

The indexer fix this repo originally shipped (`sm12x_deep_gemm_fallbacks.py`,
bind-mounted over the in-image file) is **no longer used** — its fixes are native
in the fork, and the file it patched moved in v0.22.0. The launch scripts no
longer mount it. See `patches/README.md` for what it did and why it mattered.
