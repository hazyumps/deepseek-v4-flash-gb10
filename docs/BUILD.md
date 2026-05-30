# Building the vLLM image for GB10 (sm_121)

> **Honesty up front:** as of 2026-05-30, *stock* `vllm-project/vllm` (incl.
> v0.22.0) does **not** run DeepSeek-V4-Flash on consumer Blackwell (sm_120/121
> / GB10). Its fused DeepSeek-V4 indexer + sparse-MLA kernels are sm_90/sm_100
> only, the sm12x fallback module isn't in the release, and DeepGEMM is gated off
> for sm_120 — so stock crashes at load. The working base is the **`jasl/vllm`
> fork**, which adds the SM12x DeepSeek-V4 path (Triton/torch fallbacks). This
> repo's patch + scripts sit **on top of that fork**.

## Base image

Build from the `jasl/vllm` fork (the SM12x DeepSeek-V4 effort):

- Fork: https://github.com/jasl/vllm  (branch `codex/ds4-sm120-min-enable`, the
  PR #41834 line; tracks upstream main + the v0.22.0 `deepseek_v4/` package).
- Canonical bring-up reference for bare-metal dual-Spark:
  **https://github.com/jasl/vllm-ds4-sm120-harness** —
  see `docs/dgx_spark_bare_metal_cluster.md` and `docs/sm120_optimization_notes.md`.

Pin a known-good commit (this repo was validated against fork build
`v0.1.dev16581+gdda4668b5`; newer rebases onto v0.22.0 should also work but
re-run `verify/t2_verify.py` after).

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
# 2. git clone https://github.com/jasl/vllm && git checkout <pinned-commit>
# 3. export TORCH_CUDA_ARCH_LIST=12.1a ; pip install -e . (or the fork's build path)
# 4. install FlashInfer with compute_120f
# Tag it (the scripts default to IMAGE=vllm-ds4-sm121:cu130).
```

The patch in `patches/` is **bind-mounted over** the in-image file at run time
(see the scripts), so you do **not** rebuild to apply or update it — just
restart the container.

## Why a patch instead of upstreaming

The indexer top-k op has no sm_120 kernel; the fork falls back to a torch path
that (a) ran FP32 cuBLAS → froze under concurrency, and (b) had no fused route.
`patches/sm12x_deep_gemm_fallbacks.py` fixes both (bf16 matmul inputs; a fused
tf32 Triton MQA-logits top-k path). Tracked upstream at vLLM #41063 / #41834 and
DeepGEMM #324; when those land natively for sm_120, this patch becomes moot.
