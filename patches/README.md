# patches/ — historical

`sm12x_deep_gemm_fallbacks.py` is **no longer used**. The current launch scripts
do not bind-mount it, and the in-image file it replaced moved when the fork
rebased onto vLLM v0.22.0. It is kept here for the record.

## What it did

On sm_121 the DeepSeek-V4 "lightning indexer" top-k had **no native kernel** and
fell back to a torch path. Two fixes:

1. **bf16 matmul inputs** (was FP32 → cuBLAS SGEMM on CUDA cores, tensor cores
   idle). This is what froze concurrency: 4 concurrent requests went from
   0.1 → ~60 tok/s. GB10-specific — it showed no benefit on RTX Pro 6000.
2. **Fused tf32 Triton top-k** (`_fp8_mqa_logits_topk_triton`): routed the
   per-chunk logits through the existing fused MQA-logits kernel instead of a
   bf16 cuBLAS head-loop with per-iteration ~1 GiB score materialization.
   Worth ~+29% prefill at 9k (313 → 405 tok/s), and *more accurate* than the
   bf16 path (matched the fp32 gold exactly on the top-512).

It self-fell-back to the torch path on any error, as a live-serving safety net.

## Why it's gone

The fork carries a native fused DeepGEMM top-k path now — on SM12x,
`use_fp4_indexer_cache` is off, so prefill always takes the fused route and
never materializes logits at all. That is a large part of why prefill went from
~400 tok/s to ~1.7k tok/s across the 2026-06/07 rebuilds.

`verify/t2_verify.py` (the correctness gate for the fused top-k) is likewise
only meaningful against the 2026-05-era build.
