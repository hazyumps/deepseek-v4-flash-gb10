# Tuning & root causes (what the config choices buy you)

Every non-obvious flag in the start scripts exists because of a specific failure.

## The patch: `patches/sm12x_deep_gemm_fallbacks.py`

On sm_121 the DeepSeek-V4 "lightning indexer" top-k has **no native kernel** —
it falls to a torch path. Two fixes in the patch:

1. **bf16 matmul inputs** (was FP32 → cuBLAS SGEMM on CUDA cores, tensor cores
   idle). This is what froze concurrency: 4 concurrent went 0.1 → ~60 tok/s.
   *(Note: this helps sm_121/GB10 specifically; it showed no benefit on RTX Pro
   6000, so it's a GB10 fix, not a universal one.)*
2. **Fused tf32 Triton top-k** (`_fp8_mqa_logits_topk_triton`): routes the
   per-chunk logits through the existing fused MQA-logits kernel instead of a
   bf16 cuBLAS head-loop + per-iter ~1 GiB score materialization. Result: ~+29%
   prefill at 9k (313 → 405 tok/s). It's also *more accurate* than the bf16 path
   (matches fp32 gold exactly on the top-512). Verify with `verify/t2_verify.py`.

It self-falls-back to the torch path on any error (live-serving safety net).

## Key flags

| Flag | Why |
|---|---|
| `--enable-expert-parallel` | halves expert weight per node — the OOM fix for fitting on 2 Sparks |
| `--kv-cache-dtype fp8` + MLA | extremely compact KV — 384K context fits in ~18 GiB |
| `--max-model-len 393216` + `--gpu-memory-utilization 0.80` | 384K context, ~5.5x concurrency headroom. 0.70 only gives ~1.1x at 384K. 0.80 is stable but watch unified-mem; `expandable_segments` (below) helps. |
| `--max-num-seqs 2` + `--max-num-batched-tokens 4096` | the stable point — higher values maximize "prefill starves decode" stalls on this HW |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | reduces unified-memory fragmentation/OOM on the big KV allocs |
| `--speculative-config deepseek_mtp n=2` | MTP ~doubles decode throughput. **Caveat:** MTP can cause occasional long-context retrieval misses — toggle off to test if a big-context answer looks wrong. |
| `--compilation-config FULL_AND_PIECEWISE + custom_ops:all` | recipe-validated cudagraph mode for DSv4 |
| `--no-enable-flashinfer-autotune` | avoids a 10+ min startup autotune |
| triton-cache + vllm-cache volume mounts | persist compiled kernels across restarts (else a recompile "hang" every boot) |

## Performance envelope (expect this, not cloud-GPU numbers)

- **Single-stream decode: ~31–34 tok/s** (MTP on). This is a *bandwidth* ceiling
  — GB10 unified LPDDR5x, not an HBM datacenter card. It's normal.
- **Prefill is linear, ~330–430 tok/s.** A 9k prompt ≈ 22s TTFT; 200K ≈ minutes.
  Bounded and reliable, just slow — long-context is for batch reasoning, not
  interactive. (The T2 patch helps mid-context most.)
- Concurrent throughput > single-stream once the indexer isn't freezing.

## Wedge vs. recoverable stall (hard-won calibration)

`/health`=200 **lies** (separate process). The real liveness signal is
`vllm:generation_tokens_total` advancing. On dual Spark, normal generation can
stall up to **~7.5 min** with a *single* `shm_broadcast` marker and still
recover. Only **repeated markers (3+ over ~3 min) AND sustained non-recovery
>10 min** is a true wedge — don't auto-restart on weaker signals. GPU wattage is
*not* a reliable discriminator (NCCL spin, Triton compile, and real compute all
sit ~40–75W). `py-spy` (needs `--cap-add=SYS_PTRACE`, already set) shows the
exact stuck function. The NCCL 2.30.4 upgrade (NETWORK.md) eliminates the
original hard wedges.
