# Tuning & root causes (what the config choices buy you)

Every non-obvious flag in the start scripts exists because of a specific failure.

## Speculative decoding: DSpark (GA) vs MTP (beta)

GA `-0731` replaced the beta's single MTP head with **3 DSpark draft groups + a
markov head**, so the flag changes:

```
# beta:
--speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}'
# GA:
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}'
```

**`num_speculative_tokens` must be ≥ `dspark_block_size` (5)** or the engine
crash-loops at startup. The trap: that validator is a pydantic post-init in
`vllm/config/speculative.py`, **not** in the proposer
(`v1/spec_decode/dspark.py`) — read only the proposer and you will conclude
there is no constraint. DeepSeek recommends 7; 5 is what this recipe runs and
validates.

At boot you should see, per rank:

```
DSpark draft model loaded: 99 params
Shared target model embeddings with DSpark draft model.
Using auxiliary layers from speculative config: (40, 41, 42)
```

## Key flags

| Flag | Why |
|---|---|
| `--enable-expert-parallel` | halves expert weight per node — the OOM fix for fitting on 2 Sparks |
| `--kv-cache-dtype fp8` + MLA | extremely compact KV — 1,187,206 tokens of cache at 0.80 util |
| `--max-model-len 393216` + `--gpu-memory-utilization 0.80` | 384K context, ~3.0x concurrency headroom on GA. 0.70 gives ~1.1x. 0.80 is stable but watch unified-mem; `expandable_segments` (below) helps. |
| `--max-num-seqs 4` + `--max-num-batched-tokens 4096` | 4 became stall-free with the 2026-06 fork rebase (was 2). Higher values re-introduce "prefill starves decode" stalls on this HW. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | reduces unified-memory fragmentation/OOM on the big KV allocs |
| `--compilation-config FULL_AND_PIECEWISE + custom_ops:all` | recipe-validated cudagraph mode for DSv4 |
| `--no-enable-flashinfer-autotune` | avoids a 10+ min startup autotune |
| `DG_JIT_USE_NVRTC=0` + `DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc` | DeepGEMM JIT compiles through nvcc; NVRTC misbuilds some sm_12x kernels |
| `FLASHINFER_DISABLE_VERSION_CHECK=1` | FlashInfer refuses to load against the CUDA 13 stack otherwise |
| triton / vllm / flashinfer cache volume mounts | persist compiled kernels across restarts (else a recompile "hang" every boot) |

**Expected startup warning, not a problem:** with DSpark n=5 and
`--max-num-seqs 4`, vLLM logs `max_num_scheduled_tokens is set to 4080 based on
the speculative decoding settings` — it reserves draft-token slots out of the
4096 budget. Raising `--max-num-batched-tokens` to reclaim the 16 tokens is not
worth the scheduling instability.

## `reasoning_effort` is mostly a no-op through vLLM

GA advertises a `low` reasoning tier, but vLLM's `tokenizers/deepseek_v4.py`
normalizes the value *before* the encoder's
`assert reasoning_effort in ['max', None, 'high']`, with a catch-all
`else: "high"`. So `medium` behaves exactly like `high`, and `low` is
unreachable. This is byte-identical on beta and GA — don't spend time tuning it
until the tokenizer changes.

## Performance envelope (expect this, not cloud-GPU numbers)

Measured on GA + DSpark n=5 (method in `docs/VALIDATION.md`):

- **Single-stream decode: ~47–60 tok/s** on easy/repetitive output, **~40 tok/s**
  on real mixed content. **Variable by design** — DSpark acceptance is
  content-dependent. Run 3× and report a range; a single number is noise.
- **Prefill: ~1.6–1.8k tok/s**, roughly linear (6k ≈ 3.6s, 24k ≈ 13s, 45k ≈ 27s).
  This is the stable, trustworthy metric.
- Concurrent throughput exceeds single-stream; `--max-num-seqs 4` is the tested
  ceiling.

Underneath, this is still a **bandwidth**-bound box (GB10 unified LPDDR5x, not
HBM). Long context is comfortable now but a 200K prefill is still ~2 minutes —
plan it as batch reasoning, not interactive.

## Wedge vs. recoverable stall (hard-won calibration)

`/health`=200 **lies** (separate process). The real liveness signal is
`vllm:generation_tokens_total` advancing. On dual Spark, normal generation can
stall up to **~7.5 min** with a *single* `shm_broadcast` marker and still
recover. Only **repeated markers (3+ over ~3 min) AND sustained non-recovery
>10 min** is a true wedge — don't auto-restart on weaker signals. GPU wattage is
*not* a reliable discriminator (NCCL spin, Triton compile, and real compute all
sit ~40–75W). `py-spy` (needs `--cap-add=SYS_PTRACE`, already set) shows the
exact stuck function. The NCCL 2.30.4 upgrade (NETWORK.md) eliminated the
original hard wedges, and the 2026-06 fork rebase eliminated the long-prefill
one.
