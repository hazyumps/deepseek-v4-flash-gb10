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
raises at startup and crash-loops. The trap: that check lives in
`SpeculativeConfig.__post_init__` (`vllm/config/speculative.py`), **not** in the
proposer (`v1/spec_decode/dspark.py`) — read only the proposer and you will
conclude there is no constraint.

It is a correctness guard, not a perf hint. The fork's own comment: a smaller
value "feeds the block / Markov-head machinery an unsupported layout and yields
incorrect (garbled) output rather than merely lower acceptance." DeepSeek
recommends 7; 5 is what this recipe runs and validates.

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
| `--max-model-len 393216` + `--gpu-memory-utilization 0.80` | 384K context, ~3.0x concurrency headroom on GA (verified). 0.80 is stable but watch unified-mem; `expandable_segments` (below) helps. †0.70 gave only ~1.1x on the **beta** weights; not re-measured on GA. |
| `--max-num-seqs 4` + `--max-num-batched-tokens 4096` | 4 is stall-free here (was 2 before the 2026-06 fork rebase). †The "higher values cause prefill-starves-decode stalls" finding is from the pre-GA build and has not been re-tested on GA. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | reduces unified-memory fragmentation/OOM on the big KV allocs |
| `VLLM_TRITON_MLA_SPARSE=1` | selects the Triton sparse-MLA path — the sm_12x route for DSv4 attention |
| `--compilation-config FULL_AND_PIECEWISE + custom_ops:all` | recipe-validated cudagraph mode for DSv4 |
| `--no-enable-flashinfer-autotune` | avoids a 10+ min startup autotune |
| `DG_JIT_USE_NVRTC=0` + `DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc` | routes DeepGEMM's JIT through `nvcc` instead of NVRTC and tells it where nvcc is. Both are read by the vendored `vllm/third_party/deep_gemm`. Inherited from the bring-up recipe — we have no recorded failure that it fixes, so treat it as "known-good", not "required". |
| `FLASHINFER_DISABLE_VERSION_CHECK=1` | bypasses FlashInfer's guard that `flashinfer`, `flashinfer-cubin` and `flashinfer-jit-cache` report matching versions. A source-built FlashInfer next to packaged cubin/jit-cache wheels trips it and raises at import. Nothing to do with CUDA compatibility. |
| triton / vllm / flashinfer cache volume mounts | persist compiled kernels across restarts (else a recompile "hang" every boot) |

† = carried over from the pre-GA build and not re-measured. Everything else in
this table was checked against the running GA cluster on 2026-08-03.

**Expected startup warning, not a problem:** with DSpark n=5 and
`--max-num-seqs 4`, vLLM logs `max_num_scheduled_tokens is set to 4080 based on
the speculative decoding settings` — it reserves draft-token slots out of the
4096 budget. Raising `--max-num-batched-tokens` to reclaim the 16 tokens is not
worth the scheduling instability.

## `reasoning_effort`: three tiers, not five

`vllm/tokenizers/deepseek_v4.py` normalizes the request value before the encoder
sees it, and the encoder then asserts `reasoning_effort in ['max', None, 'high']`.
The mapping is:

| you send | you get | effect |
|---|---|---|
| `none` | `thinking_mode="chat"`, effort `None` | thinking off |
| `max` or `xhigh` | `max` | `REASONING_EFFORT_MAX` prefix at message 0 |
| anything else — incl. `high`, `medium`, `low` | `high` | no prefix |

So GA's advertised `low` tier is unreachable through vLLM, and `medium` is
byte-identical to `high` — a client sending `medium` has never been getting a
middle tier. Only `none` and `max`/`xhigh` actually change the prompt. Verified
in the running GA image; the same code shipped in the beta image.

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

> Calibrated in 2026-05 on the pre-GA build, before the NCCL 2.30.4 upgrade and
> the 2026-06 fork rebase removed the wedges it was written for. The thresholds
> below have **not** been re-derived on GA — we simply have not seen a wedge
> since. Treat them as an upper bound on how patient to be, not a live spec.

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
