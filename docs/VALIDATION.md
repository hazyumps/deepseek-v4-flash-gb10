# Validation — confirm you're at the same spot

Run these after `Application startup complete`. `VLLM_URL` = the head's API
(e.g. `http://10.255.0.1:8000`).

## 1. Boot health (on the head node)
```
VLLM_URL=http://<HEAD_IP>:8000 bash verify/boot-watch.sh
```
Expect: `NCCL version 2.30.4`, many `via NET/IB`,
`DSpark draft model loaded: 99 params`, and
`Maximum concurrency for 393,216 tokens per request: ~3.0x`
(`GPU KV cache size: 1,187,206 tokens`).

If concurrency reads ~5.5x you are on the **beta** weights, not GA.

## 2. Prefill / retrieval
```
VLLM_URL=http://<HEAD_IP>:8000 python3 verify/prefill_test.py 2048
VLLM_URL=http://<HEAD_IP>:8000 python3 verify/prefill_test.py 8600
VLLM_URL=http://<HEAD_IP>:8000 python3 verify/prefill_test.py 16384
```
Reference numbers on this config (GA, DSpark n=5), measured 2026-08-03:

| arg | prompt tokens | TTFT | prefill tok/s | nonce |
|---|---|---|---|---|
| 2048 | 2,953 | 1.56s | ~1,900 | correct |
| 8600 | 12,885 | 6.93s | ~1,860 | correct |
| 16384 | 23,917 | 13.8s | ~1,730 | correct |

The retrieved nonce must be **correct** at every size — that is the real point
of this probe. It proves long-context retrieval survives speculative decoding.

**Ignore this script's `decode` line as a benchmark.** The answer is ~13 tokens,
so the rate is measured over a window too short to mean anything (it swings
23–72 tok/s across the runs above). Use §3 for decode.

## 3. Decode rate, measured properly

Single-stream decode needs isolation or the number lies. Four rules, each one a
trap that produces wrong numbers if skipped:

1. **Unique prompts** — prepend `os.urandom(8).hex()`, or `--enable-prefix-caching`
   serves prefill from cache and TTFT reads ~0.
2. **Count `usage.completion_tokens`, not stream deltas** — SSE batches several
   tokens per chunk; delta-counting undercounts by ~50x.
3. **`"ignore_eos": true`** — forces exactly `max_tokens`, so counts are comparable.
4. **`"chat_template_kwargs": {"thinking": false}`** — otherwise reasoning tokens
   dominate, land in `reasoning_content` rather than `content`, and vary per prompt.

Then subtract prefill + connection overhead by timing a 1-token and a 300-token
call: `decode_tps = (300 - 1) / (wall_300 - wall_1)`.

Reference (3 runs, GA): **60.5 / 47.0 / 54.8 tok/s.** Real mixed content lands
around **~40 tok/s**. The spread is DSpark acceptance being content-dependent —
always run 3× and report a range.

## 4. Stall check under load
```
docker logs --since 3m vllm-ds4 | grep -c shm_broadcast   # accumulating = wedge
curl -s http://<HEAD_IP>:8000/metrics | grep num_preemptions_total   # >0 = KV eviction
```
See `docs/TUNING.md` for wedge-vs-recoverable-stall calibration.

## 5. (Optional) tool-calling + perf, standardized
A community bench that hits any OpenAI-compatible endpoint:
```
uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git
tool-eval-bench --base-url http://<HEAD_IP>:8000 --short --perf
```
Reference (measured on the pre-GA build): ~93/100 tool-calling, strong tool
selection / multi-step / error-recovery; the throughput sweep shows decode
falling off under concurrency+depth (the bandwidth wall) — expected.

## Historical: patch correctness

Earlier versions of this repo bind-mounted `patches/sm12x_deep_gemm_fallbacks.py`
and gated it with `verify/t2_verify.py`. The launch no longer mounts a patch, so
this step is gone. Keep it only if you are running the 2026-05-era build; see
`patches/README.md`.

If §1–§4 pass, you're at the same spot.
