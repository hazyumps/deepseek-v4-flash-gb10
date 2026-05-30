# DeepSeek-V4-Flash on 2× NVIDIA GB10 (DGX Spark)

A reproduction recipe to serve **DeepSeek-V4-Flash** across **two GB10 / DGX
Spark** boxes (compute capability **sm_121**, consumer Blackwell) with vLLM —
fast and reliably — at **384K context**, tensor-parallel + expert-parallel over
a RoCE link, with MTP speculative decoding.

This is the config + patch + runbook that took a dual-Spark setup from
"crashes / wedges / ~12 tok/s" to "stable, 384K, ~31 tok/s single-stream,
~405 tok/s prefill @ 9k." If you have two Sparks and want DeepSeek-V4-Flash,
start here.

> Not affiliated with vLLM or DeepSeek. Built **on top of the `jasl/vllm` fork**,
> which carries the SM12x DeepSeek-V4 enablement. Stock vLLM (incl. v0.22.0) does
> **not** run this model on sm_120/121 yet — see `docs/BUILD.md`. Apache-2.0.

## What you need
- **2× GB10 / DGX Spark** (sm_121, aarch64), CUDA 13.x driver stack.
- A **RoCE point-to-point link** between the two NICs (one cable). See `docs/NETWORK.md`.
- The model weights: `deepseek-ai/DeepSeek-V4-Flash`.
- Docker with the NVIDIA runtime on both nodes.

## Quickstart
1. **Build the image** (once, on each node or shared registry) — `docs/BUILD.md`.
   Defaults to tag `vllm-ds4-sm121:cu130`.
2. **Wire the network** — `docs/NETWORK.md`. RDMA passthrough + **NCCL 2.30.4**
   (the wedge fix) are mandatory. Set MTU 9000 on the RoCE link.
3. **Configure** — `cp env.example env.sh`, edit IPs/iface/HCA for your boxes.
4. **Launch** (head first):
   ```
   # node 1:
   bash scripts/start_head.sh
   # node 2:
   bash scripts/start_worker.sh
   ```
   Cold boot ~4–5 min (148 GB weights + compile + cudagraph capture).
5. **Verify** — `docs/VALIDATION.md`. Run `verify/boot-watch.sh` (head),
   `verify/t2_verify.py` (in-container), and `verify/prefill_test.py`.
   You should see NCCL 2.30.4, `via NET/IB`, 384K @ ~5.5x concurrency, and the
   reference tok/s.

## Layout
```
patches/sm12x_deep_gemm_fallbacks.py   # the indexer fix (bf16 + fused tf32 Triton top-k); bind-mounted, no rebuild
scripts/start_head.sh, start_worker.sh # the tuned launch (TP=2 + EP, MTP n=2, 384K/0.80, NCCL 2.30.4)
env.example                            # copy -> env.sh, set your IPs/NICs
verify/                                # boot-watch, patch-correctness gate, prefill/decode probe
docs/BUILD.md                          # the image (jasl/vllm fork, CUDA 13, arch 12.1a, NCCL 2.30.4)
docs/NETWORK.md                        # RoCE + RDMA passthrough + NCCL 2.30.4 (the reliability layer)
docs/TUNING.md                         # every flag explained + root causes + perf envelope + wedge calibration
docs/VALIDATION.md                     # how to confirm you're at the same spot
```

## What's tuned (and why) — short version
- **NCCL 2.30.4 via LD_PRELOAD** — kills the `shm_broadcast` deadlock (the wedge).
- **`--device=/dev/infiniband` + caps** — makes NCCL use RDMA, not TCP (~12→~30+ tok/s).
- **The patch** — sm_121 has no native lightning-indexer kernel; the bf16 fix
  unfreezes concurrency and the fused Triton top-k adds ~29% prefill (and is more
  accurate than the bf16 fallback). `verify/t2_verify.py` proves it.
- **384K @ 0.80 mem-util, MTP n=2, EP, fp8 KV, FULL_AND_PIECEWISE cudagraph** —
  see `docs/TUNING.md`.

## Performance to expect (it's a bandwidth-bound box, not a cloud GPU)
~31–34 tok/s single-stream decode; prefill linear ~330–430 tok/s (9k ≈ 22s TTFT,
200K ≈ minutes). Long context is for batch reasoning, not interactive. Concurrent
throughput exceeds single-stream once the indexer isn't freezing.

## Status / upstream
The native fix lives at vLLM **#41834** (SM12x DeepSeek-V4) + DeepGEMM **#324**
(sm120 kernels) + tracking issue **#41063**. When those merge, stock vLLM should
serve this without the patch — until then, this recipe is the way.

## Credits
The SM12x DeepSeek-V4 enablement is **jasl**'s work (`jasl/vllm`,
`jasl/vllm-ds4-sm120-harness`). This repo adds GB10-specific tuning + an indexer
patch + a reproducible runbook on top. Model: DeepSeek. Engine: vLLM (Apache-2.0).
