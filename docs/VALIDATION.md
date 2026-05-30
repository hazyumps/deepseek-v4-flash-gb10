# Validation — confirm you're at the same spot

Run these after `Application startup complete`. `VLLM_URL` = the head's API
(e.g. `http://10.255.0.1:8000`).

## 1. Boot health (on the head node)
```
VLLM_URL=http://<HEAD_IP>:8000 bash verify/boot-watch.sh
```
Expect: `NCCL version 2.30.4`, many `via NET/IB`, and
`Maximum concurrency for 393,216 tokens per request: ~5.x`.

## 2. Patch correctness (inside the container, GPU)
```
docker cp patches/sm12x_deep_gemm_fallbacks.py vllm-ds4:/tmp/patch.py
docker cp verify/t2_verify.py vllm-ds4:/tmp/t2_verify.py
docker exec vllm-ds4 python3 /tmp/t2_verify.py /tmp/patch.py
```
Expect: `overlap(tf32-triton, fp32-gold) = 1.0000` (≥ the bf16 number) → `PASS`.
This proves the fused top-k selects the same tokens as a perfect fp32 reference.

## 3. Prefill / decode rates
```
VLLM_URL=http://<HEAD_IP>:8000 python3 verify/prefill_test.py 2048
VLLM_URL=http://<HEAD_IP>:8000 python3 verify/prefill_test.py 8600
VLLM_URL=http://<HEAD_IP>:8000 python3 verify/prefill_test.py 16384
```
Reference numbers on this config (your exact tok/s will vary a little):

| prompt | TTFT | prefill tok/s | decode tok/s | nonce |
|---|---|---|---|---|
| ~2.3k | ~5s | ~430 | ~33 | correct |
| ~9k | ~22s | ~405 | ~39 | correct |
| ~17k | ~48s | ~365 | ~42 | correct |

The retrieved nonce must be **correct** at every size (proves long-context
retrieval is intact). Single-stream decode ~31–34 tok/s is the expected GB10
ceiling.

## 4. (Optional) tool-calling + perf, standardized
A community bench that hits any OpenAI-compatible endpoint:
```
uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git
tool-eval-bench --base-url http://<HEAD_IP>:8000 --short --perf
```
Reference: ~93/100 tool-calling (★★★★★), strong tool selection / multi-step /
error-recovery; the throughput sweep shows decode falling off under
concurrency+depth (the bandwidth wall) — expected.

If all four pass, you're at the same spot.
