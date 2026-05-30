"""Patch correctness gate. Loads patches/sm12x_deep_gemm_fallbacks.py and proves
the fused-Triton top-k path (T2) is at least as accurate as the bf16 torch path,
both measured against an fp32 gold reference. Run INSIDE the vLLM container
(needs torch + the FlashInfer/Triton kernels + a GPU), e.g.:

  docker cp patches/sm12x_deep_gemm_fallbacks.py vllm-ds4:/tmp/patch.py
  docker cp verify/t2_verify.py vllm-ds4:/tmp/t2_verify.py
  docker exec vllm-ds4 python3 /tmp/t2_verify.py /tmp/patch.py

PASS = the Triton top-k indices match fp32 gold at least as well as bf16 does
(tf32's extra mantissa typically matches fp32 *exactly* on the top-512).
"""
import sys, importlib.util, torch

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/patch.py"
spec = importlib.util.spec_from_file_location("patch", path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

dev = "cuda"; torch.manual_seed(7)
seq = seqkv = 4096; H, D, K = 64, 128, 512
q = torch.randn(seq, H, D, device=dev).to(torch.float8_e4m3fn)
k = torch.randn(seqkv, D, device=dev).to(torch.float8_e4m3fn)
kscale = (torch.rand(seqkv, device=dev) + 0.1).float()
w = torch.randn(seq, H, device=dev).float()
ks = torch.zeros(seq, device=dev, dtype=torch.int32)
ke = torch.full((seq,), seqkv, device=dev, dtype=torch.int32)

qf = q.float(); kf = k.float() * kscale[:, None]
gold = torch.zeros(seq, seqkv, device=dev)
for h in range(H):
    gold.add_((qf[:, h, :] @ kf.t()).relu_() * w[:, h:h+1])
gold_idx = torch.topk(gold, K, dim=1).indices

ot = mod._fp8_mqa_logits_topk_torch((q, None), (k, kscale), w, ks, ke, K).clone()
otr = mod._fp8_mqa_logits_topk_triton((q, None), (k, kscale), w, ks, ke, K).clone()

def ov(a, b):
    return sum(len((set(a[i].tolist()) - {-1}) & (set(b[i].tolist()) - {-1})) / K
               for i in range(a.shape[0])) / a.shape[0]

o_t, o_tr = ov(ot, gold_idx), ov(otr, gold_idx)
print(f"overlap(bf16-torch , fp32-gold) = {o_t:.4f}")
print(f"overlap(tf32-triton, fp32-gold) = {o_tr:.4f}")
print("RESULT:", "PASS" if (o_tr >= o_t - 1e-4 and o_tr >= 0.98) else "FAIL")
