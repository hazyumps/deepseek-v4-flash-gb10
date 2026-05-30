"""Prefill TTFT + decode-rate probe. Sends a UNIQUE large prompt (defeats the
prefix cache) and measures time-to-first-token (prefill) and decode tok/s.

Usage:  VLLM_URL=http://10.255.0.1:8000 python3 prefill_test.py 8600
Expected on a healthy dual-GB10 (this repo's config): see docs/VALIDATION.md.
"""
import sys, os, time, json, urllib.request, random

URL = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/") + "/v1/chat/completions"
target_tok = int(sys.argv[1]) if len(sys.argv) > 1 else 8600

random.seed(os.getpid() ^ target_tok)
nonce = "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(12))
topics = ["RDMA queue pairs", "NVMe submission queues", "MoE expert routing",
          "tensor parallel all-reduce", "KV cache paging", "flash attention tiling"]
lines, i = [], 0
while sum(len(x) for x in lines) < int(target_tok * 3.6):
    lines.append(f"[{nonce}-{i:05d}] Section {i}: notes on {topics[i % len(topics)]}; "
                 f"idx {i*7 % 9973}, lat {(i*131) % 1000}us, tput {(i*977) % 100000}.")
    i += 1
prompt = "\n".join(lines) + f"\n\nQuestion: what nonce prefixes every section? Answer only the nonce."

body = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 40, "temperature": 0.0, "chat_template_kwargs": {"thinking": False},
        "stream": True, "stream_options": {"include_usage": True}}
req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                            headers={"Content-Type": "application/json"})

t0 = time.time(); t_first = None; usage = None; ans = []
with urllib.request.urlopen(req, timeout=900) as resp:
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        ch = obj.get("choices") or []
        if ch:
            c = (ch[0].get("delta", {}) or {}).get("content") or (ch[0].get("delta", {}) or {}).get("reasoning") or ""
            if c:
                if t_first is None:
                    t_first = time.time()
                ans.append(c)
t_end = time.time()

pt = usage.get("prompt_tokens") if usage else None
ct = usage.get("completion_tokens") if usage else None
ttft = (t_first - t0) if t_first else None
dec = (t_end - t_first) if t_first else None
print(f"prompt_tokens : {pt}")
print(f"TTFT (prefill): {ttft:.2f}s" + (f"  ({pt/ttft:,.0f} tok/s prefill)" if pt and ttft else ""))
print(f"decode        : {ct/dec:.1f} tok/s" if dec and ct else "decode: n/a")
print(f"answer        : {''.join(ans)[:40]!r}  (expected nonce: {nonce})")
