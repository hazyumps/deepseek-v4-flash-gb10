#!/bin/bash
# Watch the head node's cold boot to "Application startup complete".
# Run ON the head node (needs local `docker logs vllm-ds4`). ~4-5 min.
set -u
URL="${VLLM_URL:-http://localhost:8000}"
for i in $(seq 1 40); do
  L=$(docker logs vllm-ds4 2>&1 | tail -400)
  nccl=$(printf '%s\n' "$L" | grep -m1 -ioE 'NCCL version [0-9.]+')
  ib=$(printf '%s\n' "$L" | grep -c 'via NET/IB')
  conc=$(printf '%s\n' "$L" | grep -oE 'Maximum concurrency for [0-9,]+ tokens[^x]*x' | tail -1)
  err=$(printf '%s\n' "$L" | grep -iE 'out of memory|traceback|RuntimeError' | tail -1)
  up=$(printf '%s\n' "$L" | grep -m1 'Application startup complete')
  hc=$(curl -s -m5 -o /dev/null -w '%{http_code}' "$URL/health" 2>/dev/null)
  echo "[t+$((i*15))s] nccl='${nccl:-pending}' net/ib=$ib health=$hc ${conc:+| $conc}"
  [ -n "$err" ] && echo "  !! $err"
  if [ -n "$up" ] && [ "$hc" = "200" ]; then
    echo "=== STARTUP COMPLETE — $nccl | NET/IB=$ib | $conc ==="; exit 0
  fi
  sleep 15
done
echo "=== TIMEOUT (10 min) — check: docker logs vllm-ds4 ==="
