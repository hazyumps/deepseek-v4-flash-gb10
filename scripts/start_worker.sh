#!/bin/bash
# DeepSeek-V4-Flash — WORKER node (rank 1) on NVIDIA GB10 / DGX Spark (sm_121).
# Joins the head via the `mp` distributed backend. MUST mirror the head's
# EP / MTP / cudagraph / mem-util / patch for TP coherence.
# Run start_head.sh on node 1 FIRST, then this on node 2.
set -euo pipefail
[ -f "$(dirname "$0")/../env.sh" ] && source "$(dirname "$0")/../env.sh"

IMAGE="${IMAGE:?build per docs/BUILD.md}"
HEAD_IP="${HEAD_IP:-10.255.0.1}"          # the HEAD node's RoCE IP (rendezvous master)
WORKER_IP="${WORKER_IP:-10.255.0.2}"      # this node's RoCE IP
ROCE_IFACE="${ROCE_IFACE:-enp1s0f0np0}"
NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
PATCH="${PATCH:-$(cd "$(dirname "$0")/../patches" && pwd)/sm12x_deep_gemm_fallbacks.py}"
INPATH=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/deepseek_v4_ops/sm12x_deep_gemm_fallbacks.py

docker rm -f vllm-ds4-worker 2>/dev/null || true

docker run -d \
  --name vllm-ds4-worker \
  --restart unless-stopped \
  --runtime nvidia --gpus all \
  --ipc host --network host --shm-size 16g \
  --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
  --device=/dev/infiniband \
  -v "$HOME/spark/models:/root/.cache/huggingface" \
  -v "$HOME/spark/vllm-cache:/root/.cache/vllm" \
  -v "$HOME/spark/triton-cache:/root/.triton/cache" \
  -v "$PATCH:$INPATH" \
  -e VLLM_HOST_IP=$WORKER_IP \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_SOCKET_IFNAME=$ROCE_IFACE \
  -e GLOO_SOCKET_IFNAME=$ROCE_IFACE \
  -e NCCL_DEBUG=INFO \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libnccl.so.2.30.4 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_TRITON_MLA_SPARSE=1 \
  "$IMAGE" vllm serve "$MODEL" \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --enable-expert-parallel --distributed-executor-backend mp \
  --nnodes 2 --node-rank 1 --headless --master-addr $HEAD_IP --master-port 29519 \
  --kv-cache-dtype fp8 --block-size 256 --enable-prefix-caching \
  --max-model-len 393216 --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.80 \
  --no-enable-flashinfer-autotune \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}' \
  --load-format safetensors
echo "worker launched. cold boot ~4-5 min; head will reach 'Application startup complete'."
