#!/bin/bash
# DeepSeek-V4-Flash-0731 (GA) — HEAD node (rank 0) on NVIDIA GB10 / DGX Spark (sm_121).
# Dual-Spark: TP=2 + expert-parallel over a RoCE point-to-point link, DSpark
# speculative decoding, fp8 KV, 384K context. Serves an OpenAI API on :8000.
#
# EDIT the vars below for your hosts/NICs, or set them in ../env.sh and `source` it.
# See docs/NETWORK.md (RoCE + NCCL 2.30.4) and docs/BUILD.md (the image) first.
set -euo pipefail
[ -f "$(dirname "$0")/../env.sh" ] && source "$(dirname "$0")/../env.sh"

IMAGE="${IMAGE:?build per docs/BUILD.md, e.g. vllm-ds4-sm121:cu130}"
HEAD_IP="${HEAD_IP:-10.255.0.1}"        # this node's RoCE IP (NCCL rendezvous master)
ROCE_IFACE="${ROCE_IFACE:-enp1s0f0np0}" # your RoCE interface (see: ip -br link)
NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0}" # your RDMA HCA (see: ibv_devices)
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}"
CACHE="${CACHE:-$HOME/spark}"           # weights + compile caches live here

docker rm -f vllm-ds4 2>/dev/null || true

docker run -d \
  --name vllm-ds4 \
  --restart unless-stopped \
  --runtime nvidia --gpus all \
  --ipc host --network host --shm-size 16g \
  --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
  --device=/dev/infiniband \
  -v "$CACHE/models:/root/.cache/huggingface" \
  -v "$CACHE/vllm-cache:/root/.cache/vllm" \
  -v "$CACHE/triton-cache:/root/.triton/cache" \
  -v "$CACHE/flashinfer-cache:/root/.cache/flashinfer" \
  -e VLLM_HOST_IP=$HEAD_IP \
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
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e TILELANG_CLEANUP_TEMP_FILES=1 \
  -e DG_JIT_USE_NVRTC=0 \
  -e DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc \
  "$IMAGE" vllm serve "$MODEL" \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --enable-expert-parallel --distributed-executor-backend mp \
  --nnodes 2 --node-rank 0 --master-addr $HEAD_IP --master-port 29519 \
  --kv-cache-dtype fp8 --block-size 256 --enable-prefix-caching \
  --max-model-len 393216 --max-num-seqs 4 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.80 \
  --no-enable-flashinfer-autotune \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}' \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --default-chat-template-kwargs '{"thinking":true}' \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --load-format safetensors --host 0.0.0.0 --port 8000
echo "head launched. watch: docker logs -f vllm-ds4   (then start_worker.sh on node 2)"
