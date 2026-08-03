# Network & NCCL — the reliability foundation

Two Sparks run one model over a **RoCE point-to-point link** (a single cable
between the two ConnectX NICs). Get this layer right or you get silent slowness
and multi-minute "wedges." Three things matter, in order.

## 1. RDMA must actually engage (not TCP fallback)

The #1 silent failure: NCCL falls back to **TCP sockets** instead of RDMA, because
the container can't see the RDMA device. Symptom when we hit it (2026-05, on the
build of the day): **~12 tok/s instead of ~30+**, plus marker-free stalls under
load. Expect the same shape of collapse against today's higher baseline — the
`via NET/IB` check below is the reliable test, not a tok/s threshold.

- The `docker run` **must** include `--device=/dev/infiniband --cap-add=IPC_LOCK
  --ulimit memlock=-1:-1` (the start scripts do).
- Set `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` to your RoCE
  HCA/interface (see `ibv_devices` and `ip -br link`).
- **Verify:** `docker logs vllm-ds4 | grep "via NET/IB"` should return many lines
  and `grep "NET/Socket"` should return **zero**. If you see `NET/IB : No device
  found`, the device passthrough is missing.
- Set the RoCE link MTU to 9000 on both nodes if supported: `sudo ip link set
  dev <ROCE_IFACE> mtu 9000`.

## 2. NCCL 2.30.4 — the wedge fix (critical)

DeepSeek-V4 on dual Spark **deadlocks** under load with old NCCL: repeated
`"No available shared memory broadcast block found in 60 seconds"` then a hung
engine. The fix is **NCCL 2.30.4** (`libnccl2=2.30.4-1+cuda13.2`).

The catch: PyTorch ships its **own bundled NCCL** (often 2.28.x) and uses it even
if the host/container has 2.30.4 installed. Force the newer one with `LD_PRELOAD`:

```
-e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libnccl.so.2.30.4
```

(adjust the path to wherever 2.30.4's `libnccl.so.2` lives in your image; the
start scripts assume the aarch64 system path). **Verify after boot:**
`docker logs vllm-ds4 | grep "NCCL version"` must show **2.30.4**, not 2.28.x.
`torch.cuda.nccl.version()` reports the *compiled* version and will still say the
bundled one — trust the NCCL banner in the logs, not torch.

## 3. Cable / PHY sanity

These NICs run hot and a marginal cable/transceiver corrupts frames. If you see
flapping or odd switch behavior, check the NIC PHY counters
(`ethtool -S <iface> | grep -E "rx_err|symbol|fcs"`). FEC-corrected errors
(`rx_symbol_err_phy = 0`) are fine; uncorrected errors mean reseat/replace the
cable.

## What "good" looks like at boot
- `NCCL version 2.30.4`
- many `via NET/IB`, zero `via NET/Socket`
- `Application startup complete` + `/health` → 200
