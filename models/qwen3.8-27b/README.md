# V100 inference comparison

Date: 2026-10-22

## Bottom line

- The native QUASAR + DFlash2 path is the fastest tested single-stream path:
  about **139 HTTP completion tok/s**, versus **113.7 tok/s** for the current
  v100-skinny 1.2.2 service and **92.1 tok/s** for the isolated 1.5.1 port of
  the uncensored mixed ModelOpt checkpoint.
- The uncensored checkpoint on the 1.5.1 port remains the best tested
  concurrent path: **278-292 aggregate tok/s at C4**. Raising the QUASAR 8K
  profile to a 0.90 GPU-memory budget reaches **230.95 aggregate tok/s**; the
  engine ran three short requests together and queued the fourth.
- GGUF is not presently a route to QUASAR parity. The optimized GGUF work is
  newer than 1.5.1 and its own four-V100 figures remain roughly 12-19% behind
  native NVFP4.
- NInfer's V100 port cannot be reproduced on this host as-is: it requires the
  CUDA 12.8 toolkit and a single 32 GiB V100-class device, while this host has
  CUDA toolkit 12.0 and 16 GiB V100s. Updating the fork is a substantial
  re-port, not a small merge.
- No host-RAM limit was encountered. A simple offline bit-format conversion
  cannot create QUASAR weights: QUASAR is a QAT-trained representation. GGUF
  conversion is possible, but is not expected to improve speed over the
  existing checkpoint.

## Hardware and measurement

- GPUs: 4 x Tesla V100-SXM2-16GB, driver 580.178.04.
- The RTX 3090 and RTX 3080 were not used.
- Canonical requests and exact procedure are in [`payloads/`](payloads/).
- Runner: [`scripts/run_http_bench.py`](../../scripts/run_http_bench.py).
- Serial: one unreported warm-up followed by five fixed 256-token requests.
- Batch: one unreported wave followed by three measured four-request waves.
- Throughput is completion tokens divided by client wall time. It includes
  scheduling and the small prefill and is deliberately not 1Cat's
  server-side "pure decode" metric.

## 1Cat-vLLM 1.5.1, native QUASAR + DFlash2

Stack:

- `/home/tangles/1cat-vllm/.venv`
- `vllm 1.5.1`, PyTorch `2.10.0+cu128`, CUDA runtime 12.8
- TP4, Flash-Attention-V100, QPN2 NVFP4, DFlash2 K=7, E4M3 KV
- Target: `/mnt/nvme/Qwen3.8-27B-QUASAR-NVFP4`
- Draft: `/mnt/nvme/Qwen3.8-27B-DFlash2`
- The matching NInfer manifest identifies source revisions
  `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4@d8e6fbf` and
  `z-lab/Qwen3.8-27B-DFlash2@50307d4`.

### Results

| Profile | Serial aggregate | Serial median | Four-client aggregate | Effective cache concurrency |
|---|---:|---:|---:|---:|
| MML 32K, GPU mem 0.83 | 139.21 | 136.23 | 144.59 | 1.02x |
| MML 8K, GPU mem 0.83 | 138.66 | 133.97 | 144.21 | 1.62x |
| MML 8K, GPU mem 0.90 | 139.72 | 134.59 | **230.95** | 2.75x |

Units are tok/s. Raw output:

- [`quasar-http-bench.txt`](raw/quasar-http-bench.txt)
- [`quasar-8k-http-bench.txt`](raw/quasar-8k-http-bench.txt)
- [`quasar-8k-090-http-bench.txt`](raw/quasar-8k-090-http-bench.txt)
- deterministic sanity response:
  [`quasar-natural-96.json`](raw/quasar-natural-96.json)

The apparent C4 figures are not true four-way decode. At 32K the engine
reported 33,587 KV tokens and 1.02x maximum full-context concurrency; logs
showed one running request and the rest waiting. At 8K it reported 1.61 GiB
available KV and 1.62x concurrency, and still ran one of these cache-aligned
requests at a time. DFlash acceptance on the fixed-list payload was about
34-36%.

At a 0.90 memory budget, available KV rose to 2.71 GiB / 22,495 tokens and
2.75x full-8K concurrency. The short benchmark requests briefly ran three at
once (`Running: 3`, one waiting), lifting measured aggregate throughput to
230.95 tok/s. This is a useful concurrency result but still not a full four
simultaneous 8K-context guarantee.

An attempted 4K profile with smaller Mamba pages was not a valid benchmark:
the altered page contract fell off the QPN2 route and failed while trying the
Marlin fallback (`size_n=4120` not divisible by 64). This is a runtime
contract limitation, not a model or host-RAM failure.

## Existing uncensored checkpoint comparison

These earlier matched measurements came from the isolated 1.5.1 port in
`/home/tangles/v100-skinny-151-port`.

| Stack and checkpoint | Serial aggregate | Warm C4 aggregate |
|---|---:|---:|
| v100-skinny / 1Cat-vLLM 1.2.2, uncensored mixed NVFP4+FP8 | 113.70 | 29.31-31.77 |
| isolated 1.5.1 port, same uncensored checkpoint, FP8 KV | 92.05 | 278.47-291.84 |

The 1.5.1 port keeps the relevant E5 fixes from
[v100-skinny PR #9](https://github.com/dnv2003/v100-skinny/pull/9), including
pre-mutation accepted-state synchronization and rejection of out-of-vocabulary
speculative sentinels. It consumes the published checkpoint losslessly:
NVFP4 is repacked for QPN2 and FP8 bytes are preserved while fused scales are
expanded for QPN8.

## GGUF

1Cat-vLLM 1.5.1 only has upstream's experimental GGUF loader. Its
documentation describes that route as under-optimized and primarily useful
for reducing memory footprint. The serious TurboMind-backed V100 work is on
1Cat development `main` (observed revision
`6f3557c973b4602905037e086fd4acba9b0d867f`), not the 1.5.1 release.

The development report for Qwen3.8-27B UD-Q4_K_M on four
V100-SXM2-32GB cards, TP4, without MTP, reports:

| Concurrency | GGUF tok/s | native NVFP4 tok/s |
|---:|---:|---:|
| 1 | 59.49 | 65.55 |
| 4 | about 207-215 | 249 |
| 8 | 394.61 | 471.15 |
| 16 | 674.61 | 831.07 |

That leaves GGUF around 12-19% slower, even on current development code and
32 GiB cards. Development GGUF+DFlash2 profiling exists, but not a released,
comparable end-to-end result. There are no local GGUF weights, so downloading
another large model and building an unreleased stack is not justified by the
available evidence.

References:

- [GGUF TurboMind design](https://github.com/1CatAI/1Cat-vLLM/blob/main/docs/design/gguf_turbomind_model.md)
- [GGUF+DFlash2 projection ledger](https://github.com/1CatAI/1Cat-vLLM/blob/main/docs/design/gguf_dflash2_projection_ledger.md)

## NInfer V100

The research checkout is retained locally at `tools/ninfer-v100/` and is not
part of this repository.

- Fork: [`geoffwatts/ninfer-v100`](https://github.com/geoffwatts/ninfer-v100)
  at `b37d0dd3e1163b9d802d8bccfa89918bf68d793e`
- Upstream: [`Neroued/ninfer`](https://github.com/Neroued/ninfer)
  at `64d10fbf` when checked
- Divergence: 32 fork-only commits and 206 upstream-only commits
- A synthetic merge reported **207 conflicts**. Updating this fork means
  reapplying the V100 backend to a changed runtime architecture.

Local artifact:

- `/mnt/nvme/Qwen3.8-27B-QUASAR-NVFP4-NInfer/qwen3_8_27b_nvfp4.ninfer`
- 19,782,432,752 bytes, container v3
- SHA-256
  `29bd2a543dd779e28fa6cbc4ff532447e3f861018384b3c5e4a8b98ba4b2587b`
- Includes text, vision, MTP, and DFlash2; its manifest says the v2-to-v3
  upgrade preserved all tensor payloads and did not requantize weights.

No local benchmark was produced. The fork explicitly requires CUDA toolkit
12.8 because CUDA 13 removed Volta offline compilation. This host has only
`nvcc 12.0`; Ninja and the FFmpeg development headers are also absent.
There are no release binaries or CI artifacts to use instead. More
importantly, NInfer is a one-GPU resident runtime and its published V100
qualification uses V100-PCIe-32GB. The 19.8 GB artifact plus runtime buffers
is not a viable resident load on any of this host's 16 GiB V100s.

Published figures such as 218.98 tok/s are a short-context, highly
predictable, single-32GB-V100 ceiling and are not comparable to this HTTP
payload. Its own varied-text figures are much lower (roughly 68 tok/s at 2K,
93 at 8K, 55 at 32K, and 44 at 150K). Renting a 32 GiB V100 with CUDA 12.8
could reproduce the engine, but would not establish a deployable path for
this server.

## Strata-V100

[`jmnargi/Strata-V100`](https://github.com/jmnargi/Strata-V100), checked at
`b06925cf98223633b74006e19570dfc05e42ac4c`, is active and genuinely
V100-specific. It targets Qwen3.8-Flash-Next using host RAM/NVMe offload and
adaptive expert residency. Reported conversation decode on two
V100-PCIe-16GB cards is around 79-87 tok/s, with approximately 1-2K tok/s
prefill.

That is valuable for Flash-Next capacity and very long offloaded contexts,
but it is not a faster path for a resident dense 27B checkpoint on four
V100s. The fork also differs from its upstream by hundreds of commits.
Setting it up would answer a different capacity/long-context question, so it
was not benchmarked here.

## Abliterated INT8 W8A16 DFlash2 checkpoint

[`lued/Qwen3.8-27B-huihui-abliterated-INT8-W8A16-DFlash2`](https://huggingface.co/lued/Qwen3.8-27B-huihui-abliterated-INT8-W8A16-DFlash2)
is about 28 GiB for the target plus about 2.02 GiB for DFlash2 and is intended
for a patched dual-RTX-3090 stack. It may fit by TP4 across these V100s, but
1Cat's fast V100 paths are QPN2 NVFP4 and QPN8 FP8, not generic W8A16.
Consequently it would likely use generic kernels and trail both native
QUASAR and the existing mixed checkpoint.

It is potentially useful as an uncensoring/quality reference on dual 3090s,
not as the likely fastest V100 deployment. It was not downloaded.

## Recommendation

For this server, retain two practical choices:

1. Current v100-skinny for the fastest single interactive stream.
2. The isolated 1.5.1 mixed-checkpoint port for multi-agent throughput, after
   its context-capacity and quality gates are accepted.

Native QUASAR+DFlash2 establishes the performance ceiling but does not solve
the uncensored-weight requirement. Neither GGUF, NInfer, Strata, nor generic
INT8 currently offers a credible faster uncensored path on four 16 GiB
V100s. Do not spend on a requantization job unless the source format and
runtime kernel target are first shown to be losslessly compatible; a
QUASAR-format checkpoint would require the unpublished training/QAT process,
not a simple format conversion.
