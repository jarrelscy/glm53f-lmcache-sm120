# GLM-5.3-Flash (glm5_next) + LMCache on SM120 — lossless KV offload for a hybrid KDA/DSA model

Serving recipe and patch set for running **GLM-5.3-Flash-Uncensored-NVFP4**
(`orcarouter/GLM-5.3-Flash-Uncensored-NVFP4`, arch `glm5_next`) on 4x NVIDIA
RTX PRO 6000 Blackwell (SM120, 96 GB each) with **LMCache** KV offload that
survives cold restarts — including the model's heterogeneous KV cache:

- 34 **KDA linear-attention** layers (mamba-class recurrent state, served with
  vLLM `--mamba-cache-mode=align`)
- 11 **DSA sparse attention** layers (MLA latent + a slot-compressed fp8
  indexer `k_cache` + a tiny `tail_cache` boundary buffer)

Stock LMCache 0.5.4 cannot serve this model correctly. This repo contains the
overlay files (bind-mounted over the installed packages, no rebuild needed)
that make it work, plus the gate script used to prove the restore is lossless.

## Quick start

```bash
cp .env.example .env    # set VLLM_API_KEY
docker compose up -d
# wait for "Application startup complete", then:
curl -H "Authorization: Bearer $VLLM_API_KEY" http://localhost:8001/v1/models
```

Weights are expected at
`/data/huggingface/hub/models--orcarouter--GLM-5.3-Flash-Uncensored-NVFP4/snapshots/414d5eeeb2e49e09a58cfed1a4c6f839f5796b00`
(~178 GiB, 62 shards). Adjust the first `command:` arg if your snapshot hash
differs.

## Image

`vllm/vllm-openai:glm53-flash-x86_64-cu130` is a local build of upstream
[vllm-project/vllm](https://github.com/vllm-project/vllm) at commit
`487ecf187` (version `0.1.dev20051+g487ecf187`) using vLLM's standard
`docker/Dockerfile` with a CUDA 13.0 base, plus `pip install lmcache==0.5.4`.
Key package versions inside:

| package | version |
|---|---|
| vllm | 0.1.dev20051+g487ecf187 |
| lmcache | 0.5.4 |
| torch | 2.13.0+cu130 |
| flashinfer | 0.6.17 (+cu130 jit-cache) |

The overlays in `patches/` are written against exactly these versions. If you
build a different vLLM/LMCache, diff the originals before mounting.

## Why stock LMCache fails on glm5_next, and what each overlay fixes

The model registers 67 KV tensors per GPU: 11 indexer `k_cache` + 11 MLA
latent + 11 `tail_cache` + 34 KDA state pages. vLLM groups them into 6 engine
groups; chunk size must equal the logical block size (1792 tokens).

### LMCache overlays (`lmcache/integration/vllm/...`)

**`kv_cache_group_edits.py`** — three fixes on top of stock:

1. *Per-layer inner specs.* glm5_next wraps a group's specs in
   `UniformTypeKVCacheSpecs`; stock 0.5.4 passes the wrapper (aggregate page
   size, `compress_ratio=1`) to the edit rules, so the indexer caches are
   mis-matched and boot dies with "N kernel pages do not tile the logical
   page". The overlay resolves each layer's real inner spec first.
2. *`_SlidingWindowTailViewEdit`.* The DSA `tail_cache` is a rank-4
   dim-0-padded sliding-window tensor; LMCache's format detector maps rank-4
   to a non-block-axis format and the MP server dies in `register_kv_caches`.
   The edit folds it to a rank-3 block-axis view.
3. **`_CompressedIndexerSubpagedViewEdit` — the cold-restore corruption fix.**
   The indexer `k_cache` (`MLAAttentionSpec` with `compress_ratio=4`)
   registers *kernel pool pages*, not logical blocks: shape
   `(num_blocks*7, 64, 132)` — 7 contiguous 64-slot pool pages per logical
   1792-token block (59,136 bytes). LMCache's compression path infers geometry
   from the tensor and takes dim-0 as the block axis, so it transfers 8,448
   bytes per block **from kernel page `b`, which physically belongs to block
   `b//7`**. Store "succeeds", every block count checks out, MLA and KDA bytes
   round-trip perfectly — but on cold restore the indexer pools for the whole
   restored prefix are unwritten garbage, and all 11 DSA layers run their
   top-k token selection over it. Output is deterministically corrupt while
   everything in the logs looks healthy. The edit re-views the tensor at
   logical-block granularity, `(N*7, 64, 132) -> (N, 448, 132)`, computing the
   page ratio from **bytes** (`spec.page_size_bytes // kernel_page_bytes = 7`,
   not the token/slot ratio 28). After the re-view, the inferred slot
   compression (1792/448 = 4) matches the spec's declared `compress_ratio`
   exactly. Identity page tiling (block `b` = pages `7b..7b+6`) is guaranteed
   by the model's `Glm5NextIndexerCache` ("the storage block is virtually
   split into pool pages").

**`kv_cache_groups.py`** — excludes real sliding-window layers (the
`tail_cache`) from server-side kernel grouping. Their KV is deliberately not
transferred: the tail is a 4-token (`index_kpool`) boundary buffer that
prefill re-seeds, and a partial hit always leaves >=1 chunk to recompute, so
exclusion is sound. Without this the server expects `num_chunks * 448` block
IDs the store never sends and skips every store ("STORE block ID underflow").
Align-mode Mamba layers get `sw_size_tokens = block_size` (a one-block
window: only the last state snapshot is needed on restore).

**`lmcache_mp_metadata.py`** — excludes the sliding-window groups from the
store-coverage `min` over engine groups (the tail group allocates 1 block of
4 tokens and would floor coverage to 4 tokens = zero chunks stored forever)
and zeroes their block IDs. Adds `[store-probe]`/`[grp-probe]` diagnostics.

**`lmcache_mp_connector.py`** — detects real sliding-window groups via their
specs and threads `sw_excluded_groups` into the metadata calls.

**`vllm_multi_process_adapter.py`** — failed retrieves now flag their block
IDs in `error_block_ids` so vLLM recomputes them. Stock only logs the error;
because the MP server transfers object groups sequentially and stops at the
first failure, a swallowed failure resumes the request on unwritten VRAM =
silent corruption.

### vLLM overlays

SM120 (Blackwell workstation) support for glm5_next's DSA sparse MLA:
`backend.py`, `flashinfer_mla_sparse_sm120.py`, `cuda.py`,
`glm5next_model.py`, `glm5next_mtp.py`. Plus `chat_template.jinja` (zai-org
template; the checkpoint's own template is text-only and breaks tool-result
dedup).

### Entrypoint: `lmcache-mp-entry.sh`

Starts the LMCache MP cache server (`python -m lmcache.v1.multiprocess.server`,
ZMQ tcp://localhost:5555, L1 DRAM + L2 `fs_native` disk at `/lmcache/disk`,
`--separate-object-groups`, chunk 1792, reaper disabled) and then execs
`vllm serve` with `--kv-transfer-config` (LMCacheMPConnector),
`--mamba-cache-mode=align` and `--num-gpu-blocks-override`.

`ENABLE_LMCACHE=0` gives a plain `vllm serve` (no LMCache anywhere in the
path) — useful as a byte-for-byte reference for the gates below.

### Non-obvious constraints (learned the hard way)

- **Weight-only NVFP4 needs `VLLM_TEST_FORCE_FP8_MARLIN=1`.** This checkpoint
  has no calibrated `input_scale`; the native FP4-activation MoE kernel emits
  token-loop garbage that eventually crashes the KDA Triton kernel with an
  illegal memory access. Marlin w4a16 dequant is the working path.
- **Chunk size must equal the logical block size (1792).** mamba-align forces
  the attention logical block to 1792; LMCache chunks must not split blocks.
- **`--num-gpu-blocks-override` to a multiple of 4.** The DSA indexer re-view
  needs `num_blocks * 7` kernel pages divisible by the MLA ratio 28. Pick the
  largest multiple of 4 <= the natural block count (2924 at util 0.95 here).
- **The MP server reaper must be off** (`--worker-reap-timeout-seconds 0`):
  idle GPU contexts otherwise get reaped after 3600 s and lookups fail with
  "No GPU context found".

## Gating losslessness (do not skip)

`patches/needle_gate.py` builds a long prompt with a unique needle at a chosen
depth and prints a sha256 over `reasoning + "\x1e" + content` at temperature 0.

```bash
export VLLM_API_KEY=...
# 1. reference: recompute path (or ENABLE_LMCACHE=0 boot)
python3 patches/needle_gate.py 5000 0.55 A1     # fresh boot, exercises STORE
# 2. warm repeat — must match A1's sha
python3 patches/needle_gate.py 5000 0.55 A2
# 3. cold restart (docker compose down && up), same prompt — exercises the
#    disk-tier RETRIEVE on a cold engine. Sha must equal A1 byte-for-byte.
python3 patches/needle_gate.py 5000 0.55 B1
```

Verify in the logs that the transfer actually fired (vLLM: "External prefix
cache hit rate"; MP server: "Retrieved N tokens"), and check L2 object sizes:
with the fix, the full-attention object group is
`11 x (1,175,552 + 59,136) = 13,581,568` bytes per chunk (the broken geometry
stored `13,024,000`). The KDA object group is `34 x 1,175,552 = 39,968,768`.

For multi-request safety, `patches/contamination_gate.py` fires N concurrent
prompts that each carry a **unique** passphrase and unique filler salt, so any
cross-request KV mixup (wrong blocks restored, shared recurrent state, indexer
pool crosstalk) surfaces as a foreign passphrase or a changed sha:

```bash
python3 patches/contamination_gate.py 8 16000 CONT-S   # concurrent store
# docker compose down && up, wait for ready
python3 patches/contamination_gate.py 8 16000 CONT-R   # concurrent cold restore
```

Per-request shas must match between the two phases and no reply may contain
another request's passphrase.

A correct-looking retrieve with corrupt output has exactly one more place to
hide: block *counts* are not block *bytes*. If you change anything in this
stack, gate with the sha, not with the logs.

## Gate results (2026-09-05, this exact stack)

All gates below ran on the fixed overlays with `ENABLE_LMCACHE=1`:

- Needle sha match, byte-for-byte, vs the recompute reference: warm repeat,
  cold restart at 11K tokens, cold restart at 116K tokens, and 3 concurrent
  cold restores at depths 0.25/0.55/0.85.
- 8-way concurrent contamination gate: all 8 unique passphrases answered
  correctly in both phases, all 8 shas match store vs cold restore, zero
  foreign passphrases. Restore latency 27 s vs 70 s recompute.
- L2 object sizes confirm the fixed geometry: 13,581,568 B (full-attention
  group) and 39,968,768 B (KDA group) per 1792-token chunk.
- 4 terminal-bench-2.1 tasks (terminus-2 agent, multi-turn agentic load) with
  LMCache on: cancel-async-tasks 1.0, gcode-to-text 1.0 (0.0 without LMCache
  in the original 89-task run), filter-js-from-html 0.0 (also 0.0 originally),
  chess-best-move hit an in-task 120 s command timeout (flake class also seen
  in the original run). Zero engine errors across everything.

## Hardware

4x RTX PRO 6000 Blackwell Max-Q (96 GB, SM120), TP4, PCIe (no NVLink).
~178 GiB weights, ~1M-token context at `--gpu-memory-utilization 0.95`.
