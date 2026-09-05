#!/bin/bash
# Entrypoint wrapper for glm-5.3f (GLM-5.3-Flash-Uncensored, glm5_next).
#
# glm5_next has a HETEROGENEOUS KV cache: 34 KDA linear-attention (recurrent
# mamba-class) layers + 11 DSA/MLA full-attention layers. The stock in-process
# LMCacheConnectorV1 does NOT subclass SupportsHMA, so vLLM auto-disables the
# hybrid KV manager and then dies trying to unify the two KV spec types
# ("failed to convert the KV cache specs to one unified type").
#
# The fix on this newer vLLM (0.1.dev20051, HMA-native) is LMCache's MULTI-
# PROCESS connector: LMCacheMPConnector subclasses SupportsHMA and validates
# mamba step-alignment + KV groups, so vLLM keeps the hybrid manager on and the
# KDA recurrent state is served correctly. It talks to a co-located LMCache MP
# cache server (ZMQ, tcp://localhost:5555) that holds the L1 (DRAM) + L2 (disk)
# tiers. This wrapper starts that server, waits for it, then execs vLLM.
#
# ENABLE_LMCACHE=0 skips the server AND the connector flags entirely, giving a
# byte-for-byte fallback to the known-good no-LMCache config.
set -u

LMCACHE_PORT="${LMCACHE_PORT:-5555}"
LMCACHE_L1_GB="${LMCACHE_L1_GB:-24}"
LMCACHE_DISK_GB="${LMCACHE_DISK_GB:-400}"
LMCACHE_DISK_PATH="${LMCACHE_DISK_PATH:-/lmcache/disk}"
# LMCache chunk size must be a multiple of vLLM's block size. In mamba-align
# mode glm5_next forces the attention block size to 1792 (to match the KDA
# mamba page size), so the chunk size must be a multiple of 1792.
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-1792}"

EXTRA_ARGS=()

if [ "${ENABLE_LMCACHE:-1}" = "1" ]; then
  mkdir -p "${LMCACHE_DISK_PATH}"
  echo "[lmcache-entry] starting LMCache MP server: L1=${LMCACHE_L1_GB}G DRAM, L2=${LMCACHE_DISK_GB}G disk @ ${LMCACHE_DISK_PATH}, port ${LMCACHE_PORT}, chunk ${LMCACHE_CHUNK_SIZE}"
  # Disable the worker-liveness reaper (--worker-reap-timeout-seconds 0). The
  # reaper reclaims a GPU registration once it has been "silent" for the reap /
  # registration-grace window (default 3600s), judged by register/PING/store/
  # retrieve activity. On this build the worker HeartbeatThread's PINGs do not
  # refresh server-side liveness (server logs pinged=False), so an idle context
  # was reaped at the 3600s grace — after which every store/retrieve silently
  # no-ops (L1 stays 0, External prefix cache hit rate 0.0%). We own the whole
  # box: the MP server is (re)started by this same entrypoint whenever vLLM
  # (re)starts, so a crashed worker never leaves a stale context behind for the
  # reaper to clean up. Disabling it keeps the KDA/MLA registration alive across
  # arbitrarily long idle gaps so LMCache actually stores and restores.
  python3 -m lmcache.v1.multiprocess.server \
    --host localhost --port "${LMCACHE_PORT}" \
    --chunk-size "${LMCACHE_CHUNK_SIZE}" \
    --l1-size-gb "${LMCACHE_L1_GB}" \
    --eviction-policy LRU \
    --worker-reap-timeout-seconds 0 \
    --separate-object-groups \
    --l2-adapter "{\"type\":\"fs_native\",\"base_path\":\"${LMCACHE_DISK_PATH}\",\"max_capacity_gb\":${LMCACHE_DISK_GB}}" \
    > "${LMCACHE_DISK_PATH}/mp_server.log" 2>&1 &
  LM_PID=$!

  # Wait for the ZMQ server to bind before vLLM tries to hand-shake in the
  # connector __init__ (a missing server there is a hard boot failure).
  bound=0
  for _ in $(seq 1 90); do
    if ! kill -0 "${LM_PID}" 2>/dev/null; then
      echo "[lmcache-entry] MP server exited during startup; tail of log:"
      tail -n 40 "${LMCACHE_DISK_PATH}/mp_server.log"
      echo "[lmcache-entry] refusing to start vLLM without its cache server"
      exit 1
    fi
    if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('localhost',${LMCACHE_PORT}))==0 else 1)"; then
      bound=1; break
    fi
    sleep 1
  done
  if [ "${bound}" != "1" ]; then
    echo "[lmcache-entry] MP server did not bind port ${LMCACHE_PORT} in time; tail of log:"
    tail -n 40 "${LMCACHE_DISK_PATH}/mp_server.log"
    exit 1
  fi
  echo "[lmcache-entry] MP server listening on ${LMCACHE_PORT}; launching vLLM with LMCacheMPConnector"

  KVCFG='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":'"${LMCACHE_PORT}"'}}'
  # KDA (mamba-class) layers must snapshot reusable per-block recurrent state.
  EXTRA_ARGS+=("--mamba-cache-mode=align")
  EXTRA_ARGS+=("--kv-transfer-config=${KVCFG}")
  # KV block-count alignment for LMCache's subpaged re-view.
  # In mamba-align mode vLLM forces the attention *logical* block to 1792 and
  # the DSA indexer tensor's kernel-page count comes out as num_gpu_blocks * 7
  # (glm5_next tensor layout, kv_cache_utils.py:1645). LMCache re-views that
  # tensor from kernel pages -> logical blocks and REQUIRES the page count to be
  # a multiple of ratio 28 (=1792/64), else: "kernel page count N is not a
  # multiple of the logical/kernel block ratio 28". Since pages = num_blocks*7,
  # that means num_gpu_blocks must be a multiple of 4 (7*4 = 28).
  # The profiler's natural count at GLM_UNC_UTIL=0.95 is 2929 (2929*7 = 20503,
  # not divisible by 28). Pin num_gpu_blocks to the largest multiple of 4 just
  # below it: 2924 (2924*7 = 20468 = 731*28; ~5-block/0.17% cushion for profile
  # drift; concurrency ~34x, one 1M-token request fits many times over).
  # COUPLED TO util=0.95: if GLM_UNC_UTIL changes, the natural count moves and
  # this must be recomputed. The boot log prints the natural value it replaces
  # ("Overriding num_gpu_blocks=<natural> with num_gpu_blocks_override=2924");
  # keep override <= natural or KV allocation will OOM.
  EXTRA_ARGS+=("--num-gpu-blocks-override=${LMCACHE_KV_BLOCKS:-2924}")
else
  echo "[lmcache-entry] ENABLE_LMCACHE=0 — serving without LMCache (known-good fallback)"
fi

exec vllm serve "$@" "${EXTRA_ARGS[@]}"
