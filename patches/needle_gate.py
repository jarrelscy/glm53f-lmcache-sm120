#!/usr/bin/env python3
"""Deep-needle lossless gate for glm-5.3f + LMCache.

Builds a long filler prompt with a unique needle sentence buried at a chosen
depth, sends it at temperature 0, and prints a stable fingerprint of the reply
(needle answer + full-text sha256 + timing + prompt tokens). Run it once before a
container restart and once after: the sha256 must match byte-for-byte (lossless
restore of the KDA recurrent state + MLA/indexer caches), and the second run
should be much faster (LMCache hit) rather than a full recompute.

Usage: needle_gate.py <approx_prompt_tokens> <depth_frac 0..1> <tag>
Env: VLLM_API_KEY (read from environment; never printed).
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

N_TOK = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
DEPTH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.55
TAG = sys.argv[3] if len(sys.argv) > 3 else "run"

KEY = os.environ["VLLM_API_KEY"]
URL = "http://localhost:8001/v1/chat/completions"

# Unique needle -- fixed string so pre/post restart prompts are byte-identical.
NEEDLE_CODE = "TAMARIND-VELVET-7391"
NEEDLE = (
    f"IMPORTANT FACT: the secret vault passphrase is {NEEDLE_CODE}. "
    "Remember this exact passphrase."
)

# ~7 filler words/line, ~10 tokens/line. Build enough lines for N_TOK tokens.
lines = []
n_lines = max(200, N_TOK // 10)
for i in range(n_lines):
    lines.append(
        f"Line {i:06d}: the quarterly logistics ledger records routine warehouse "
        f"transfers and inventory counts with no notable anomalies here."
    )
insert_at = int(n_lines * DEPTH)
lines[insert_at] = f"Line {insert_at:06d}: {NEEDLE}"
body = "\n".join(lines)

prompt = (
    "You are auditing a long log. Read it carefully.\n\n"
    + body
    + "\n\nQuestion: What is the exact secret vault passphrase mentioned in the "
    "log? Reply with ONLY the passphrase, nothing else."
)

req = {
    "model": "glm-5.3f",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "max_tokens": 2048,
    "seed": 3407,
}
data = json.dumps(req).encode()
r = urllib.request.Request(
    URL, data=data,
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
)
t0 = time.time()
with urllib.request.urlopen(r, timeout=1800) as resp:
    out = json.load(resp)
dt = time.time() - t0

msg = out["choices"][0]["message"]["content"] or ""
reasoning = out["choices"][0]["message"].get("reasoning_content") or ""
finish = out["choices"][0]["finish_reason"]
usage = out.get("usage", {})
full = reasoning + "\x1e" + msg
sha = hashlib.sha256(full.encode()).hexdigest()

print(f"[{TAG}] prompt_tokens={usage.get('prompt_tokens')} "
      f"completion_tokens={usage.get('completion_tokens')} "
      f"latency={dt:.1f}s finish={finish}")
print(f"[{TAG}] needle_found={NEEDLE_CODE in (msg + reasoning)} answer={msg!r}")
print(f"[{TAG}] sha256={sha}")
if os.environ.get("NEEDLE_DUMP"):
    print(f"[{TAG}] reasoning_head={reasoning[:400]!r}")
    print(f"[{TAG}] reasoning_tail={reasoning[-200:]!r}")
    print(f"[{TAG}] msg_head={msg[:400]!r}")
