#!/usr/bin/env python3
"""Cross-request contamination gate for glm-5.3f + LMCache.

Fires N concurrent long prompts, each with a UNIQUE needle passphrase and a
unique filler salt (so every request has distinct KV content end to end). Each
reply must contain exactly its own passphrase — any cross-request KV mixup
(wrong blocks restored, shared mamba state, indexer pool crosstalk) shows up
as the wrong passphrase or a corrupted reply.

Run once after a fresh boot (store phase), then cold-restart the container and
run again with the same args (restore phase): per-request sha256 must match
byte-for-byte between the two runs, and no reply may contain another request's
passphrase.

Usage: contamination_gate.py <n_requests> <base_tokens> <phase_tag>
Writes per-request shas to /tmp/conc-gate/<phase_tag>.json for comparison.
Env: VLLM_API_KEY (never printed).
"""
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.request

N_REQ = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BASE_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
TAG = sys.argv[3] if len(sys.argv) > 3 else "run"

KEY = os.environ["VLLM_API_KEY"]
URL = "http://localhost:8001/v1/chat/completions"

FRUITS = ["MANGO", "PAPAYA", "LYCHEE", "DURIAN", "GUAVA", "LOQUAT", "SAPOTE",
          "JUJUBE", "POMELO", "SALAK", "MEDLAR", "QUINCE"]
FABRICS = ["SATIN", "TWEED", "CHIFFON", "DENIM", "VELOUR", "MUSLIN", "TAFFETA",
           "BROCADE", "GINGHAM", "ORGZA", "CAMBRIC", "DAMASK"]


def build_prompt(idx):
    # Unique needle + unique numeric code per request.
    code = f"{FRUITS[idx % len(FRUITS)]}-{FABRICS[idx % len(FABRICS)]}-{1000 + idx * 917}"
    needle = (
        f"IMPORTANT FACT: the secret vault passphrase is {code}. "
        "Remember this exact passphrase."
    )
    # Unique size and filler salt so KV content differs everywhere.
    n_tok = BASE_TOK + idx * 2048
    n_lines = max(200, n_tok // 10)
    depth = 0.15 + (idx * 0.09) % 0.75
    lines = []
    for i in range(n_lines):
        lines.append(
            f"Line {i:06d} [stream-{idx:02d}]: warehouse {chr(65 + (i + idx) % 26)} "
            f"logged {100 + (i * 7 + idx * 13) % 900} routine crate transfers with "
            f"no notable anomalies in shift {(i + idx) % 3}."
        )
    insert_at = int(n_lines * depth)
    lines[insert_at] = f"Line {insert_at:06d} [stream-{idx:02d}]: {needle}"
    body = "\n".join(lines)
    prompt = (
        "You are auditing a long log. Read it carefully.\n\n"
        + body
        + "\n\nQuestion: What is the exact secret vault passphrase mentioned in "
        "the log? Reply with ONLY the passphrase, nothing else."
    )
    return code, prompt


def fire(idx):
    code, prompt = build_prompt(idx)
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
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=1800) as resp:
        out = json.load(resp)
    dt = time.time() - t0
    msg = out["choices"][0]["message"]
    reasoning = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    sha = hashlib.sha256((reasoning + "\x1e" + content).encode()).hexdigest()
    return idx, code, content.strip(), sha, dt


results = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=N_REQ) as ex:
    for idx, code, answer, sha, dt in ex.map(fire, range(N_REQ)):
        results[idx] = {"code": code, "answer": answer, "sha": sha, "secs": round(dt, 1)}

all_codes = {v["code"] for v in results.values()}
ok = True
for idx in sorted(results):
    r = results[idx]
    own = r["code"] in r["answer"]
    foreign = [c for c in all_codes - {r["code"]} if c in r["answer"]]
    status = "OK" if own and not foreign else "FAIL"
    if status == "FAIL":
        ok = False
    print(f"[{TAG}-{idx:02d}] {status} own={own} foreign={foreign} "
          f"answer={r['answer']!r} sha={r['sha']} {r['secs']}s")

os.makedirs("/tmp/conc-gate", exist_ok=True)
with open(f"/tmp/conc-gate/{TAG}.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"[{TAG}] {'ALL OK' if ok else 'CONTAMINATION DETECTED'} ({N_REQ} requests)")
sys.exit(0 if ok else 1)
