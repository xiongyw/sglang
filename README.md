# SGLang: RX 7900 XTX / Qwen3.8-27B TP2 appliance branch

[简体中文](README.zh-CN.md)

This branch is a narrow, measured specialization of SGLang for the following appliance:

```text
Hardware:   2 × AMD Radeon RX 7900 XTX, gfx1100, 24 GiB each
Runtime:    ROCm 7.2.4 / HIP 7.2.26015 / PyTorch 2.11.0+rocm7.2
Target:     Qwen3.8-27B W4A16 AutoRound-GPTQ
Drafter:    Qwen3.8-27B DFlash2 W4A16
Topology:   TP=2, PP=1
Activations: BF16
KV cache:   BF16
```

It is not a general gfx1100 backend and is not a generic TP=2 solution. Unsupported model layouts, quantization formats, GPU profiles, and speculative layouts should not be treated as covered by this branch.

## Branch lineage and scope

The branch is just a "copy" from JWC's [`StevenChenSE/sglang`](https://github.com/StevenChenSE/sglang) `gfx1100-support` work. It differs from that work in three deliberate ways:

1. **Commit reorganization:** the appliance changes were reworked into small, independently reviewable commits instead of preserving the original monolithic history.
2. **Upstream rebase:** the stack is rebased onto the current `xiongyw/main` tip at the time of publication, while retaining the appliance commits on top.
3. **Narrowed target:** the supported scope is explicitly the two-card RX 7900 XTX TP=2/PP=1 Qwen3.8-27B W4A16 appliance. TP=1 and PP=2 experiments are historical evidence, not supported branch targets.

The active branch name is:

```text
7900xtx-qwen38-27b-tp2pp1
```

## What is included

- gfx1100 AOT build and RDNA3 W4A16 GPTQ GEMM support.
- Qwen3.8-27B AutoRound GPTQ checkpoint loading, including TP-aware `g_idx` validation.
- TP=2/PP=1 DFlash2 support on the hybrid GDN/Mamba target.
- Correctness fixes for the packed W4A16 DFlash2 drafter:
  - GPTQ zero-point convention;
  - BF16 activation range preservation in RDNA3 W4A16 kernels;
  - quantized DFlash context projection loading.
- Decode CUDA graphs for the production batch-1 profile.
- Exact-target gfx1100 split-KV verify admission for the Qwen3.8 geometry.
- Guarded fused context-KV materialization for the packed W4A16 DFlash2 QKV layout.
- Custom all-reduce support and correctness tests, while the measured production recipe retains NCCL/RCCL.

The current HEAD is the fused-KV commit. DSpark exploration, HIP DFlash fast-path restoration, and further hardware-dependent tuning remain separate backlog items.

## Hardware caveat: the P2P rig is not ideal

The two cards currently have asymmetric PCIe links: one is Gen2 ×8 and the other is Gen3 ×8. Cross-card traffic is therefore constrained by the slower link. This is a property of the target hardware configuration, not a branch target; the link-speed cause is intentionally not investigated here.

Consequences:

- TP=2 correctness is validated, but performance numbers are specific to this topology.
- Prefill and communication-sensitive measurements may improve after the slower card is retrained to Gen3 ×8.
- Do not compare these numbers directly with a symmetric PCIe or NVLink system.
- HIP peer access is required and was verified for the target topology in both directions.

## Measured results

Measurements below are from the real Qwen3.8-27B W4A16 target and TP=2/PP=1 DFlash2 drafter on this two-card rig. They are decode-only medians from five-run controls unless noted otherwise; graphs were enabled for the production profile.

### Production profile: batch 1, short context

```text
Profile                         Median decode       Result
Target-only, graphs on           38.459 tok/s       control
DFlash2 W4A16, graphs on        129.096 tok/s       3.36× target-only
```

Acceptance remained nonzero and outputs matched the target-only greedy control in the correctness probes.

### Long-context curve: graphs on

```text
Prompt depth       Target-only       DFlash2       DFlash2 advantage
~1.6K                 38.459          129.096          3.36×
~8.5K                 37.563           92.973          2.48×
~32K                  33.616           46.503          1.38×
~200K                   —              ~33 tok/s       measured envelope
```

Acceptance stayed approximately constant through the depth curve; the decline is per-step verification cost, not draft acceptance collapse. The branch's split-KV verify port restores much of the long-context loss: the measured ~32K speculative profile reached approximately 85 tok/s in a later control, but compare only runs with identical harness settings when drawing a final number.

### Fused context-KV A/B

```text
Context       Fused path       Sequential fallback       Delta
~8.5K           114.505             114.353              +0.13%
~32K             90.484              90.304              +0.20%
```

The fused path is retained as a guarded correctness-preserving enablement. These measurements do not establish an end-to-end throughput win on this workload.

### Concurrency

At ~8.5K context and four submitted sessions, DFlash2 reached approximately 239 tok/s aggregate versus approximately 123 tok/s target-only. At ~32K, the scheduler admitted only three sessions concurrently; the run was capacity/scheduling-limited.

These results are appliance measurements, not performance guarantees.

### TP scaling and the drafter's depth limit

Measured 2026-09-26 on `7900xtx-qwen38-27b-tp2pp1` at `b691c69a9f`, same filler-log prompt, greedy, 96 forced tokens, median of four warm runs, no concurrency.

```text
Target-only decode, ~25k prompt tokens
  TP=2                 34.05 tok/s      29.36 ms/step
  TP=1                 25.95 tok/s      38.53 ms/step
  speedup 1.31x against an ideal 2.0x.
  Of that shortfall, 5.85 ms per step is measured collective time: a decode
  step issues 128 all-reduces (two row-parallel reductions per layer x 64
  layers) and a dependent chain of 128 such calls costs 45.7 us each at
  (1, 5120) bf16, measured on a 2-rank torchrun bench. TP=1 pays none of it,
  so removing it lifts TP=2 to 1.64x; the remainder is per-rank kernel
  efficiency at halved N.

Speculative vs plain at matching prompts
  depth     target-only   DFlash2 w512   DFlash2 no window flag
  ~25k      34.05         45.45          28.24
  ~39k      32.05         39.05          -
  ~59k      29.55         23.02          -
  ~115k     24.19         17.07          16.31
```

Three qualifications follow, and they bound where the drafter is worth using.

1. The drafter crosses from a win to a loss between ~39k and ~59k prompt tokens. The decline is monotone, so treat the crossover as ~45-50k rather than a sharp boundary. Both terms push the same way: acceptance falls with depth while the speculative step grows faster than the plain step (68.4 -> 119.5 ms against 29.4 -> 41.4 ms). At ~115k the target-only path is 29-33% faster than every speculative setting measured, and it is not a window problem: windows of 512, 1024, 2048 and 4096 all land within 4.7% at 119.5-120.2 ms per step, with acceptance 1.50-2.35 against the 2.89 tokens per step that a 115k speculative step needs to break even.
2. The ~200K envelope of roughly 33 tok/s is not reproduced under this harness: the best speculative arm at ~115k measured 17.07 tok/s. Treat that envelope as unreproduced until a matching configuration is identified.
3. Greedy output is not bit-reproducible across server restarts on this build. A target-only continuation was compared with itself across two restarts and diverged at token 48 of 96, while the two speculative runs were identical. The claim above that outputs matched the target-only greedy control therefore holds only up to the first divergence, and any bit-identity check needs repeats rather than a single agreeing run.
4. A speculative number needs more repetitions than a plain one. The target-only warm runs agreed to 0.1% at every depth (32.03-32.05 tok/s at ~39k), while the speculative warm runs spanned 26.45-42.25 tok/s at ~39k and 22.55-40.54 at ~59k. Compare speculative arms on medians of several runs, never on a single run.

## Setup and installation

The commands below assume Debian/Linux, ROCm 7.2.4, Python 3.12, and a user-provided installation layout. Set these variables to match your machine before running the commands:

```bash
export SGLANG_DIR="${SGLANG_DIR:-$PWD}"
export VENV_DIR="${VENV_DIR:-$HOME/venv/sglang}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/models/safetensors/Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ}"
export DRAFT_MODEL="${DRAFT_MODEL:-$HOME/models/safetensors/syvai/Qwen3.8-27B-DFlash2-W4A16}"
export SETUP_DIR="${SETUP_DIR:-$HOME/sglang-setup}"
export PYTHON="${PYTHON:-$VENV_DIR/bin/python}"
export LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-$SETUP_DIR/launch_tp2_dflash_8080.sh}"
```

Use equivalent checkpoint paths if your models are stored elsewhere. The commands below use these variables rather than assuming a particular username or home-directory layout.

### 1. Preflight

```bash
rocminfo | grep -E 'Name:|gfx'
hipconfig --version
readlink -f /opt/rocm

$PYTHON -c \
  'import torch; print(torch.__version__, torch.version.hip, torch.cuda.device_count())'
```

Expected reference values:

```text
gfx1100
ROCm 7.2.4
2.11.0+rocm7.2 7.2.26015 2
```

Do not run `rocm-smi --gpureset` on these gfx1100 cards; it has previously been observed to lock the PCIe root port.

### 2. Preserve ROCm Torch and install SGLang without dependency replacement

```bash
source "$VENV_DIR/bin/activate"
cd "$SGLANG_DIR/python"

SGLANG_BUILD_RUST_EXTS=none \
  uv pip install --no-build-isolation --no-deps -e .

python -c \
  'import torch, sglang; print(torch.__version__, torch.version.hip); print(sglang.__file__)'
```

Do not use a dependency-resolving install that can replace the ROCm Torch build with CUDA Torch. Install missing Python imports individually if the launcher reports them.

### 3. Build gfx1100 AOT kernels

```bash
source "$VENV_DIR/bin/activate"
cd "$SGLANG_DIR/python/sglang/kernels/aot"

PYTORCH_ROCM_ARCH=gfx1100 \
  "$PYTHON" setup_rocm.py build_ext --inplace

DEST="$VIRTUAL_ENV/lib/python$(python -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/sgl_kernel"
mkdir -p "$DEST"
cp -a python/sgl_kernel/. "$DEST/"

python -c 'import sgl_kernel, sgl_kernel.allreduce; print("sgl_kernel OK")'
```

The full `python/sgl_kernel/` package must be copied, not only the shared object.

### 4. Verify the target and drafter paths

```bash
test -f "$TARGET_MODEL/config.json"
test -f "$DRAFT_MODEL/config.json"
```

The W4A16 DFlash2 alternative requires a different explicit drafter quantization setting and is not the default production recipe.

### 5. Keep the target MTP tensors in BF16

The target checkpoint stores its MTP tensors as BF16, while the original checkpoint metadata contains positive GPTQ rules matching `mtp.*`. A fresh copy of the target checkpoint must have those MTP rules replaced with one negative BF16 exclusion rule before serving.

This patch is idempotent and changes only the target `config.json`; it does not modify model weights:

```bash
export TARGET_MODEL="${TARGET_MODEL:-$HOME/models/safetensors/Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ}"
$PYTHON - <<'PY'
import json
import os

path = os.path.join(os.environ["TARGET_MODEL"], "config.json")
with open(path, encoding="utf-8") as f:
    config = json.load(f)

quant = config.setdefault("quantization_config", {})
dynamic = quant.setdefault("dynamic", {})
for key in list(dynamic):
    if "mtp" in key.lower():
        del dynamic[key]
dynamic["-:.*mtp.*"] = {"bits": 16, "group_size": 128}

with open(path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
print(f"patched {path}")
PY
```

Verify the patch before starting the server:

```bash
$PYTHON - <<'PY'
import json
import os

path = os.path.join(os.environ["TARGET_MODEL"], "config.json")
config = json.load(open(path, encoding="utf-8"))
dynamic = config["quantization_config"]["dynamic"]
mtp = {k: v for k, v in dynamic.items() if "mtp" in k.lower()}
assert mtp == {"-:.*mtp.*": {"bits": 16, "group_size": 128}}, mtp
assert not any(k.startswith("+:") for k in mtp), mtp
print("MTP BF16 exclusion verified:", mtp)
PY
```

The branch also contains a loader-side safeguard for the GPTQ/AutoRound quantization names, but the checkpoint metadata patch is part of reproducing the validated local setup and should be applied to a fresh target download.

## Start and stop the appliance

The tested launcher is below. Save it as `$LAUNCH_SCRIPT` (by default
`$SETUP_DIR/launch_tp2_dflash_8080.sh`), make it executable, and run it: this is the script the
measurements in this README were produced with. Every path, port and knob is an environment
variable with a default, so nothing here is tied to a particular home directory.

```bash
#!/usr/bin/env bash
# Launch the TP=2/PP=1 Qwen3.8-27B W4A16 + DFlash2 appliance.
# Every path, port and knob below is an environment variable with a default, so
# the script is not tied to a particular home directory.
set -euo pipefail

SGLANG_ROOT="${SGLANG_ROOT:-${SGLANG_DIR:-$HOME/github/sglang}}"
VENV_PYTHON="${VENV_PYTHON:-${PYTHON:-$HOME/venv/sglang/bin/python}}"
TARGET_MODEL="${TARGET_MODEL:-$HOME/models/safetensors/Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ}"
DRAFT_MODEL="${DRAFT_MODEL:-$HOME/models/safetensors/syvai/Qwen3.8-27B-DFlash2-W4A16}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
SERVED_MODEL="${SERVED_MODEL:-qwen3.8-27b-7900xtx-dflash-tp2}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
LOG_DIR="${LOG_DIR:-${SETUP_DIR:-$HOME/sglang-setup}/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/tp2-dflash-$PORT.log}"
HEALTH_HOST="${HEALTH_HOST:-127.0.0.1}"

usage() {
  printf 'Usage: %s [--kill]\n' "$0"
  printf '  no option  Launch the TP=2/PP=1 DFlash2 server on %s:%s\n' "$HOST" "$PORT"
  printf '  --kill     Stop the process group owning port %s\n' "$PORT"
}

port_pid() {
  ss -lptnH "sport = :${PORT}" 2>/dev/null \
    | grep -oP 'pid=\K[0-9]+' \
    | head -1 || true
}

fail_if_port_busy() {
  local pid
  pid="$(port_pid)"
  if [[ -n "${pid}" ]]; then
    printf 'Port %s is already owned by PID %s.\n' "$PORT" "$pid" >&2
    printf 'Inspect it with: ps -fp %s\n' "$pid" >&2
    exit 1
  fi
}

kill_server() {
  local pid pgid remaining
  pid="$(port_pid)"
  if [[ -z "${pid}" ]]; then
    printf 'Port %s is already free.\n' "$PORT"
    exit 0
  fi
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ' || true)"
  printf 'Stopping port %s owner PID %s (process group %s).\n' "$PORT" "$pid" "${pgid:-unknown}"
  if [[ -n "${pgid}" && "${pgid}" != "0" ]]; then
    kill -TERM -- "-${pgid}" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi
  for _ in $(seq 1 90); do
    remaining="$(port_pid)"
    [[ -z "${remaining}" ]] && { printf 'Stopped.\n'; return 0; }
    sleep 1
  done
  remaining="$(port_pid)"
  if [[ -n "${remaining}" ]]; then
    printf 'Graceful stop timed out; killing PID %s.\n' "${remaining}" >&2
    kill -KILL "${remaining}" 2>/dev/null || true
  fi
  [[ -z "$(port_pid)" ]] && printf 'Stopped.\n'
}

case "${1:-}" in
  "") ;;
  --kill) kill_server; exit 0 ;;
  --help|-h) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

for path in "$SGLANG_ROOT" "$TARGET_MODEL" "$DRAFT_MODEL"; do
  [[ -e "$path" ]] || { printf 'Missing path: %s\n' "$path" >&2; exit 1; }
done
[[ -x "$VENV_PYTHON" ]] || { printf 'Missing Python: %s\n' "$VENV_PYTHON" >&2; exit 1; }

fail_if_port_busy
mkdir -p "$(dirname "$LOG_FILE")"

export PYTHONPATH="$SGLANG_ROOT/python/sglang/kernels/aot/python:$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
export SGLANG_USE_AITER_AR="${SGLANG_USE_AITER_AR:-0}"
export SGLANG_DFLASH_PHASE_TIMING="${SGLANG_DFLASH_PHASE_TIMING:-0}"
export SGLANG_DFLASH_DEVICE_TIMING="${SGLANG_DFLASH_DEVICE_TIMING:-0}"
unset SGLANG_USE_1STAGE_ALLREDUCE SGLANG_ENABLE_DETERMINISTIC_INFERENCE

# Compact draft-cache window for DFlash2. With the flag absent the engine default
# is None, and dflash_worker_v2 sets use_compact_draft_cache = window is not None,
# i.e. the drafter attends over the full context. Measured on this appliance at
# ~25k tokens: 28.24 tok/s without the flag versus 45.45 tok/s with 512, at an
# unchanged per-step cost (acceptance 1.50 -> 3.30; the flag changes how many
# tokens a step yields, not what it costs). 512 is also the setting the depth-curve
# figures in this branch were produced with. Set SPECULATIVE_DRAFT_WINDOW=none to
# restore the engine default.
SPECULATIVE_DRAFT_WINDOW="${SPECULATIVE_DRAFT_WINDOW:-512}"
WINDOW_ARGS=()
if [[ "${SPECULATIVE_DRAFT_WINDOW}" != "none" ]]; then
  WINDOW_ARGS=(--speculative-draft-window-size "${SPECULATIVE_DRAFT_WINDOW}")
fi

cd "$SGLANG_ROOT"
printf 'Starting branch: '
git branch --show-current
printf 'Commit: '
git rev-parse --short HEAD
printf 'Endpoint: http://%s:%s\n' "$HOST" "$PORT"
printf 'Log: %s\n' "$LOG_FILE"
printf 'Fused KV: enabled (set SGLANG_DFLASH2_DISABLE_FUSED_KV=1 before running to disable)\n'
printf 'Draft window: %s\n' "${SPECULATIVE_DRAFT_WINDOW}"

setsid "$VENV_PYTHON" -m sglang.launch_server \
  --model-path "$TARGET_MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tp-size 2 \
  --pp-size 1 \
  --quantization gptq \
  --dtype bfloat16 \
  --mamba-ssm-dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --attention-backend triton \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$MEM_FRACTION" \
  --max-running-requests 1 \
  --max-mamba-cache-size 8 \
  --cuda-graph-bs-decode 1 \
  --triton-attention-num-kv-splits 16 \
  --chunked-prefill-size 2048 \
  --decode-log-interval 10 \
  --sleep-on-idle \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT_MODEL" \
  --speculative-draft-model-quantization compressed-tensors \
  --speculative-num-draft-tokens 8 \
  --speculative-dflash-block-size 8 \
  --speculative-draft-attention-backend triton \
  "${WINDOW_ARGS[@]}" \
  ${EXTRA_ARGS:-} \
  >"$LOG_FILE" 2>&1 < /dev/null &

launcher_pid=$!
printf 'Launcher PID: %s\n' "$launcher_pid"

for _ in $(seq 1 300); do
  if curl -fsS "http://${HEALTH_HOST}:${PORT}/health" >/dev/null 2>&1; then
    server_pid="$(port_pid)"
    printf 'READY\n'
    printf 'Server PID: %s\n' "${server_pid:-unknown}"
    printf 'Test with: curl http://%s:%s/v1/models\n' "$HOST" "$PORT"
    printf 'Log tail:\n'
    tail -40 "$LOG_FILE"
    exit 0
  fi
  if ! kill -0 "$launcher_pid" 2>/dev/null; then
    printf 'Server exited before becoming healthy. Last log lines:\n' >&2
    tail -100 "$LOG_FILE" >&2 || true
    exit 1
  fi
  sleep 2
done

printf 'Timed out waiting for /health. Last log lines:\n' >&2
tail -100 "$LOG_FILE" >&2 || true
exit 1
```

The defaults resolve from the setup variables above: `${SGLANG_DIR:-$HOME/github/sglang}`,
`${PYTHON:-$HOME/venv/sglang/bin/python}`, `$TARGET_MODEL`, `$DRAFT_MODEL` and
`${SETUP_DIR:-$HOME/sglang-setup}`; the server log goes to `$SETUP_DIR/logs/tp2-dflash-8080.log`
unless `LOG_FILE` overrides it.

Start on all host interfaces, port 8080:

```bash
"$LAUNCH_SCRIPT"
```

The script launches TP=2/PP=1 with:

```text
host:                 0.0.0.0
port:                 8080
context length:       262144
mem fraction:         0.90
decode CUDA graph:    batch size 1
attention backend:    triton
verify KV splits:     16
chunked prefill:      2048
reasoning parser:     qwen3
tool-call parser:     qwen3_coder
DFlash2 drafter:      W4A16 compressed-tensors
draft tokens/block:   8 / 8
draft window:         512 (compact draft cache)
```

The draft window line is load-bearing for the measurements below. `--speculative-draft-window-size` defaults to unset, and `dflash_worker_v2.py` sets `use_compact_draft_cache = draft_window_size is not None`, so launching without the flag leaves the drafter attending over the full context. Measured at ~25k prompt tokens with everything else identical: 28.24 tok/s without the flag versus 45.45 tok/s with `512`, at an unchanged per-step cost (acceptance 1.50 vs 3.30). Set `SPECULATIVE_DRAFT_WINDOW=none` to restore the engine default.

Check the API from another machine on the LAN:

```bash
curl http://<host-ip>:8080/v1/models
curl http://<host-ip>:8080/health
```

Stop only the server owning port 8080:

```bash
"$LAUNCH_SCRIPT" --kill
```

The script refuses to overwrite an occupied port during launch and kills the owning process group only when `--kill` is explicitly requested.

## Validation checklist

Before trusting a new build or rebase:

```bash
cd "$SGLANG_DIR"

$PYTHON -m py_compile \
  python/sglang/srt/speculative/dflash_utils.py \
  python/sglang/srt/speculative/dflash_worker_v2.py \
  python/sglang/srt/models/dflash.py \
  python/sglang/srt/models/dspark.py \
  python/sglang/kernels/ops/speculative/fused_kv_materialize.py \
  python/sglang/srt/layers/attention/triton_backend.py

$PYTHON -m unittest \
  test.registered.unit.spec.test_dflash_fused_kv_quant \
  test.registered.unit.spec.test_dflash_hip_admission \
  test.registered.unit.spec.test_dflash_hip_accept_policy \
  test.registered.unit.spec.test_dflash_hip_greedy_accept \
  test.registered.unit.spec.test_dflash_hip_prepare_policy \
  test.registered.unit.spec.test_dflash_hip_selector_policy \
  test.registered.unit.test_7900xtx_gptq_identity_gidx \
  test.registered.unit.test_7900xtx_gptq_dispatch \
  test.registered.unit.test_7900xtx_gptq_adapter
```

The reference sanity run after rebasing onto `xiongyw/main` passed 25/25 tests. A real appliance validation should also confirm:

- both TP ranks initialize;
- `/health` and `/v1/models` return 200;
- logs show target-verify and draft CUDA graph capture;
- logs show `reasoning_parser=qwen3` and `tool_call_parser=qwen3_coder` in `server_args`;
- a tool-enabled request returns structured `tool_calls`, not raw `<tool_call>` text;
- greedy output matches the no-drafter control on the fixed probe set.

## References

- Original gfx1100 support: https://github.com/StevenChenSE/sglang/tree/gfx1100-support
- Official upstream: https://github.com/sgl-project/sglang
- DFlash2 model: https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2
