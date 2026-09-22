# SGLang：RX 7900 XTX / Qwen3.8-27B TP2 appliance 分支

这是一个面向以下 appliance 配置、经过实测的 SGLang 专用分支：

```text
硬件：      2 × AMD Radeon RX 7900 XTX，gfx1100，每张 24 GiB
运行时：    ROCm 7.2.4 / HIP 7.2.26015 / PyTorch 2.11.0+rocm7.2
目标模型：  Qwen3.8-27B W4A16 AutoRound-GPTQ
起草模型：  Qwen3.8-27B DFlash2 W4A16
拓扑：      TP=2，PP=1
激活：      BF16
KV Cache：  BF16
```

本分支不是通用的 gfx1100 后端，也不是通用 TP=2 方案。其他模型布局、量化格式、GPU 配置或 speculative decoding 布局不属于本分支的验证范围。

## 分支来源与范围

本分支基本上就是拷贝了 JWC 的 [`StevenChenSE/sglang`](https://github.com/StevenChenSE/sglang) `gfx1100-support` 工作。相对于该工作，本分支有三个次要变化：

1. **重新组织提交：** 将 appliance 相关修改拆分为较小、可独立审查的提交，不保留原始的大型单体提交历史。
2. **重新基于上游：** 将这些提交 rebase 到发布时 `xiongyw/main` 的最新提交之上。
3. **收窄目标范围：** 明确只支持双 RX 7900 XTX、TP=2、PP=1、Qwen3.8-27B W4A16 appliance。TP=1 和 PP=2 实验仅作为历史验证证据，不属于本分支的支持目标。

当前分支名：

```text
7900xtx-qwen38-27b-tp2pp1
```

## 包含的内容

- gfx1100 AOT 构建以及 RDNA3 W4A16 GPTQ GEMM 支持。
- Qwen3.8-27B AutoRound GPTQ checkpoint 加载，包括 TP-aware `g_idx` 校验。
- 针对 hybrid GDN/Mamba 目标模型的 TP=2/PP=1 DFlash2 支持。
- packed W4A16 DFlash2 起草模型的正确性修复：
  - GPTQ zero-point convention；
  - RDNA3 W4A16 kernel 保留 BF16 激活动态范围；
  - 正确加载量化 DFlash context projection。
- 生产 batch-1 配置的 decode CUDA graphs。
- 针对 Qwen3.8 几何形状的精确 gfx1100 split-KV verify admission。
- packed W4A16 DFlash2 QKV 布局的受保护 fused context-KV materialization。
- custom all-reduce 支持和正确性测试；实测生产配置仍保留 NCCL/RCCL。

当前 HEAD 是 fused-KV 提交。DSpark 探索、HIP DFlash fast path 恢复以及进一步的硬件相关调优仍在 backlog 中。

## 硬件限制：当前 P2P 拓扑并不理想

两张卡的 PCIe 链路不对称：一张是 Gen2 ×8，另一张是 Gen3 ×8。因此跨卡通信会受较慢链路限制。这是目标硬件配置的属性，不是本分支的调优目标；不在本分支中调查链路速度差异的原因。

影响如下：

- TP=2 正确性已经验证，但性能数据只适用于这个拓扑。
- 较慢的卡重新训练到 Gen3 ×8 后，prefill 和通信敏感型测量可能改善。
- 不要直接将这些数据与对称 PCIe 或 NVLink 系统比较。
- 双向 HIP peer access 是必要条件，目标拓扑的两个方向均已验证。

## 实测结果

以下数据来自真实 Qwen3.8-27B W4A16 目标模型和 TP=2/PP=1 DFlash2 起草模型，运行于上述双卡拓扑。除非另有说明，数据是五次运行的 decode-only 中位数；生产配置启用了 decode graphs。

### 生产配置：batch 1、短上下文

```text
配置                             decode 中位数       结果
仅目标模型，graphs on              38.459 tok/s       对照
DFlash2 W4A16，graphs on          129.096 tok/s       目标模型的 3.36×
```

正确性 probe 中，起草模型保持非零 acceptance，输出与仅目标模型的 greedy 对照一致。

### 长上下文曲线：graphs on

```text
prompt 深度       仅目标模型       DFlash2       DFlash2 优势
~1.6K              38.459          129.096          3.36×
~8.5K              37.563           92.973          2.48×
~32K               33.616           46.503          1.38×
~200K                   —           ~33 tok/s       实测范围
```

整个深度曲线中 acceptance 基本保持不变；性能下降来自每步 verify 成本，而不是起草模型 acceptance 崩溃。分支中的 split-KV verify port 恢复了大部分长上下文损失：后续控制测量中约 32K speculative profile 达到约 85 tok/s，但只有在 harness 设置完全一致时才应比较具体数值。

### Fused context-KV A/B

```text
上下文          fused path       sequential fallback       差异
~8.5K             114.505             114.353              +0.13%
~32K               90.484              90.304              +0.20%
```

fused path 作为受保护的正确性保持型 enablement 保留。这些数据尚未证明它在本 workload 上带来端到端吞吐提升。

### 并发

在约 8.5K 上下文、四个并发 session 下，DFlash2 达到约 239 tok/s aggregate，而仅目标模型约为 123 tok/s。约 32K 时调度器只能同时接纳三个 session；该运行受容量和调度限制。

这些结果是 appliance 测量值，不是性能保证。

## 安装与配置

以下命令假设 Debian/Linux、ROCm 7.2.4、Python 3.12，以及由用户自行指定的安装布局。执行前请根据你的机器设置这些变量：

```bash
export SGLANG_DIR="${SGLANG_DIR:-$PWD}"
export VENV_DIR="${VENV_DIR:-$HOME/venv/sglang}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/models/safetensors/Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ}"
export DRAFT_MODEL="${DRAFT_MODEL:-$HOME/models/safetensors/syvai/Qwen3.8-27B-DFlash2-W4A16}"
export SETUP_DIR="${SETUP_DIR:-$HOME/sglang-setup}"
export PYTHON="${PYTHON:-$VENV_DIR/bin/python}"
export LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-$SETUP_DIR/launch_tp2_dflash_8080.sh}"
```

如果模型存储在其他位置，请设置对应的 checkpoint 路径。下面的命令使用这些变量，不依赖特定用户名或 home 目录布局。

### 1. 预检查

```bash
rocminfo | grep -E 'Name:|gfx'
hipconfig --version
readlink -f /opt/rocm

$PYTHON -c \
  'import torch; print(torch.__version__, torch.version.hip, torch.cuda.device_count())'
```

目标环境值：

```text
gfx1100
ROCm 7.2.4
2.11.0+rocm7.2 7.2.26015 2
```

不要在这些 gfx1100 卡上运行 `rocm-smi --gpureset`；此前观察到它可能锁死 PCIe root port。

### 2. 保留 ROCm Torch，避免依赖安装覆盖它

```bash
source "$VENV_DIR/bin/activate"
cd "$SGLANG_DIR/python"

SGLANG_BUILD_RUST_EXTS=none \
  uv pip install --no-build-isolation --no-deps -e .

python -c \
  'import torch, sglang; print(torch.__version__, torch.version.hip); print(sglang.__file__)'
```

不要使用会重新解析依赖的安装方式，以免将 ROCm Torch 替换为 CUDA Torch。如果 launcher 报告缺少 Python 模块，请逐个安装缺失模块。

### 3. 构建 gfx1100 AOT kernels

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

必须复制完整的 `python/sgl_kernel/` package，不能只复制 shared object。

### 4. 验证目标模型和起草模型路径

```bash
test -f "$TARGET_MODEL/config.json"
test -f "$DRAFT_MODEL/config.json"
```

W4A16 DFlash2 是默认生产起草模型。BF16 DFlash2 需要不同的显式起草模型量化设置，不属于默认配置。

### 5. 保持目标模型 MTP tensors 为 BF16

目标 checkpoint 将 MTP tensors 存储为 BF16，而原始 checkpoint metadata 含有匹配 `mtp.*` 的正向 GPTQ 规则。新下载的目标 checkpoint 在启动服务前必须将这些 MTP 规则替换为一个负向 BF16 exclusion rule。

下面的 patch 幂等，只修改目标模型的 `config.json`，不会修改模型权重：

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

启动服务前验证 patch：

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

分支还包含针对 GPTQ/AutoRound quantization names 的 loader-side safeguard，但 checkpoint metadata patch 仍是复现已验证配置的一部分；对新下载的目标 checkpoint 应执行该 patch。

## 启动与停止 appliance

仓库包含经过测试的 launcher：

```text
$LAUNCH_SCRIPT
```

在所有 host interfaces 上以 8080 端口启动：

```bash
"$LAUNCH_SCRIPT"
```

该脚本使用以下 TP=2/PP=1 配置：

```text
host：                 0.0.0.0
port：                 8080
context length：       262144
mem fraction：         0.90
decode CUDA graph：    batch size 1
attention backend：    triton
verify KV splits：     16
chunked prefill：      2048
reasoning parser：     qwen3
tool-call parser：     qwen3_coder
DFlash2 drafter：      W4A16 compressed-tensors
```

从 LAN 上的另一台机器检查 API：

```bash
curl http://<host-ip>:8080/v1/models
curl http://<host-ip>:8080/health
```

只停止占用 8080 端口的服务：

```bash
"$LAUNCH_SCRIPT" --kill
```

正常启动时，如果端口已被占用，脚本会拒绝覆盖；只有显式传入 `--kill` 时才会停止端口所属的 process group。

## 验证清单

在信任新的构建或 rebase 之前：

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

rebase 到 `xiongyw/main` 后的参考 sanity run 通过了 25/25 个测试。真正的 appliance 验证还应确认：

- 两个 TP rank 都成功初始化；
- `/health` 和 `/v1/models` 返回 200；
- 日志显示 target-verify 和 draft CUDA graph capture；
- `server_args` 中显示 `reasoning_parser=qwen3` 和 `tool_call_parser=qwen3_coder`；
- 启用 tools 的请求返回结构化 `tool_calls`，而不是原始 `<tool_call>` 文本；
- greedy 输出与固定 probe 集合的无起草模型对照一致。

## 参考

- 原始 gfx1100 支持来源：https://github.com/StevenChenSE/sglang/tree/gfx1100-support
- 官方上游：https://github.com/sgl-project/sglang
- DFlash2 模型：https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2
