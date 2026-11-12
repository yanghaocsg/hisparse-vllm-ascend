#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build beginner-friendly HiSparse landing notebooks for A5 (PR #20926) and A3 (hisparse-a3)."""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}

def write(side, fname, cells):
    path = os.path.join(ROOT, side, fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = {"cells": cells,
           "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)",
                                       "language": "python", "name": "python3"},
                        "language_info": {"name": "python"}},
           "nbformat": 4, "nbformat_minor": 5}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("written:", os.path.relpath(path, ROOT))

# =====================================================================
# A5 —— PR #20926，Ascend 950（上游基础）
# =====================================================================

A5 = []

A5.append(("01-what-is-hisparse.ipynb", [
md("""# HiSparse 入门 (1/8)：为什么需要它 —— 长上下文 decode 的带宽墙

> **一句话结论**：HiSparse 是一种长上下文 KV 卸载方案。它不把全部历史 KV 塞进设备 HBM，而是每一步只把当前 top-k 命中的那几行搬进设备热区，其余历史留在 Host 侧。

## 1. 先理解问题：decode 一步在干什么

Transformer 生成文本时，每一步只新算 **1 个 token**，但注意力要回看**整条历史序列**：

- 序列越长，每步要读取的 KV 行越多；
- 设备 HBM 容量有限，多请求、长序列放不下全部历史；
- 于是瓶颈往往不是算力，而是 **KV 读取带宽**（数据搬不动）。

## 2. 一个生活比喻

把历史 KV 想象成一座图书馆：

- **Host 池** = 整座图书馆（容量大，但取书慢）；
- **Resident 池** = 你个人书架（已经搬回家的常用书）；
- **Hot 池** = 书桌桌面（当前这一步正在翻的几本书）。

每一步只需要把"当前要翻的书"放到桌面，不必把整座图书馆搬上来。HiSparse 做的就是这套"挑书—搬运—放回"的索引与搬移管理。

## 3. 它解决什么，不解决什么

- **解决**：长上下文下 KV 驻留容量与读取带宽问题；把缺失历史按需搬入设备，注意力在设备热区原地读取。
- **不改变**：注意力的数学公式本身；算子只负责索引、搬移、组装与失效，注意力算术沿用既有稀疏 Flash Attention（SFA）。

## 4. 在 vllm-ascend 里的位置

这套能力分成两层：

- **vLLM Core**：持有 HMA 分配、Coordinator 准入/淘汰、请求生命周期、分组预取、Host 镜像完成与内存预算口径；
- **Ascend 后端**：供应驻留解析、Host 传输映射、共享 slab 视图适配、缓存算子与流/事件围栏。

## 5. 小结

- 长上下文 decode 的主要压力是 **KV 读取带宽与 HBM 容量**；
- HiSparse 用三级存储 + 按需搬移，让设备只保留当前命中的热行；
- 下一篇讲清这三级池的职责边界与统一行格式。

**下一篇**：`02-three-tier-cache.ipynb`
"""),
]))

A5.append(("02-three-tier-cache.ipynb", [
md("""# HiSparse 入门 (2/8)：三级存储池 —— Host / Resident / Hot

> **一句话结论**：历史 KV 分三级存放，越往设备侧越快、越小；三级之间用统一的 **656B 行**和 **int32 物理行号**寻址，`-1` 表示空行。

## 1. 三级池各自装什么

| 层级 | 位置 | 职责 | 比喻 |
| --- | --- | --- | --- |
| Host 池 | CPU 锁页内存（每 rank 私有，ACL AIV 注册） | 保存完整历史，容量最大，按需上行 | 图书馆仓库 |
| Resident 池 | 设备内存 | 每个请求已驻留的历史页表，常驻本请求 | 个人书架 |
| Hot 池 | 设备热区（slab） | 本步 top-k 命中行 + 热区槽表（dgi）+ LRU 顺序表 | 书桌桌面 |

## 2. 两个必须记住的不变量

### 2.1 统一行长：656B

无论数据在哪一级，一行的字节布局完全相同（下一篇细拆）。搬运时按整行拷贝，不需要按字段重组。

### 2.2 物理行号即地址

注意力拿到的是 int32 物理行号，直接换算地址：

```
地址 = pool_base + 行号 × 656
```

没有中间查表。行号为 `-1` 表示这一行是空行（掩码），注意力要跳过它。

## 3. 行号寻址示意（纯计算，不依赖 NPU）
"""),
code("""ROW_BYTES = 656
pool_base = 0  # 示意：实际是设备池起始地址

def row_address(row_idx):
    if row_idx == -1:
        return None  # 空行掩码
    return pool_base + row_idx * ROW_BYTES

for r in [0, 7, 128, -1]:
    print(f"row {r:>4} -> {row_address(r)}")
"""),
md("""## 4. 三级之间怎么流动

1. 新请求的历史先在 Host 池留底；
2. resolver 判断哪些行本步需要，缺失的从 Host 搬到 Resident；
3. 本步 top-k 命中的行进入 Hot 池供注意力直读；
4. 用完按 LRU 顺序淘汰，被覆盖的旧行在索引里失效。

## 5. 小结

- 三级池：Host（完整历史）→ Resident（请求常驻）→ Hot（本步命中）；
- 统一 656B 行、物理行号直读、`-1` 空行掩码是跨层级的契约；
- 下一篇讲 resolver：谁来决定"哪本书上桌面"。

**下一篇**：`03-resolver-on-a5.ipynb`
"""),
]))

A5.append(("03-resolver-on-a5.ipynb", [
md("""# HiSparse 入门 (3/8)：Resolver 解算器 —— 把逻辑 token 翻译成物理行

> **一句话结论**：resolver 每步、每个 indexer 组调用一次，完成两件事：把 top-k 逻辑 token 翻译成可直读的物理行号，并产出本步搬移计划（缺哪些行、淘汰哪些槽）。

## 1. 它在流程里的位置

可以把 resolver 想成前台服务员：

- 手里拿着读者要找的书单（逻辑索引）；
- 查书架和仓库（Host / Resident / Hot 页表）；
- 告诉助手去仓库补哪些书（搬移计划），并把桌面上书的位置报给读者（注意力索引）。

## 2. 算法分五步（语义在 A5/A3 两端一致）

### 2.1 fast path（快速通道）

如果请求"完全驻留"：Resident 页表非空、几何满足、没有 hot 需求，直接把逻辑映射到已驻留行，输出物理索引。**不碰 hash、不碰 dgi、不碰 LRU。**

### 2.2 hash union（未命中行合并）

对走不到快速通道的行：从 Host 池取行号，开放寻址插入 hash 表。同一个 Host 行只保留**最小首次引用位置**，这个顺序后面决定谁先被淘汰。

### 2.3 hot 命中

用 dgi 热区槽表与 Host 行号比对：命中就输出 `kHit | slot`（命中标记 + 槽位）；否则记为 miss。

### 2.4 稳定 LRU 分区

- 命中项保持原 LRU 顺序排在前部；
- 未命中项按 LRU 逆序排到尾部（尾部最先被逐出）；
- **stable**：相同输入永远产生相同输出，便于逐位验收。

### 2.5 victim 回填

按首次引用顺序，把 union 里的新行分配到尾部槽：写 `swap_host` / `swap_device` 搬移清单，置 `miss_mask`；最后压缩出注意力索引，空位置填 `-1`，并给出有效长度 counts。

## 3. A5 上的实现形态

- 内核是 **AscendC SIMT AIV**（`csrc/hisparse/hisparse_resolver.cpp`）；
- 每个连续请求组由一个 AIV 一次 launch 完成；
- UB（统一缓冲）预算由环境变量 `HISPARSE_RESOLVER_UB_BYTES=175616` 控制；
- 编译目标面向 dav-3510；A5 上没有 Triton resolver 回退路径。

## 4. 小结

- resolver = 翻译（逻辑→物理）+ 调度（搬移/淘汰计划）；
- 五步语义：fast path → hash union → hot 命中 → 稳定 LRU → victim 回填；
- A5 用 SIMT AIV 内核跑这套算法；
- 下一篇看配套的缓存算子家族。

**下一篇**：`04-cache-ops-kernels.ipynb`
"""),
]))

A5.append(("04-cache-ops-kernels.ipynb", [
md("""# HiSparse 入门 (4/8)：缓存算子家族 —— gather / invalidate / pack / scatter

> **一句话结论**：resolver 出计划之后，由一组缓存算子执行搬运、失效、组装与写回；A5 上缓存类算子用 Triton，resolver 用原生 AscendC，注意力复用既有 SFA。

## 1. 四个基础动作

| 动作 | 职责 | 通俗解释 |
| --- | --- | --- |
| gather | 把缺失行从 Host/Resident 字节拷贝到目标池 | 把仓库/书架上的书取到桌面 |
| invalidate | 写槽后清掉所有指向旧行号的索引（置 `-1`） | 书换了内容，旧书签全部作废 |
| pack | 把 KV 分量组装成标准 656B 行 | 把散页装进标准信封 |
| scatter | 把组装好的行写回池 | 信封放回对应格子 |

## 2. A5 上的实现分工

- **Resolver**：原生 AscendC SIMT AIV 内核（见上一篇）；
- **缓存/失效/打包/散布**：Triton 算子（`ops/triton/hisparse.py`）；
- **注意力**：复用原生 SFA，不另写注意力算术：
  - BF16 KV：`npu_sparse_flash_attention`；
  - C8 量化 KV：`npu_kv_quant_sparse_flash_attention_vllm`。

## 3. 图捕获安全约束（很重要）

这些算子会被录进 CUDA/Ascend 图，因此必须满足：

- 图捕获内**不做堆分配**；scratch / workspace 提前预分配并复用；
- 输出张量预先分配好；
- 捕获路径上**不做 host 同步**；
- 相同输入产出稳定布局，保证图可以重放。

## 4. 小结

- 四个动作：gather 取行、invalidate 作废旧索引、pack 装行、scatter 写回；
- A5：Triton 跑缓存算子，AscendC 跑 resolver，注意力走既有 SFA；
- 所有算子受图捕获安全约束；
- 下一篇拆 656B 行的字节布局。

**下一篇**：`05-c8-row-format-a5.ipynb`
"""),
]))

A5.append(("05-c8-row-format-a5.ipynb", [
md("""# HiSparse 入门 (5/8)：656B 行格式 —— A5 上的 fp8 C8 长什么样

> **一句话结论**：一行固定 656B，分三段：512B NoPE 量化值 + 128B BF16 RoPE + 16B FP32 scales；A5 用硬件原生 E4M3FN fp8 存 NoPE。

## 1. 三段拆解

| 段 | 字节 | 类型 | 装什么 |
| --- | --- | --- | --- |
| NoPE（无位置信息内容） | 512B | E4M3FN fp8 | 量化后的 KV 主值 |
| RoPE（位置信息） | 128B | BF16 | 64 维 × 2B，位置编码分量 |
| scales | 16B | FP32 ×4 | 4 个块的量化缩放系数 |

合计：512 + 128 + 16 = **656B**。

## 2. 字节校验（纯计算）
"""),
code("""nope_bytes = 512      # E4M3FN fp8, 1 byte/element
rope_bytes = 128     # BF16, 2 bytes/element
scale_bytes = 4 * 4  # 4 x FP32

row_bytes = nope_bytes + rope_bytes + scale_bytes
print("row bytes:", row_bytes)
assert row_bytes == 656
"""),
md("""## 3. A5 的硬件前提

- A5（Ascend 950）支持**原生 E4M3FN fp8**，块量化可由既有链/硬件完成；
- Main 与 Indexer 的 dtype 可以独立选择；
- 注意力在池上按物理行号原地直读这 656B，不做二次重组。

## 4. 一个比喻

一行就是一个标准信封：

- 主内容（NoPE）占最大格子；
- 位置条（RoPE）单独一格，避免被量化破坏；
- 每块的换算系数（scales）贴在信封背面。

## 5. 小结

- 656B = 512B fp8 NoPE + 128B BF16 RoPE + 16B FP32 scales；
- A5 依赖硬件原生 fp8；这一点到 A3 会成为关键差异；
- 下一篇把这些部件串成一步完整的前向流程。

**下一篇**：`06-forward-flow-a5.ipynb`
"""),
]))

A5.append(("06-forward-flow-a5.ipynb", [
md("""# HiSparse 入门 (6/8)：一步前向怎么走 —— decode / prefill / 混合批次

> **一句话结论**：decode 走"解析→补缺→直读"，prefill 走"暂存计划→搬入行→组装→注意力"，混合批次两条路径并行、输出拼接。

## 1. Decode 路径

1. `prepare_for_batch`：按批次准备 resolver 输入；
2. swap-in 解析：resolver 把逻辑 token 翻译成物理行，产出搬移与淘汰计划；
3. 缺失行 gather 到 Resident / Hot；
4. 生成物理索引，SFA 在热区原地读取 656B 行做注意力。

## 2. Prefill 路径

1. 生成 staging 暂存计划；
2. `stage_prefill_rows`：把 prefill 行搬入 Resident；
3. pack 组装标准 656B 行；
4. 注意力读取。C8 全驻留时可走分页捷径。

## 3. 混合批次

同一批里 decode 请求与 prefill 请求并存：

- decode 行走解析路径；
- prefill 行走暂存路径；
- 两路输出在注意力层拼接成批次结果。

## 4. 写入与 Host 镜像

- `write_target()`：把新 KV 行 scatter 写回 Resident / Hot；
- Connector 把 Resident 行镜像回 Host 池，保证 Host 留底完整；
- 捕获期写回延迟发布；draft 与最终请求重叠时做围栏判定；
- 分组回收时按组失效旧槽。

## 5. 小结

- decode：解析 → 补缺 → 直读；
- prefill：暂存 → 搬入 → 组装 → 注意力；
- 写入后镜像回 Host，捕获期遵守围栏与延迟发布；
- 下一篇看后端在 vllm-ascend 里的接线与模块分工。

**下一篇**：`07-backend-wiring.ipynb`
"""),
]))

A5.append(("07-backend-wiring.ipynb", [
md("""# HiSparse 入门 (7/8)：后端接线 —— Core 与 Ascend 各管什么

> **一句话结论**：Core 持有生命周期与池分配口径，Ascend 后端包 `vllm_ascend/hisparse/` 供应算子、视图与传输适配；平台层通过注册把后端类挂进框架。

## 1. Core 与 Ascend 分工

| 职责 | 归属 |
| --- | --- |
| HMA 分配、Coordinator 准入/淘汰、请求生命周期 | vLLM Core |
| 分组预取（含 MTP 校验）、Host 镜像完成、draft finalizer | vLLM Core |
| 内存预算口径、`write_target()` 缓存与分组失效 ABI | vLLM Core |
| 驻留解析、Host 传输映射、共享 slab 视图适配 | Ascend 后端 |
| SFA 接入、流/事件围栏、算子 workspace 预分配 | Ascend 后端 |

## 2. 后端包模块一览

| 文件 | 职责 |
| --- | --- |
| `backend.py` | `AscendHiSparseBackend`：可用性检查、Host 注册、cache_ops 构造、D2H 分批（2047 上限）、profiling 前 workspace 预分配 |
| `binding.py` | slab 规格对齐、Indexer 视图、FP8 K·scale 别名 |
| `cache_ops.py` | `HiSparseNPUOps` + ResolverWorkspace：驻留解析、gather、失效、Resident 拷贝、行打包、散布 |
| `c8.py` | 打包、视图、物理索引映射、紧凑化 |
| `host_pool.py` | 锁页私有 Host + `aclrtHostRegister` |
| `host_mirror.py` | Target 写回与 draft 重叠围栏 |
| `recycled_invalidate.py` | 分组回收失效 |

## 3. 框架接线点

- `NPUPlatform.get_hisparse_backend_cls()` 返回 `vllm_ascend.hisparse.backend.AscendHiSparseBackend`；
- `worker.prepare_hisparse_workspaces` 在 profiling 前预分配 workspace；
- `patch/platform/patch_kv_cache_coordinator.py` 让 HiSparse 组走 Core 协调器；
- `attention/indexer.py` 提供 `AscendHiSparseIndexerBackend / MetadataBuilder`；
- `patch/worker/patch_attn_utils.py` 按 hisparse_config 选择 Indexer 后端。

## 4. 小结

- Core 管"什么时候留、什么时候淘汰、生命周期"；Ascend 管"怎么搬、怎么装、怎么接算子"；
- 后端包 7 个文件各管一块；平台注册 + worker 预分配 + coordinator 补丁完成接线；
- 最后一篇看 A5 的支持范围与当前状态。

**下一篇**：`08-scope-and-status-a5.ipynb`
"""),
]))

A5.append(("08-scope-and-status-a5.ipynb", [
md("""# HiSparse 入门 (8/8)：A5 支持范围、排除项与当前状态

> **一句话结论**：这是面向 Ascend 950 的后端基础合入；已划定支持与排除边界，整机正确性、精度与最大并发资格仍待完成。

## 1. 已支持范围

- Model Runner V2；
- Ascend 950，SIMT 能力，CANN 9.2；
- BF16 KV，或原生 E4M3FN C8 + FP32 scales；
- Main / Indexer dtype 独立可选；
- 每 rank 私有 Host 池；
- TopK 为 128 的整数倍。

## 2. 明确排除

- INT4 KV；
- TurboQuant；
- INT8 格式 C8；
- Context Parallel（CP）；
- 外部 P/D 传输；
- A2 / A3 设备未认证。

## 3. 当前状态

- 定位是**上游后端基础**：把可插拔 ABI、池布局、算子接口在 vllm-ascend 内落地；
- 整机正确性、精度、最大并发资格测试待完成；
- 性能数字属于软件观测口径，不等同物理 DMA 实测。

## 4. 与 A3 的关系预告

A3（910_93）不在上述认证范围内。由于硬件能力缺口不同，A3 落地时对 resolver、失效、C8 dtype 与 pack 链路做了逐项替换——这正是 `../a3/` 目录笔记要讲的内容。

## 5. 小结

- 记住支持矩阵与排除清单，避免在不支持的格式/并行方式上排查；
- 基础合入 ≠ 整机资格完成；
- 至此 A5 八篇入门结束；可对照 `../a3/01-why-a3-need-own-port.ipynb` 看移植差异。
"""),
]))

# =====================================================================
# A3 —— hisparse-a3，Ascend 910_93（移植增量）
# =====================================================================

A3 = []

A3.append(("01-why-a3-need-own-port.ipynb", [
md("""# HiSparse on A3 入门 (1/8)：为什么 A3 需要自己的一套移植

> **一句话结论**：hisparse-a3 不是另一套框架，而是 PR #20926 同一数据面在 Ascend 910_93 上的移植增量；框架契约不变，只把 A5 上由硬件/编译链免费提供的能力逐项换成软件实现。

## 1. 基线与环境

- 基线：`pr20926 @ 5a54eb466`（2026-11-09），仓内只包含在此之上的增量；
- 硬件：Ascend 910_93（Atlas A3），16 die / TP16 / DP1，无 MTP；
- 软件：CANN 9.2.1，Python 3.12.16；
- 模型：GLM-5.2-w4a8c8（同族验证）/ GLM-5.3-Flash（生产）。

## 2. A3 撞上的四个能力缺口

| 缺口 | A5 上的情况 | A3 上的后果 |
| --- | --- | --- |
| SIMT 编译链 | 原生 AscendC SIMT AIV 可编译 | `dav-c220` 无 `asc_simt`，resolver 编译直接失败 |
| fp8 block quant | 硬件原生 E4M3FN | `dst_type=fp8` 报 161002，NoPE 只能改 int8 |
| 原生 int8 KV cache | 硬件可直接落量化行 | 无原生 int8 KV cache，每行都要软件构建 |
| Triton 发射开销 | A5 上可接受 | 2688 program 发射约 25.7–30 ms/步，压关键路径 |

## 3. 一句话理解移植成本

> **A5 的 C8 是硬件免费，A3 的 C8 是软件税。**

A5 上由编译链、硬件算子、发射效率承担的部分，到 A3 上要靠 AICPU 任务、AIV 融合算子、int8 桥接与分块 gather 补回来。

## 4. 不变的部分

三级池职责、656B 行寻址、resolver 算法语义、Core 生命周期、图捕获安全约束，两端完全一致。后面七篇逐个讲四项替换（M1–M4）、前向流程、性能账本与踩坑。

## 5. 小结

- a3 = 同一数据面 + 四项软件替换；
- 先记住四个缺口：SIMT、fp8、原生 int8 KV、Triton 发射；
- 下一篇讲 M1：int8 C8 直读桥。

**下一篇**：`02-m1-int8-c8-bridge.ipynb`
"""),
]))

A3.append(("02-m1-int8-c8-bridge.ipynb", [
md("""# A3 移植 (2/8)：M1 —— int8 C8 直读桥

> **一句话结论**：A3 做不了 fp8 块量化，NoPE 改用 int8；注意力侧按 dtype 与算子可得性分流，fp8 走定制 `_vllm` 算子，int8 回落 vendor 原生量化稀疏注意力。

## 1. 问题是什么

A5 的 C8 行 NoPE 是原生 E4M3FN fp8。A3 上请求 `dst_type=fp8` 直接报 161002（不支持 fp8 block quant），所以 NoPE 分量改成 **int8**，scales 仍用 FP32。

## 2. 改了哪三处

1. **dtype 门放宽**：C8 cache dtype 从只接受 `E4M3FN` 放宽到 `(E4M3FN, int8)`；
2. **注意力 dispatch**：
   - 若 `_vllm` 定制算子存在且 dtype 是 fp8，走 `npu_kv_quant_sparse_flash_attention_vllm`；
   - 否则回落 vendor `torch_npu.npu_kv_quant_sparse_flash_attention`；
3. **c8 视图/打包**：`pack_c8_components` / `c8_storage_view` 同步接受 int8。

## 3. int8 回落路径的契约

vendor 原生算子按 **TND 布局 + 绝对物理行号 + `key_quant_mode=2`** 工作；行号契约与上游一致，`-1` 仍表示空行掩码。

## 4. 探针验证结论

单卡探针确认：int8 656B 行 + 空洞 slab + `-1` 掩码的 SFA 路径 PASS，max_err 0.012、cos 1.0，图捕获 replay 误差 0.0000。

## 5. 收益（4k bs16）

- 步时：170.8 → 128.1 ms/步；
- TTFT：17.3 → 9.8 s。

## 6. 小结

- M1 = dtype 门放宽 + 注意力分流 + 视图适配；
- int8 走 vendor 原生算子，行号/掩码契约不变；
- 下一篇讲 M2：resolver 从 SIMT 搬到 AICPU。

**下一篇**：`03-m2-aicpu-resolver.ipynb`
"""),
]))

A3.append(("03-m2-aicpu-resolver.ipynb", [
md("""# A3 移植 (3/8)：M2 —— AICPU 解算器

> **一句话结论**：A3 编译链不支持 SIMT，上游 resolver 算法被逐字转写成 AICPU C++；分两阶段跑，scratch 按 device 缓存复用，保证图捕获内无堆分配。

## 1. 问题是什么

A5 的 resolver 是 AscendC SIMT AIV 内核。A3 的编译目标 `dav-c220` 在 CANN 9.x 没有 `asc_simt` 支持，SIMT 编译直接失败，offload 根本起不来。

## 2. 转写方式

上游算法**逐字转写**为 AICPU C++（kernel 558 行），五步语义不变：fast path → hash union → hot 命中 → 稳定 LRU → victim 回填。

### 两阶段并行

- **Phase 1**：`CpuKernelUtils::ParallelFor` 按行分片处理 fast-path——只读共享表、写互斥切片，天然安全；这是 AICPU 6 核唯一的用武之地；
- **Phase 2**：串行处理共享 hash scratch（开放寻址 union）。

### 捕获安全

- scratch 按 device 缓存复用，不在图捕获内申请堆内存；
- 输出预分配；几何门控：topk=2048、block=128、hot>0 才走该路径。

## 3. 接口规模

20 个输入张量原地改写，25 个几何属性；输出物理索引、搬移清单、stats 与有效长度。

## 4. 成本（4k bs16）

- 图内 264 µs/次，每步 20.8 次；
- 合计 **+5.24 ms/步**，是 A3 账本里下一最大单项；
- 服务零错误，语义与上游逐位对齐；17 项 op 测试在 A3 通过。

## 5. 小结

- M2 = SIMT 不可用 → AICPU 逐字转写；
- Phase 1 行级并行，Phase 2 串行共享 hash；scratch 缓存复用；
- 代价 +5.24 ms/步；后续可考虑 AI-Core 化；
- 下一篇讲 M3：写槽失效从 Triton 搬到 AICPU。

**下一篇**：`04-m3-aicpu-invalidate.ipynb`
"""),
]))

A3.append(("04-m3-aicpu-invalidate.ipynb", [
md("""# A3 移植 (4/8)：M3 —— AICPU 写槽失效

> **一句话结论**：Triton 全扫描在 A3 上发射开销约 25.7 ms/步且与计算零重叠；改成一个 AICPU task 扫描全部单元后，压到 0.55 ms/步。

## 1. 问题是什么

KV 写入热槽后，所有 indexer 组里指向该旧行号的 dgi 项都必须置 `-1`，否则注意力会读到已变内容。

上游是 Triton 全扫描：例如 tokens=8、groups=21 时启动 **2688 个 program**。A3 上每个 program 发射约 11 µs，合计 **~30 ms/call**；stream-timeline 实测它独占一条流、与计算流**零重叠**，25.7 ms/步直接压在关键路径上。

## 2. AICPU 做法

一个 AICPU task 扫描全部 `tokens × groups` 单元（约 4 ns/cell，2688×4096 ≈ 11M cell，共 656 µs）：

- `group_table` 是 int64 **设备地址表**（每行一个 dgi 裸指针），AICPU 直接解引用，不需要 torch 层先 gather；
- `(token, group)` 单元的并发只发生在相同 `(state, value)` 上；多个 token 写同一 state 的同值 `-1` 是**良性竞争**（同值写）；
- **无 capture 门**：图里录的是 AICPU task，捕获期不回退 Triton。

## 3. 成本对比

| 路径 | 耗时（4k bs16） |
| --- | --- |
| Triton 全扫描 | 25.7 → 30 ms/步 |
| AICPU 单 task | **0.55 ms/步** |

## 4. 比喻

从"开 2688 个小工分头去找旧书签"变成"一个工人按清单批量扫一遍表格"，省去了大量分头出发的调度开销。

## 5. 小结

- M3 = Triton 发射税 → AICPU 单 task 扫描；
- 设备地址表裸指针解引用；同值写竞争良性；
- 25.7 → 0.55 ms/步；
- 下一篇讲 M4：AIV 融合 C8 pack。

**下一篇**：`05-m4-aiv-fused-pack.ipynb`
"""),
]))

A3.append(("05-m4-aiv-fused-pack.ipynb", [
md("""# A3 移植 (5/8)：M4 —— AIV 融合 C8 pack

> **一句话结论**：A3 没有原生 int8 KV cache，原来 6–7 个碎 kernel 组成装链约 +5.9 ms/步；新写一个 AIV 融合算子把整条链压成一次调用，388 → 21.4 µs（18×）。

## 1. 问题：软件 C8 税

每层 decode 都要把 bf16 KV 变成 656B 行。上游碎算子链拆开是：

| 步骤 | 算子 | ms/步 |
| --- | --- | --- |
| 切分 512/64 | Slice ×2 | 0.93 |
| rmsnorm | `npu_rms_norm` | 1.06 |
| RoPE | `npu_interleave_rope` | 0.98 |
| 4×128 块量化 | `npu_dynamic_block_quant` | 1.02 |
| 分量打包 | `pack_c8_components`（torch.cat） | 0.61 |
| 转置/视图辅助 | `TransposeBMM` | 0.99 |
| scatter 到池 | Triton `scatter_packed_kv` | 0.29 |
| **合计** | | **≈5.9** |

## 2. 融合算子 `HisparsePackC8`

一行一核，单算子完成：

1. rmsnorm：fp32 计算，输出舍入到 bf16；
2. 4 个 128 块 int8 量化：absmax/127 作为 fp32 scale；
3. RoPE lane 代数；
4. 656B 组装；
5. scatter 写回池。

输出 rope / latent / scales 三个预分配分量。

## 3. 语义先钉死，再融合

融合前用探针逐字段对齐 vendor 链：rmsnorm 舍入规则、scale codes 取整规则、RoPE 逐位对齐；6 组寻址对抗用例逐字节一致。

## 4. 门控与回退

- 环境变量 `HISPARSE_PACK_FUSED=1` 启用；
- 仅 rows ≤ 64 的 decode 路径；
- prefill、以及 A5 路径回退原碎链。

## 5. 收益

- 算子：388 → 21.4 µs/call（18×）；
- e2e 再省 2.5 ms/步。

## 6. 小结

- M4 = 碎链税 → AIV 一行一核融合；
- 语义先对齐再融合；门控仅限 decode 小批量；
- 下一篇看 A3 改动后的一步前向与时序约束。

**下一篇**：`06-a3-forward-flow.ipynb`
"""),
]))

A3.append(("06-a3-forward-flow.ipynb", [
md("""# A3 移植 (6/8)：改动后的一步前向与时序约束

> **一句话结论**：数据面骨架与 A5 相同，只是 resolver、失效、pack 换成 AICPU/AIV 实现；注意图捕获形状在 TP 对齐下被收敛，小批量会被 padding 到 16 重放 FULL 图。

## 1. 一步 decode 时序

1. 模型产出本步 KV / 查询；
2. **AICPU resolver**：翻译物理行 + 搬移/淘汰计划；
3. 缓存算子：gather 补缺、**AICPU invalidate** 作废旧索引；
4. **AIV 融合 pack**（decode 小批量）组装 656B 行；
5. SFA 在热区原地直读：int8 走 vendor 原生量化稀疏注意力；
6. 写回池并镜像 Host。

## 2. 图捕获形状与 padding 税

- 候选 capture sizes `[1, 2, 4, 8, 16]` 被 TP 对齐收敛为 **`[16]`**；
- 实际 bs1–15 会被 padding 到 16，重放 FULL 图；
- 慢点来自 15 个 dummy 请求稀释热区，**不是 eager 路径慢**；
- 因此主性能口径以 **bs16** 为准。

## 3. 算子捕获要求（A3 同样遵守）

- resolver：无堆分配，scratch 缓存复用；
- invalidate：捕获期不回退 Triton，录 AICPU task；
- pack：输出预分配，捕获路径无 host 同步；
- 相同输入产出稳定布局，保证图可重放。

## 4. 小结

- A3 前向骨架 = A5 骨架换引擎；
- 小批量 padding 到 16 重放 FULL 图，解读性能时注意口径；
- 下一篇读性能归因账本。

**下一篇**：`07-performance-attribution.ipynb`
"""),
]))

A3.append(("07-performance-attribution.ipynb", [
md("""# A3 移植 (7/8)：性能账本怎么读 —— ON/OFF 归因

> **一句话结论**：4k bs16 稳态 ON 55.3 vs OFF 38.9 ms/步（+16.4 ms，+42%）；账本按项归因，先分清哪项是结构性成本、哪项是测量噪声。

## 1. 对比矩阵读法

- **OFF**：关掉 HiSparse 的基线臂；
- **bf16 桥**：走 bf16 直读；
- **int8 直读**：M1 桥接后的 int8 路径；
- 新机 ON/OFF：同日同机 A/B，避免混型号混批次比较。

## 2. 稳态归因账本（4k bs16）

| 项 | ms/步 | 说明 |
| --- | --- | --- |
| Resolver AICPU | +5.24 | 20.8 × 264 µs，下一最大单项 |
| int8 KV 行构建（M4 后） | ≈+3.4 | 融合后残留；prefill 未融合 |
| 注意力净差 | +2.44 | int8 SFA 122 µs/层 vs bf16 91 µs/层（1.34×） |
| 量化 Q 路径 | +2.10 | QuantBatchMatmulV3 254 → 334 次/步 |
| reduce_scatter | +1.31 | 3.22 vs 1.91 |
| gather_rows | +0.66 | 77.8 × 9 µs |
| invalidate | +0.55 | 0.8 × 656 µs |
| 其它 | +0.7 | 余量项 |

## 3. 口径注意

- 主口径是 **bs16**；bs1–15 受图 padding 税影响；
- 16k/32k × bs16 的 churn 格两臂轮间散布 **±10 ms**，属噪声；融合收益需 fresh-server + 受控 churn 协议才能定论；
- 软件观测口径不等同物理 DMA 实测；
- pack scale 用 `Muls` 倒数，与 vendor `amax/127` 有 1 ULP 差；codes ≤7/8192 差 ±1，属量化噪声级。

## 4. 小结

- 最大结构性成本：resolver AICPU、int8 行构建、int8 SFA 慢 1.34×、量化 Q；
- 读账本先排除 padding 税与 churn 噪声；
- 最后一篇看真实踩坑清单与遗留方向。

**下一篇**：`08-pitfalls-and-next.ipynb`
"""),
]))

A3.append(("08-pitfalls-and-next.ipynb", [
md("""# A3 移植 (8/8)：踩坑清单、遗留成本与下一步

> **一句话结论**：这批工作踩过的坑都按"症状→根因→修法"记录；剩余成本集中在 resolver AI-Core 化、int8 SFA 差距、量化 Q 路径与 prefill pack 融合。

## 1. 真实踩坑（入门版）

| 坑 | 根因 | 修法 |
| --- | --- | --- |
| GM 越界写（507035） | kernel 入口参数顺序与 op def 错位，静默换位指针 | 以 `*_def.cpp` 输入输出顺序逐位核对，顶部注释标注顺序 |
| 标量读到旧值/写错位置 | 标量读写底层用 `absAddr`，不是 `bufferAddr` | 标量用绝对下标 `GetValue/SetValue`，或走 TQue |
| sum/amax 脏读 | 向量流水线(V)与标量单元(S)不同步 | 取事件 id 做 `V_S`/`S_V` 栅栏；id 由 TPipe 取，不自选 |
| mapping 读到陈旧行号 | GM 标量直读在 A3 工具链不稳定 | 标量 `DataCopyEx` 进 UB 快照，寄存器数组内循环不再读标量 |
| TQue 对齐问题 | buffer 未 32B 对齐 | 按 32B 对齐分配 |
| 算子库缺符号 | OPP 装器共享库是累积产物，未全家族一次构建 | 算子私有 OPP 独立构建，全家族一次打包 |

## 2. 遗留结构性成本

- Resolver AICPU +5.24 ms/步：后续可 AI-Core 化；
- int8 SFA 比 bf16 SFA 慢 1.34×；
- 量化 Q 路径 +2.10 ms/步；
- prefill 的 int8 行构建尚未融合（M4 仅 decode rows≤64）。

## 3. 工程注意

- 实验机上 `csrc/build_aclnn.sh` 曾被替换为 stub，真实脚本保留为 `.real`；
- A3 未实现 `aclrtHostMemMapCapabilities`，用 `HISPARSE_SKIP_HOSTMAP_CAP=1` 跳过；
- gather 按 `HISPARSE_GATHER_CHUNK_BYTES=256 MiB` 分块，避免 4.6–37 GiB 临时张量 OOM。

## 4. 小结

- 内核层的坑多在参数顺序、标量寻址、V/S 栅栏与对齐；
- 剩余优化方向：resolver 下 AI-Core、int8 注意力降差、量化 Q 减负、prefill pack 融合；
- A3 八篇入门结束。对照 `../a5/` 可回看上游基础实现。
"""),
]))

for fname, cells in A5:
    write("a5", fname, cells)
for fname, cells in A3:
    write("a3", fname, cells)

print("DONE:", len(A5), "a5 notebooks,", len(A3), "a3 notebooks")
