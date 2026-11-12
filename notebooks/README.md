# HiSparse 落地入门笔记

面向零基础读者，分两条路径讲清 HiSparse 在两类昇腾设备上的落地。核心部件保留英文，讲解用中文。

## A5 · Ascend 950（PR #20926 上游基础）

目录：[`a5/`](./a5)

| # | 笔记 | 讲什么 |
| --- | --- | --- |
| 1 | [what-is-hisparse](./a5/01-what-is-hisparse.ipynb) | 长上下文 decode 的带宽墙与 HiSparse 思路 |
| 2 | [three-tier-cache](./a5/02-three-tier-cache.ipynb) | Host / Resident / Hot 三级池与 656B 行契约 |
| 3 | [resolver-on-a5](./a5/03-resolver-on-a5.ipynb) | 解算器五步语义与 SIMT AIV 实现 |
| 4 | [cache-ops-kernels](./a5/04-cache-ops-kernels.ipynb) | gather / invalidate / pack / scatter 算子分工 |
| 5 | [c8-row-format-a5](./a5/05-c8-row-format-a5.ipynb) | 656B 行拆解：fp8 NoPE + BF16 RoPE + FP32 scales |
| 6 | [forward-flow-a5](./a5/06-forward-flow-a5.ipynb) | decode / prefill / 混合批次前向流程 |
| 7 | [backend-wiring](./a5/07-backend-wiring.ipynb) | Core 与 Ascend 分工、后端包模块与接线点 |
| 8 | [scope-and-status-a5](./a5/08-scope-and-status-a5.ipynb) | 支持范围、排除项与当前状态 |

## A3 · Ascend 910_93（hisparse-a3 移植增量）

目录：[`a3/`](./a3)

| # | 笔记 | 讲什么 |
| --- | --- | --- |
| 1 | [why-a3-need-own-port](./a3/01-why-a3-need-own-port.ipynb) | 基线、环境与四个硬件能力缺口 |
| 2 | [m1-int8-c8-bridge](./a3/02-m1-int8-c8-bridge.ipynb) | M1：int8 C8 直读桥与注意力分流 |
| 3 | [m2-aicpu-resolver](./a3/03-m2-aicpu-resolver.ipynb) | M2：SIMT 不可用 → AICPU 逐字转写 |
| 4 | [m3-aicpu-invalidate](./a3/04-m3-aicpu-invalidate.ipynb) | M3：Triton 发射税 → AICPU 单 task 扫描 |
| 5 | [m4-aiv-fused-pack](./a3/05-m4-aiv-fused-pack.ipynb) | M4：碎算子链 → AIV 融合 pack |
| 6 | [a3-forward-flow](./a3/06-a3-forward-flow.ipynb) | 改动后前向时序、图捕获形状与 padding 税 |
| 7 | [performance-attribution](./a3/07-performance-attribution.ipynb) | ON/OFF 归因账本与口径注意 |
| 8 | [pitfalls-and-next](./a3/08-pitfalls-and-next.ipynb) | 踩坑清单、遗留成本与下一步方向 |

## 事实来源

- A5：vllm-project/vllm-ascend PR #20926（头部分支 `feat/hisparse-foundation-20261104`）
- A3：gitcode.com/SamWang921/hisparse-a3（基线 `pr20926@5a54eb466`）
- 仓根分析文档：`pr20926_hisparse_architecture.md`、`hisparse-a3-vs-pr20926.md`
