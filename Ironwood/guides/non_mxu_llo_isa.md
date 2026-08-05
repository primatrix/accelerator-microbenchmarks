# TPU v7x 非 MXU 计算 LLO 指令参考（实测草案）

本文面向 Pallas kernel 实现、硬件约束判断和性能分析，记录 JAX/Pallas 源码在 TPU v7x 上编译后，JF `final_bundles` 中实际出现的非 MXU 计算指令。

本文同时使用三种不同层次的名字，不能混为一谈：

1. **JAX/Pallas primitive**：例如 `lax.div`。
2. **LLO in-memory enum**：例如参考资料中的 `kVectorReciprocalF32` 及其内部编号。它是编译器调度前的 LLO IR opcode。
3. **final-bundle mnemonic**：例如 `vrcp.f32`、`vpop.eup`。这是本轮 dump 直接可见、已经过 late decomposition、寄存器分配和 bundle scheduling 的指令形式。

final bundle 会打印第 3 类信息，包括 mnemonic、类型后缀、源/目标寄存器、立即数、predicate/mask、bundle 序号和同周期的并发发射；它**不会打印第 2 类 enum 的数字值**。因此本文把“enum 对照”和“bundle 实测”分列，不从文本中的 `0x3` 推断 opcode 编号——`0x3` 是 bundle 序号。

## 1. 当前结论

| 项目 | 结果 |
|---|---:|
| Probe case | 53 |
| 取得命名 kernel final bundle | 53 / 53 |
| final bundle 中不同的全部 mnemonic | 126 |
| 按指令族排除 load/store、DMA、同步和其他 control 后的非 MXU 计算 mnemonic | 112 |
| MXU mnemonic | 0 |
| 有数值 reference check 的 case | 49 / 49 通过 |
| 暂不做数值 reference check 的 case | 4（pack、unpack、roll、stochastic round） |

测试环境：TPU v7x-8（`2x2x1`，8 个可见设备），Python 3.12.12，JAX/JAXLIB 0.10.2，libtpu 0.0.42.1。主 sweep 的 Falcon experiment 是 `exp-ecm4j8q5nj`；修正 iota 生成方式后的单项复测是 `exp-nidqyvf1nb`。

当前状态应理解为：**已完整保存并解释这 53 个 case 实际发出的 112 种计算 mnemonic，但尚不能声称已经触发参考 enum 中每一个非 MXU 计算成员。** 未触发项和下一轮 probe 矩阵见第 11 节。

## 2. 证据与复现

### 2.1 源码

- Probe catalog 与 kernel：[`benchmark_non_mxu_pallas.py`](../src/benchmark_non_mxu_pallas.py)
- 每 case 独立进程、dump 选择与 final bundle 收集：[`benchmark_non_mxu_pallas_sweep.py`](../src/benchmark_non_mxu_pallas_sweep.py)
- final bundle parser 与 primitive 映射：[`analyze_non_mxu_final_bundles.py`](../src/analyze_non_mxu_final_bundles.py)
- Falcon v7x 配置：[`non_mxu_llo_sweep_v7x.yaml`](../falcon/non_mxu_llo_sweep_v7x.yaml)

每个 case 在一个新的 Python/libtpu 进程中运行，使用：

```text
--xla_jf_dump_to=<case-specific-directory>
--xla_jf_dump_llo_text=true
--xla_jf_emit_annotations=true
```

选择 final bundle 时，不按“目录里最大的任意文件”猜测，而是要求文件中包含该 case 的命名 entry marker：

```text
entry bundle: %non_mxu_<case_id>
```

### 2.2 本地 review 目录

全部 53 个 bundle 位于：

```text
artifacts/non_mxu_llo/exp-ecm4j8q5nj/cases/<case_id>/compiler/llo/final_bundle.llo
```

辅助产物：

| 文件 | 用途 |
|---|---|
| `analysis_refined/report.md` | 53 case 的 primitive → emitted mnemonic 完整表 |
| `analysis_refined/opcode_counts.csv` | 126 个全部 mnemonic 的出现次数和分类 |
| `analysis_refined/primitive_to_llo.csv` | 可机器处理的 primitive/case/mnemonic/count/归因强度 |
| `analysis_refined/summary.json` | case 级 bundle、指令计数和族统计 |
| `case_observations.json` | 原 sweep 的运行与正确性观测；保留首次非法 f32 iota 的诊断 |
| `cases/iota_f32/metrics.rerun.json` | 合法 integer iota 后 convert-to-f32 的复测结果 |
| `final_bundles_complete.tar.gz` | 53 个 final bundle 的完整本地归档（含 iota 复测） |

Falcon 内置的旧 final-bundle 分析插件能找到 52 个 LLO 文件，但对本版 JF 文本语法得到 0 个 opcode；这不是 dump 缺失。仓库 parser 直接解析实际 `{ inst ;; inst }` 语法，并保留连字符 mnemonic（例如 `vclamps-f32`）。原始 bundle 始终是最高级证据。

### 2.3 如何读一行 bundle

```text
0x3 : { %7 = vsyncpa ... ;; %v14_v2 = vadd.f32 %v13_v1, %v12_v0 ;; %s59_s13 = smov ... }
```

- `0x3`：调度后的 bundle 序号，不是 opcode enum value。
- `{ ... ;; ... }`：花括号中的多条指令在同一 VLIW bundle 中发射；`;;` 分隔不同 slot。
- `%v14_v2`：SSA value `%v14` 被分配到物理向量寄存器 `v2`。
- `%s...`、`%p...`、`%vm...`：scalar、predicate、vector-mask value/register。
- `.f32`、`.bf16`、`.s32`、`.u32`、`.b16/.b32`：数据解释或操作宽度。
- `/* ... */`：由 `--xla_jf_emit_annotations=true` 产生的 shape、allocation、bounds check、HLO 来源等注释。

## 3. v7x 执行模型与约束

下表中“实测”来自本轮 bundle；“参考”来自 LloOpcode enum 及其相邻 ISA 页面。参考资料基于 libtpu 0.0.40，而本轮实测使用 0.0.42.1，因此任何跨版本的**数字编号**都要再次核验。

| 项目 | v7x 约束/含义 | 证据 |
|---|---|---|
| 向量几何 | 1 个 32-bit vreg 覆盖 `8 sublanes × 128 lanes = 1024` 元素；bf16 等子 32-bit 类型在寄存器中打包 | 参考；本轮 `(8,128)` 基线实测通过 |
| 向量寄存器 | 架构命名空间 `v0..v1023`；单 slot 的直接编码窗口为 64 个 vreg | 参考 |
| Scalar 寄存器 | 32 个 SREG，5-bit selector | 参考 |
| Scalar lanes | 2 个 scalar issue lanes | 参考 |
| VALU slots | 每 bundle 最多 4 个 VALU slots | 参考；bundle 中可见并发 vector op |
| XLU | 2 个 XLU，mnemonic 以 `.xlu0/.xlu1` 指明资源 | 参考与实测 |
| Predicate | 16 个 scalar predicate regs；一个 bundle 的 vector op 最多选择两个不同的活动 predicate pool entry | 参考 |
| Vector mask | mask 是独立 `%vm` 值；比较、mask logic 和 `vsel` 消费/产生它 | 实测 |
| EUP | transcendental 是 issue/push 与后续 `vpop.eup` 分离的 deferred-result pipeline | 实测 |
| XLU result | reduction、permute、transpose 同样由 issue 与 `vpop.*` 分离 | 实测 |
| Bundle 大小 | v7x TensorCore bundle 为 64 B | 参考 |
| Immediate | 6 个 20-bit immediate slots；部分常量还可由硬连线常量/Y-source selector 提供 | 参考 |

本轮已证明的 shape/size 边界只有：

- 绝大多数向量 case：`(8,128)`；f32/s32/u32 window 为 4096 B，bf16 window 为 2048 B。
- transpose：`(128,128)`，产生 16 段 transpose sequence。
- scalar-grid：逻辑 shape `(32,128)`，用于让 `program_id` 进入 SPU 数据通路。

这些结果证明上述形状在该软件栈上合法，**不能反推出任意 VMEM/HBM 基地址的最小对齐值**。地址对齐、stride 和非整 vreg tail 需要单独做负例/边界 sweep。

## 4. Enum 计算族范围

参考 enum 是 dense in-memory `LloOpcode`：461 个 live value，`0x000..0x1CC`。它不同于 1-based、带空洞的 `LloOpcodeProto` wire value。当前非 MXU 计算范围定义如下：

| 计算族 | in-memory enum 范围 | 当前处理 |
|---|---|---|
| SPU scalar compute/control | `0x085..0x089`，`0x16B..0x1AA` 中的 scalar 成员 | 包含 |
| VPU clamp/MAC | `0x048..0x05A` | 包含；只触发 clamp 子集 |
| convert/pack/unpack | `0x05B..0x076`，`0x107..0x11A`，`0x125..0x127` | 包含；已触发常用 f32/bf16/s32 与 stochastic/pack |
| XLU permute/rotate | `0x036..0x03B`，pattern `0x08B..0x08C` | 包含；已触发 roll |
| reduction/sequence | `0x0F5..0x106` | 包含；已触发 sum/min/max/index/iota 子集 |
| VPU add/sub | `0x11B..0x124` | 包含 |
| EUP transcendental | `0x128..0x14D`，result `0x14E` | 包含 |
| XLU result | `0x14F..0x151`，transpose `0x154..0x155` | 包含 |
| VPU multiply/logic/wide multiply | `0x156..0x166` | 包含 |
| predicate/compare | `0x0E1..0x0E8`，`0x167..0x16A` | 包含 |
| VPU unary/shift/minmax | `0x180..0x184`，`0x19A..0x1A2` | 包含 |
| vector-mask logic | `0x193..0x199` | 包含 |
| MXU latch/matprep/matmul/result | `0x08D..0x0AB` 中 MXU 成员，`0x152..0x153` | **排除** |
| load/store、DMA、sync、sequencer、pseudo/constants | 各自范围 | 只当编译器脚手架统计，不归因给 source primitive |
| BarnaCore/SparseCore block | `0x1AC..0x1CC` | 当前 TensorCore 范围排除 |

注意：同一个 enum opcode（如 `kVectorCompare`）可根据 datatype/order/condition 编码成多个 final mnemonic；反过来，伪指令也可能被 late decomposer 展开成多个 bundle mnemonic。因此 enum 数量和本轮 112 个 final mnemonic 数量不应直接做覆盖率除法。

## 5. 实测指令参考：SPU 与 scalar predicate

本节列出本轮出现的全部 scalar/predicate 计算 mnemonic。`scalar_arith` 和 `scalar_logic_compare` 是复合 case：其中 loop/bounds/control 与被测 scalar primitive 共用 SPU，因此这里只能建立 case-level correlation，不能逐条声称 1:1 lowering。

| final mnemonic | 语义；操作数 → 结果 | 类型/约束 | 观测来源 |
|---|---|---|---|
| `sadd.s32` / `ssub.s32` | scalar 加/减：`sx, sy/imm → sdst` | 32-bit two's-complement | scalar-grid |
| `smul.u32` | 低 32-bit scalar 乘法 | u32 | scalar-grid |
| `smulhi.u32` | u32×u32 的高 32-bit | u32 wide-product helper | scalar-grid |
| `sdivrem.u32` | 发起 unsigned divide/remainder，产生 deferred DRF 结果 | u32；不是普通单结果 ALU | `scalar_arith` |
| `spop.drf` | 从 divide/remainder result FIFO 取结果 | 必须依赖相应 push/token | `scalar_arith` |
| `sand.u32` / `sor.u32` / `sxor.u32` | scalar bitwise AND/OR/XOR | u32 bit pattern | scalar-grid |
| `sshll.u32` / `sshrl.u32` | scalar logical left/right shift | u32；shift amount 的越界行为需单测 | scalar-grid 与地址脚手架 |
| `smin.u32` | unsigned scalar minimum | u32 | `scalar_arith` |
| `scvt.s32.f32` | signed scalar integer 转 f32 | `s32 → f32` | scalar-grid |
| `smov` | scalar copy/materialized constant | 32-bit scalar | case 与通用脚手架 |
| `sphi` | scalar SSA phi/loop merge | 控制流伪/合流语义，不是数值 primitive | scalar-grid |
| `scmp.eq.s32.totalorder`, `scmp.ne.s32.totalorder`, `scmp.lt.s32.totalorder`, `scmp.ge.s32.totalorder`, `scmp.gt.s32.totalorder` | scalar signed compare：`sx, sy → p` | s32 total order | scalar-grid；部分也是 bounds check |
| `scmp.lt.u32.totalorder` | scalar unsigned less-than：`sx, sy → p` | u32 total order | bounds/control 与 scalar-grid |
| `pneg` | predicate NOT | `p → p` | scalar-grid |
| `pnand` | predicate NAND | `p,p → p`；AND 常可由 De Morgan 组合实现 | scalar-grid 与 bounds check |
| `por` | predicate OR | `p,p → p` | scalar-grid 与 bounds check |

资源判断：scalar 指令占 scalar slot，predicate 指令占 predicate/相关 scalar 控制资源；同一 bundle 中能否与 VPU、load/store、EUP/XLU 并发，以实际 `;;` 排布为准。高频 `smov/sshll/scmp/pnand/por` 多数来自 Pallas 调用、地址计算和 bounds check。它们仍包含在全局 112-mnemonic inventory 中，但已从向量 primitive 的 source-attribution 表中剔除。

## 6. 实测指令参考：VPU、compare 与 mask

### 6.1 算术、bitwise、shift 与 unary

除另行说明外，向量操作逐 element 作用于完整 vreg；`x/y` 可以是 vreg 或可编码的立即数/Y-source，结果为 vreg。

| final mnemonic | 语义；操作数 → 结果 | 数据类型/特别约束 |
|---|---|---|
| `vadd.f32`, `vadd.bf16`, `vadd.s32` | `x + y → vdst` | f32/bf16/s32 |
| `vsub.f32`, `vsub.bf16`, `vsub.s32` | `x - y → vdst` | f32/bf16/s32 |
| `vmul.f32`, `vmul.bf16`, `vmul.u32` | 逐 element 乘法；u32 取低 32-bit | f32/bf16/u32 |
| `vmul.u32.u64.low` | u32×u32 的低 32-bit，并形成 wide-product pair | u32；与 high 形式关联 |
| `vmul.u32.u64.high` | 从两操作数及已算 low 部分形成高 32-bit | u32；bundle 原文有显式 `low=` operand |
| `vmin.f32`, `vmax.f32`, `vmin.bf16`, `vmax.bf16`, `vmin.u32` | 逐 element minimum/maximum | 注意 s32 min/max 本轮被 compare+`vsel` 合成 |
| `vand.u32`, `vor.u32`, `vxor.u32` | 逐 bit AND/OR/XOR | u32 bit pattern；也用于 f32 sign/exponent manipulation |
| `vshll.u32`, `vshrl.u32`, `vshra.s32` | logical left、logical right、arithmetic right shift | u32/s32；大 shift amount 边界待测 |
| `vclz`, `vpcnt` | count-leading-zero、population-count | 输入按 32-bit word 解释 |
| `vceil.f32`, `vfloor.f32`, `vtrunc.f32` | ceil、floor、toward-zero truncate | f32 → integral-valued f32 |
| `vround.rtna.f32` | round-to-nearest，tie away from zero | f32 |
| `vround.rtne.f32` | round-to-nearest，tie to even | f32 |
| `vmov` | vreg copy 或 materialized vector constant | 32-bit lane payload |
| `vclamps-f32` | 对称 f32 clamp；实测 `x, 1.0` 对应 `[-1,1]` | f32；lower/upper 的编码形式需更多边界 case |
| `vweird.f32` | 产生 f32 classification mask；在 `lax.is_finite` 和 trig range reduction 中出现 | `%v → %vm`；具体 bit truth table 尚待 NaN/Inf/subnormal 单测，不把名字臆解成完整语义 |
| `vc.u32` | wide-add carry/进位 mask；实测由两段 u64 乘积相加触发 | `u32,u32 → vm`；对应 enum compare/add-carry 族 |
| `vlaneseq` | 生成 lane sequence | 结果必须为 integer/index vector；不能直接生成 f32 |

### 6.2 compare、select 与 vector mask

| final mnemonic | 语义；操作数 → 结果 | 数据类型/顺序 |
|---|---|---|
| `vcmp.eq.f32.partialorder`, `vcmp.ne.f32.partialorder`, `vcmp.lt.f32.partialorder`, `vcmp.le.f32.partialorder`, `vcmp.gt.f32.partialorder`, `vcmp.ge.f32.partialorder` | 逐 element f32 比较，产生 `%vm` | `partialorder` 明确提示 NaN 参与时不是整数 total order；NaN truth table 需专项验证 |
| `vcmp.eq.s32.totalorder`, `vcmp.ne.s32.totalorder`, `vcmp.lt.s32.totalorder`, `vcmp.le.s32.totalorder`, `vcmp.gt.s32.totalorder`, `vcmp.ge.s32.totalorder` | signed 32-bit 比较，产生 `%vm` | s32 total order |
| `vcmp.gt.u32.totalorder`, `vcmp.ge.u32.totalorder` | unsigned 32-bit 比较，产生 `%vm` | u32 total order；大量出现在 software integer div/rem |
| `vsel` | `vm ? on_true_vy : on_false_vx → vdst` | mask 与数据 shape 必须一致；true/false 可为 vreg/立即数 |
| `vmand`, `vmor`, `vmxor` | vector-mask AND/OR/XOR | `%vm,%vm → %vm` |
| `vmneg` | vector-mask NOT | `%vm → %vm` |
| `vmmov` | mask copy 或 mask constant | `%vm/imm → %vm` |
| `vcmask` | 用一个 packed immediate 创建二维矩形 mask | 本轮 `1043456 /* [0:3,0:127] */`；end bound 在 dump 注释中为 inclusive |

v7x predicate 资源限制与 `%vm` mask 寄存器是两层概念：bundle predicate 控制整条 vector op 是否执行，`%vm` 控制 vreg 内哪些 element 参与。一个 bundle 内过多不同 active predicate 可能无法编码，即使还有空 VALU slot。

## 7. 实测指令参考：convert、pack、EUP 与 XLU

### 7.1 Convert / pack / unpack

| final mnemonic | 语义；操作数 → 结果 | 形状/布局观测 |
|---|---|---|
| `vcvt.s32.f32` | `s32 → f32` vector convert | iota case 先生成 s32 sequence，再转 f32 |
| `vcvt.f32.s32` | `f32 → s32` vector convert | 与 `vtrunc.f32` 一起出现，表明 source conversion 不是单条 lowering |
| `vcvt.sr.f32.bf16` | 以显式 random-bits operand 做 stochastic f32→bf16 rounding | `random_bits, f32 → bf16-like payload` |
| `vpack.c.bf16` | convert/compact f32 lanes 为 bf16 packed representation | f32→bf16 与 bf16 reduction output 均触发 |
| `vpack.c.b16` | 将两个 b16 half 组合为 32-bit container | stochastic round output path |
| `vpack.i.bf16` | `pltpu.pack_elementwise` 的 interleaving pack | 两个 unpacked value → packed u32 container |
| `vunpack.c.l.bf16` | 从 compact bf16 取 low half并扩为 f32 | bf16→f32 与 bf16 reduction input path |
| `vunpack.i.l.bf16` | interleaved packed bf16 的 low element unpack | `pltpu.unpack_elementwise(index=0)` |

直接 `lax.bitcast_convert_type(f32,u32)` 没有留下计算 mnemonic：它是相同 32-bit payload 的类型重解释，被编译器优化为无指令。这是“optimized away”，不是硬件不支持。

### 7.2 EUP deferred-result pipeline

| issue mnemonic | 数学语义 | 结果收集 | 本轮 primitive |
|---|---|---|---|
| `vtanh.f32` | tanh(x) | `vpop.eup token → vreg` | `lax.tanh` |
| `vpow2.f32` | 2^x | `vpop.eup` | `lax.exp`、`lax.exp2`、logistic、pow |
| `vrcp.f32` | 1/x | `vpop.eup` | f32 div/rem、tan、logistic |
| `vlog2.f32` | log2(x) | `vpop.eup` | log/log1p、pow、erf_inv |
| `vrsqrt.f32` | 1/sqrt(x) | `vpop.eup` | sqrt/rsqrt、erf_inv |
| `vsinq.f32`, `vcosq.f32` | quadrant-reduced sine/cosine core | `vpop.eup` | sin/cos/tan 复合 range-reduction path |
| `vpop.eup` | 从 EUP result FIFO 取对应 issue 的结果 | `token → vreg` | 17 次 |

关键约束：不能把 issue 与 pop 当成一个不可拆分指令来做调度或性能计数。实测 `div_f32` 在 bundle `0x2` 发 `vrcp.f32`，到 `0x8` 才 `vpop.eup`，最后以 `vmul.f32` 完成 `x * reciprocal(y)`。参考 cost 页面给出的 EUP latency 目前是低置信度估计，本文不把它当作 v7x 已验证延迟。

### 7.3 Reduction / permute / transpose XLU

| final mnemonic | 语义；操作数 → 结果 | 资源/形状 |
|---|---|---|
| `vadd.xlane.f32.xlu0`, `vadd.xlane.f32.xlu1` | 发起跨 lane f32 sum reduction | 占指定 XLU，返回 token |
| `vmin.xlane.f32.xlu0`, `vmin.xlane.f32.xlu1`, `vmax.xlane.f32.xlu0`, `vmax.xlane.f32.xlu1` | 发起跨 lane f32 min/max | 占指定 XLU，返回 token |
| `vmin.xlane.bf16.xlu0`, `vmax.xlane.bf16.xlu0` | bf16 min/max reduction | 本轮输出前后可见 pack/unpack/mask helper |
| `vmin.index.xlane.f32.xlu0`, `vmax.index.xlane.f32.xlu0` | 同时归约 value 与 index（argmin/argmax） | tie-breaking 规则需重复值专项 case |
| `vpop.xlane.xlu0`, `vpop.xlane.xlu1` | 收集对应 XLU reduction result | token → vreg |
| `vrot.lane.b32.xlu0` | lane 维 32-bit rotate/permutation issue | `pltpu.roll` |
| `vpop.permute.xlu0` | 收集 permute/rotate result | token → vreg |
| `vxpose.xlu0.b32.start` | 开始 b32 transpose sequence | 实测 width=128，`[1/16]` |
| `vxpose.xlu0.b32.cont` | transpose 中间 chunk | 实测 `[2/16]..[15/16]`，共 14 条 |
| `vxpose.xlu0.b32.end` | 结束 transpose issue sequence | 实测 `[16/16]` |
| `vpop.trf.xlu0` | 从 transpose result FIFO 逐 chunk 取结果 | 实测 16 条，对应 128×128 f32 tile |

Probe 为保持输出 shape 与输入一致，把单行 reduction result broadcast 回 `(8,128)`；因此 bundle 展示的是“reduction + broadcasted materialization”的完整 kernel，不应把所有 helper 都算成纯 reduction latency。

## 8. 完整 final-bundle LLO 指令 pattern

本节给出当前 112 个实测计算 mnemonic 的完整语法模板。它描述的是 `final_bundles.txt` 中可直接看到的 scheduled form，而不是高层 enum constructor 的 C++ 函数签名。

Pattern 元变量：

| 元变量 | 含义 |
|---|---|
| `%v_dst`, `%v_x`, `%v_y` | vector SSA value；dump 通常继续显示物理分配，如 `%v14_v2` |
| `%s_dst`, `%s_x`, `%s_y` | scalar SSA value/寄存器 |
| `%p_dst`, `%p_guard`, `%pred` | scalar predicate；`%pred` 可写成 `%p_guard` 或 `!%p_guard` |
| `%vm_dst`, `%vm_mask` | vector mask |
| `%token` | EUP/XLU/DRF deferred-result dependency token，不是普通 vreg |
| `imm` | 可编码立即数或 materialized constant |
| `src` | 对该 operand 可合法编码的 vreg/sreg/立即数之一；具体集合受 slot 编码限制 |
| `[%pred,]` | 可选的 scalar instruction predicate；方括号在这里表示 pattern 可选项，不是 bundle 原文 |

### 8.1 SPU 与 scalar predicate pattern

| 指令大类 | 完整 LLO 指令 pattern | 含义 |
|---|---|---|
| `sadd` / `ssub` | `%s_dst = sadd.s32 [%pred,] src_y, src_x`<br>`%s_dst = ssub.s32 [%pred,] src_y, src_x` | 32-bit scalar 加/减；source 可为 sreg 或立即数。 |
| `smul` | `%s_dst = smul.u32 [%pred,] src_y, src_x`<br>`%s_hi = smulhi.u32 [%pred,] src_y, src_x` | unsigned scalar 乘法的低/高 32-bit 结果。 |
| `sdivrem` / `spop` | `%token = sdivrem.u32 src_dividend, src_divisor`<br>`%s_dst = spop.drf %token` | 发起 u32 除法/余数并从 DRF FIFO 取结果。 |
| `smin` | `%s_dst = smin.u32 [%pred,] src_y, src_x` | unsigned scalar minimum。 |
| `scalar bitwise` | `%s_dst = sand.u32 [%pred,] src_y, src_x`<br>`%s_dst = sor.u32 [%pred,] src_y, src_x`<br>`%s_dst = sxor.u32 [%pred,] src_y, src_x` | scalar AND/OR/XOR。 |
| `scalar shift` | `%s_dst = sshll.u32 [%pred,] src_x, src_shift`<br>`%s_dst = sshrl.u32 [%pred,] src_x, src_shift` | scalar logical left/right shift。 |
| `scvt` | `%s_dst = scvt.s32.f32 %s_x` | s32 → f32 scalar conversion。 |
| `smov` | `%s_dst = smov [#allocationN]` | scalar copy/materialized constant；本轮实测为 allocation form。 |
| `sphi` | `%s_dst = sphi %s_init, %s_backedge` | scalar SSA loop/control-flow merge。 |
| `scmp.eq` | `%p_dst = scmp.eq.s32.totalorder src_y, src_x` | signed scalar equal compare。 |
| `scmp.ne` | `%p_dst = scmp.ne.s32.totalorder src_y, src_x` | signed scalar not-equal compare。 |
| `scmp.lt` | `%p_dst = scmp.lt.s32.totalorder src_y, src_x`<br>`%p_dst = scmp.lt.u32.totalorder src_y, src_x` | signed/unsigned scalar less-than compare。 |
| `scmp.ge` / `scmp.gt` | `%p_dst = scmp.ge.s32.totalorder src_y, src_x`<br>`%p_dst = scmp.gt.s32.totalorder src_y, src_x` | signed scalar greater-or-equal/greater-than compare。 |
| `predicate NOT` | `%p_dst = pneg %p_x` | scalar predicate NOT。 |
| `predicate NAND` | `%p_dst = pnand %p_y, %p_x` | scalar predicate NAND；常用于合成 AND/bounds check。 |
| `predicate OR` | `%p_dst = por %p_y, %p_x` | scalar predicate OR。 |

### 8.2 VPU arithmetic、bitwise 与 unary pattern

| 指令大类 | 完整 LLO 指令 pattern | 含义 |
|---|---|---|
| `vadd` | `%v_dst = vadd.f32 src_y, src_x`<br>`%v_dst = vadd.bf16 src_y, src_x`<br>`%v_dst = vadd.s32 src_y, src_x` | 逐 element f32/bf16/s32 addition。 |
| `vsub` | `%v_dst = vsub.f32 src_y, src_x`<br>`%v_dst = vsub.bf16 src_y, src_x`<br>`%v_dst = vsub.s32 src_y, src_x` | 逐 element subtraction。 |
| `vmul` | `%v_dst = vmul.f32 src_y, src_x`<br>`%v_dst = vmul.bf16 src_y, src_x`<br>`%v_dst = vmul.u32 src_y, src_x` | 逐 element multiplication；u32 form 返回低 32 bits。 |
| `wide vmul low` | `%v_low = vmul.u32.u64.low %v_y, %v_x` | u32×u32 wide product 的低 32 bits。 |
| `wide vmul high` | `%v_high = vmul.u32.u64.high /*lhs_vy=*/%v_y, /*rhs_vx=*/%v_x, /*low=*/%v_low` | wide product 的高 32 bits；显式依赖配对的 low value。 |
| `vmin` | `%v_dst = vmin.f32 src_y, src_x`<br>`%v_dst = vmin.bf16 src_y, src_x`<br>`%v_dst = vmin.u32 src_y, src_x` | 逐 element minimum。 |
| `vmax` | `%v_dst = vmax.f32 src_y, src_x`<br>`%v_dst = vmax.bf16 src_y, src_x` | 逐 element maximum。 |
| `vector bitwise` | `%v_dst = vand.u32 src_y, src_x`<br>`%v_dst = vor.u32 src_y, src_x`<br>`%v_dst = vxor.u32 src_y, src_x` | 逐 bit vector AND/OR/XOR。 |
| `vector shift` | `%v_dst = vshll.u32 %v_x, src_shift`<br>`%v_dst = vshrl.u32 %v_x, src_shift`<br>`%v_dst = vshra.s32 %v_x, src_shift` | logical-left、logical-right、arithmetic-right shift。 |
| `bit count` | `%v_dst = vclz %v_x`<br>`%v_dst = vpcnt %v_x` | 每个 32-bit element 的 leading-zero/population count。 |
| `rounding` | `%v_dst = vceil.f32 %v_x`<br>`%v_dst = vfloor.f32 %v_x`<br>`%v_dst = vtrunc.f32 %v_x`<br>`%v_dst = vround.rtna.f32 %v_x`<br>`%v_dst = vround.rtne.f32 %v_x` | ceil/floor/truncate/nearest-away/nearest-even。 |
| `vmov` | `%v_dst = vmov src` | vector copy 或 materialized vector constant。 |
| `vclamps` | `%v_dst = vclamps-f32 %v_x, imm_bound` | 对称 f32 clamp；实测 `imm_bound=1.0`。 |
| `classification` | `%vm_dst = vweird.f32 %v_x` | f32 classification mask；精确 special-value truth table 尚待补测。 |
| `carry mask` | `%vm_dst = vc.u32 %v_y, %v_x` | u32 addition carry/进位 mask。 |
| `lane sequence` | `%v_dst = vlaneseq` | 生成 integer/index lane sequence；无显式 source operand。 |

### 8.3 Vector compare、mask 与 select pattern

| 指令大类 | 完整 LLO 指令 pattern | 含义 |
|---|---|---|
| `f32 compare` | `%vm_dst = vcmp.eq.f32.partialorder src_y, src_x`<br>`%vm_dst = vcmp.ne.f32.partialorder src_y, src_x`<br>`%vm_dst = vcmp.lt.f32.partialorder src_y, src_x`<br>`%vm_dst = vcmp.le.f32.partialorder src_y, src_x`<br>`%vm_dst = vcmp.gt.f32.partialorder src_y, src_x`<br>`%vm_dst = vcmp.ge.f32.partialorder src_y, src_x` | 逐 element f32 比较并产生 vector mask；NaN 使用 partial-order 语义。 |
| `s32 compare` | `%vm_dst = vcmp.eq.s32.totalorder src_y, src_x`<br>`%vm_dst = vcmp.ne.s32.totalorder src_y, src_x`<br>`%vm_dst = vcmp.lt.s32.totalorder src_y, src_x`<br>`%vm_dst = vcmp.le.s32.totalorder src_y, src_x`<br>`%vm_dst = vcmp.gt.s32.totalorder src_y, src_x`<br>`%vm_dst = vcmp.ge.s32.totalorder src_y, src_x` | signed 32-bit total-order compare。 |
| `u32 compare` | `%vm_dst = vcmp.gt.u32.totalorder src_y, src_x`<br>`%vm_dst = vcmp.ge.u32.totalorder src_y, src_x` | unsigned 32-bit total-order compare。 |
| `vsel` | `%v_dst = vsel /*vm=*/%vm_mask, /*on_true_vy=*/src_true, /*on_false_vx=*/src_false` | 按 `%vm_mask` 逐 element 选择 true/false source。 |
| `mask logic` | `%vm_dst = vmand %vm_y, %vm_x`<br>`%vm_dst = vmor %vm_y, %vm_x`<br>`%vm_dst = vmxor %vm_y, %vm_x` | vector-mask AND/OR/XOR。 |
| `mask negate` | `%vm_dst = vmneg %vm_x` | vector-mask NOT。 |
| `mask move` | `%vm_dst = vmmov src_mask_or_imm` | mask copy/materialized mask constant。 |
| `mask create` | `%vm_dst = vcmask packed_bounds_imm /* [s_lo:s_hi,l_lo:l_hi] */` | 从 packed rectangle bounds 创建 2-D vector mask。 |

### 8.4 Convert、pack 与 unpack pattern

| 指令大类 | 完整 LLO 指令 pattern | 含义 |
|---|---|---|
| `integer/float convert` | `%v_dst = vcvt.s32.f32 %v_x`<br>`%v_dst = vcvt.f32.s32 %v_x` | s32→f32 与 f32→s32 vector conversion；f32→s32 前可另有 `vtrunc`。 |
| `stochastic convert` | `%v_dst = vcvt.sr.f32.bf16 %v_random_bits, %v_f32` | 使用显式 random bits 对 f32 做 stochastic bf16 rounding。 |
| `compact pack` | `%v_dst = vpack.c.bf16 %v_y, %v_x`<br>`%v_dst = vpack.c.b16 %v_y, %v_x` | 将两个转换后 half/bf16 payload 打包到 32-bit lanes。 |
| `interleaved pack` | `%v_dst = vpack.i.bf16 %v_y, %v_x` | `pltpu.pack_elementwise` 的 bf16 interleaving pack。 |
| `compact unpack` | `%v_dst = vunpack.c.l.bf16 %v_x` | 取 compact bf16 low half并扩展到 f32 lane。 |
| `interleaved unpack` | `%v_dst = vunpack.i.l.bf16 %v_x` | 取 interleaved packed bf16 的 low element。 |

### 8.5 EUP pattern

EUP issue 的左值是 `%token`；最终 vector 结果由 `vpop.eup` 返回。下面每条 issue 都是本轮 bundle 中出现的完整形式。

| 指令大类 | 完整 LLO 指令 pattern | 含义 |
|---|---|---|
| `tanh` | `%token = vtanh.f32 %v_x`<br>`%v_dst = vpop.eup %token` | 发起 tanh 并延迟取回结果。 |
| `pow2` | `%token = vpow2.f32 %v_x`<br>`%v_dst = vpop.eup %token` | 发起 2^x 并延迟取回结果。 |
| `reciprocal` | `%token = vrcp.f32 %v_x`<br>`%v_dst = vpop.eup %token` | 发起 1/x 并延迟取回结果。 |
| `log2` | `%token = vlog2.f32 %v_x`<br>`%v_dst = vpop.eup %token` | 发起 log2(x) 并延迟取回结果。 |
| `rsqrt` | `%token = vrsqrt.f32 %v_x`<br>`%v_dst = vpop.eup %token` | 发起 reciprocal-sqrt 并延迟取回结果。 |
| `sinq` | `%token = vsinq.f32 %v_x`<br>`%v_dst = vpop.eup %token` | 发起 quadrant-reduced sine core。 |
| `cosq` | `%token = vcosq.f32 %v_x`<br>`%v_dst = vpop.eup %token` | 发起 quadrant-reduced cosine core。 |

### 8.6 XLU reduction、permute 与 transpose pattern

| 指令大类 | 完整 LLO 指令 pattern | 含义 |
|---|---|---|
| `sum reduce` | `%token = vadd.xlane.f32.xlu0 %v_x`<br>`%token = vadd.xlane.f32.xlu1 %v_x`<br>`%v_dst = vpop.xlane.xlu0 %token`<br>`%v_dst = vpop.xlane.xlu1 %token` | 在指定 XLU 发起 f32 cross-lane sum，并从同一 XLU 取结果。 |
| `f32 min reduce` | `%token = vmin.xlane.f32.xlu0 %v_x`<br>`%token = vmin.xlane.f32.xlu1 %v_x`<br>`%v_dst = vpop.xlane.xlu0 %token`<br>`%v_dst = vpop.xlane.xlu1 %token` | f32 cross-lane minimum。 |
| `f32 max reduce` | `%token = vmax.xlane.f32.xlu0 %v_x`<br>`%token = vmax.xlane.f32.xlu1 %v_x`<br>`%v_dst = vpop.xlane.xlu0 %token`<br>`%v_dst = vpop.xlane.xlu1 %token` | f32 cross-lane maximum。 |
| `bf16 min/max reduce` | `%token = vmin.xlane.bf16.xlu0 %v_x`<br>`%token = vmax.xlane.bf16.xlu0 %v_x`<br>`%v_dst = vpop.xlane.xlu0 %token` | bf16 cross-lane min/max。 |
| `index reduce` | `%token = vmin.index.xlane.f32.xlu0 %v_x`<br>`%token = vmax.index.xlane.f32.xlu0 %v_x`<br>`%v_dst = vpop.xlane.xlu0 %token` | value+index argmin/argmax reduction。 |
| `lane rotate` | `%token = vrot.lane.b32.xlu0 %v_x, %s_or_imm_amount`<br>`%v_dst = vpop.permute.xlu0 %token` | 在 XLU0 发起 32-bit lane rotate 并取 permute result。 |
| `transpose start` | `%token = vxpose.xlu0.b32.start [1/N] /*vx=*/%v_x, /*width=*/width` | 开始 N-chunk b32 transpose sequence。 |
| `transpose continue` | `%token = vxpose.xlu0.b32.cont [i/N] /*vx=*/%v_x, /*width=*/width` | 提交第 `2..N-1` 个 transpose input chunk。 |
| `transpose end` | `%token = vxpose.xlu0.b32.end [N/N] /*vx=*/%v_x, /*width=*/width` | 提交最后一个 transpose input chunk。 |
| `transpose pop` | `%v_dst = vpop.trf.xlu0` | 无显式 token operand；按 FIFO 顺序逐 chunk 取 transpose result。 |

### 8.7 Pattern 的编码与调度边界

- Pattern 表达的是 final-bundle text grammar，不保证所有 `src` 组合都能在同一个 slot 编码；寄存器窗口、Y-source、immediate slot 和 predicate pool 仍可能迫使 compiler 插入 move 或拆 bundle。
- `.xlu0/.xlu1` 是实际资源选择的一部分，不能从性能模型中删掉。
- `vpop.eup`、`vpop.xlane.*`、`vpop.permute.*` 的 `%token` 是调度依赖；`vpop.trf.xlu0` 则在实测文本中没有显式 token operand。
- Pattern 没有列入 `vld/vst/dma/vsync/scalar_lea/shalt` 等非计算脚手架；它们仍完整保留在原始 bundle。
- 当前表覆盖本轮 112/112 个实测计算 mnemonic。新 probe 出现新 mnemonic 时，应同时更新 pattern 表、语义表和 primitive mapping。

## 9. JAX/Pallas primitive → final LLO 摘要

下面只列可直接指导 kernel 设计的关键结果；53 个 case 的逐 mnemonic 次数在 `analysis_refined/primitive_to_llo.csv` 和 `report.md` 中完整保存。

| Source primitive/case | 主要实测 lowering | 结论 |
|---|---|---|
| `lax.add/sub/mul` f32/bf16/s32/u32 | 对应 `vadd.*` / `vsub.*` / `vmul.*` | 单 primitive case，归因强 |
| `lax.div` f32 | `vrcp.f32` → `vpop.eup` → `vmul.f32` | 不是 native vector divide |
| `lax.div/rem` s32/u32 | 31/32 轮级别的 compare/select/shift/add/sub 展开 | 极大代码体积；性能分析不能按一条 div 计 |
| `lax.min/max` f32/bf16 | `vmin.*` / `vmax.*` | native elementwise |
| `lax.min/max` s32 | compare + `vsel` | 本轮没有 emitted `vmin.s32/vmax.s32` |
| `lax.eq/ne/lt/le/gt/ge` | `vcmp.<cond>.<dtype>.<order>` | 输出 `%vm` |
| `lax.select` | `vsel` | mask-select native |
| `lax.exp/exp2` | scale/add + `vpow2.f32`/`vpop.eup` | exp 通过 pow2 core |
| `lax.log/log1p` | `vlog2.f32`/`vpop.eup` + correction arithmetic | log1p 为复合 lowering |
| `lax.sqrt/rsqrt` | `vrsqrt.f32`/`vpop.eup` + multiply/select | sqrt 由 rsqrt core 派生 |
| `lax.sin/cos/tan` | range reduction + `vsinq/vcosq/vrcp` + pops | 复合、高代码量 |
| `lax.tanh/logistic` | `vtanh`；logistic 用 pow2+reciprocal | 两条不同 EUP 路径 |
| f32↔bf16 | `vpack.c.bf16` / `vunpack.c.l.bf16` | packing 是数据布局的一部分 |
| f32↔s32 | `vtrunc`+`vcvt.f32.s32` / `vcvt.s32.f32` | f32→s32 是多条 |
| `pltpu.stochastic_round` | `vcvt.sr.f32.bf16` + `vpack.c.b16` | 需要 random-bits operand |
| `lax.reduce_sum/min/max` | `v*.xlane.*` → `vpop.xlane.*` | deferred XLU result |
| `lax.argmin/argmax` | `vmin/max.index.xlane` → pop | value/index 联合归约 |
| `lax.transpose` 128×128 f32 | start + 14 cont + end + 16 pop | shape 决定多 chunk 序列 |
| `pltpu.roll` | `vrot.lane.b32.xlu0` → `vpop.permute.xlu0` | native XLU permute path |
| `lax.iota` → f32 | integer `vlaneseq` → `vcvt.s32.f32` → add | `tpu.iota` result 必须先是 integer/index |
| f32→u32 bitcast | 无计算 mnemonic | zero-cost type reinterpretation |

归因规则：单 primitive case 可作为较强的 source→LLO 证据；复合 case 只证明这些 mnemonic 与该 primitive 集合共同出现。向量 case 的映射已排除 scalar bounds/address/DMA 脚手架。

## 10. 最小可复现观测

### 9.1 Native vector add

Source：`lax.add(x, y)`，`x/y: f32[8,128]`。

```text
0x1 : { %v12_v0 = vld ... }
0x2 : { %v13_v1 = vld ... }
0x3 : { ... ;; %v14_v2 = vadd.f32 %v13_v1, %v12_v0 ;; ... }
```

结果：一条 `vadd.f32`；numerical check 通过。

### 9.2 f32 divide

Source：`lax.div(x, y)`，正数 f32 输入避免除零干扰。

```text
0x2 : { ... ;; %32 = vrcp.f32 %v13_v0 ;; ... }
0x8 : { %v33_v2 = vpop.eup %32 }
0x9 : { %v34_v3 = vmul.f32 %v33_v2, %v12_v1 }
```

结果：reciprocal issue、延迟 pop、multiply；numerical check 通过。

### 9.3 Reduction

Source：`jnp.sum(x, axis=1, keepdims=True)` 后 broadcast，`x: f32[8,128]`。

```text
0x2 : { ... ;; %10 = vadd.xlane.f32.xlu0 %v9_v0 ;; ... }
0x8 : { %v11_v1 = vpop.xlane.xlu0 %10 }
```

结果：XLU issue/pop 分离；numerical check 通过。

### 9.4 非法 direct f32 iota 与合法生成方式

直接让 `tpu.iota` 产生 f32 的 verifier 诊断是：

```text
tpu.iota op result #0 must be vector of integer or index values,
but got vector<1x128xf32>
```

合法方式是先在 kernel 中生成 `jnp.arange(..., dtype=jnp.int32)`，再 `.astype(jnp.float32)`。复测 bundle 出现：

```text
vlaneseq
vcvt.s32.f32
vadd.f32
```

结果 shape `[8,128]`，max absolute error `0.0`。

## 11. 硬件支持范围与待补 probe

本轮只把“在 v7x + JAX 0.10.2 + libtpu 0.0.42.1 成功编译并执行”的成员标记为 **v7x observed-supported**。参考 enum 中存在但未触发的成员不能据此判定“不支持”。优先补充：

| 缺口 | 需要的最小 sweep | 要验证的约束 |
|---|---|---|
| F8/HF16、s8/u8/s4/u4 convert | 每种 source/target dtype 的 isolated case | v7x dtype availability、pack order、饱和/rounding |
| EUP bf16、erf、shifted-sigmoid、fused AndPop | 单函数、单输出 case | bare push vs fused lowering、FIFO depth/latency |
| segmented reduction | segment size 1/2/4/8/非幂二 | pattern setup、合法 segment/shape、两个 XLU 冲突 |
| permute/combine/broadcast | 每个 `pltpu`/内部 primitive 单独触发 | pattern encoding、source-bus hazard、result chunk count |
| scalar f32 floor/ceil/min/max/CLZ/add-carry | 用 program_id/SMEM scalar value 防止 vectorization | scalar lane availability 与 operand encoding |
| compare special values | `±0`、NaN、±Inf、subnormal | `partialorder` 精确 truth table、isfinite `vweird` 语义 |
| integer div/rem 边界 | 0、1、-1、INT_MIN/MAX、不同常量除数 | divide-by-zero、overflow、constant specialization |
| transpose shape matrix | 高/宽 8..256，b16/b32，非方形 | chunk/reservation 公式、支持模式、两个 XLU 冲突 |
| tail/alignment | lane 1..127、sublane 1..8、地址偏移 sweep | VMEM/HBM 对齐、mask tail、非法 shape 诊断 |
| register/slot pressure | 1..5 independent VALU op、1..3 predicate、双 XLU | 4 VALU slot、predicate pool、source-bus/issue conflict |

每个新增 case 应保存：source、输入生成、dtype/shape、compile status、verifier error（负例）、final bundle、数值 reference、实际 mnemonic 与 bundle distance。只有在这些边界 sweep 完成后，才把“参考 enum 存在”升级为“v7x 支持范围已验证”。

## 12. 性能分析使用规则

1. 按 final bundle 序号分析 issue，而不是按源码 primitive 数量。
2. 同一 `{}` 内由 `;;` 分隔的指令是可并发 slot issue，不应把条数直接相加为 cycle。
3. EUP/XLU 的 push→pop bundle distance 是依赖可见距离，但不等同于硬件 latency；中间可能有 scheduler 隐藏的独立工作和 reservation 约束。
4. software integer div/rem、trig、erf_inv、nextafter 等复合 lowering 必须用实际 bundle 数/slot occupancy 分析。
5. load/store、DMA、bounds check 与 source compute 分开统计；但做 end-to-end kernel 性能时仍需全部计入。
6. `bitcast` 等 optimized-away primitive 不产生计算指令；不要强行为它分配一个虚构 opcode 成本。
7. 数字 enum 只用于 LLO IR 对照；最终硬件约束判断以对应 generation 的 bundle encoder 和实测 scheduled mnemonic 为准。

## 13. 参考与证据等级

- LloOpcode enum 与家族范围：<https://gh.evko.io/crucible-notes/libtpu/isa/llo-opcode-enum.html>
- **Observed**：本轮 v7x final bundle 直接出现，且关联 case 成功。
- **Observed + checked**：除 bundle 证据外，输出与 JAX reference 相符。
- **Reference**：来自上述 reverse-engineered enum/ISA 页面，尚未由本轮 case 单独验证。
- **Inferred**：由指令组合或上下文推断；本文对 `vweird.f32`、NaN ordering、地址对齐等未完成项明确保留，不将其写成已确认事实。
