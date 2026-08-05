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

当前状态应理解为：**已完整保存并解释这 53 个 case 实际发出的 112 种计算 mnemonic，但尚不能声称已经触发参考 enum 中每一个非 MXU 计算成员。** 未触发项和下一轮 probe 矩阵见第 10 节。

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

5–7 章中的 pattern 元变量统一如下：`%v/%s/%p/%vm` 分别表示 vector/scalar/predicate/vector-mask SSA value，`%token` 表示 deferred-result dependency，`src` 表示该位置可编码的寄存器或立即数，`[(%pred),]` 表示可选的 scalar predicate，其中 `%pred` 可为 `%p_guard` 或 `!%p_guard`。dump 中如 `%v14_v2` 的前半是 SSA ID，后半是物理寄存器。

| final mnemonic | 完整 final-bundle pattern | 语义；operand → result | 类型/约束 | 观测来源 |
|---|---|---|---|---|
| `sadd.s32` / `ssub.s32` | `%s_dst = sadd.s32 [(%pred),] src_y, src_x`<br>`%s_dst = ssub.s32 [(%pred),] src_y, src_x` | scalar 加/减：`src_y, src_x → %s_dst` | 32-bit two's-complement；source 可为 sreg/imm | scalar-grid |
| `smul.u32` / `smulhi.u32` | `%s_dst = smul.u32 [(%pred),] src_y, src_x`<br>`%s_hi = smulhi.u32 [(%pred),] src_y, src_x` | u32×u32 的低/高 32-bit | wide-product helper | scalar-grid |
| `sdivrem.u32` / `spop.drf` | `%token = sdivrem.u32 src_dividend, src_divisor`<br>`%s_dst = spop.drf %token` | 发起 unsigned divide/remainder，再从 DRF FIFO 取结果 | u32；issue/pop 必须配对，不是普通单结果 ALU | `scalar_arith` |
| `sand.u32` / `sor.u32` / `sxor.u32` | `%s_dst = sand.u32 [(%pred),] src_y, src_x`<br>`%s_dst = sor.u32 [(%pred),] src_y, src_x`<br>`%s_dst = sxor.u32 [(%pred),] src_y, src_x` | scalar bitwise AND/OR/XOR | u32 bit pattern | scalar-grid |
| `sshll.u32` / `sshrl.u32` | `%s_dst = sshll.u32 [(%pred),] src_x, src_shift`<br>`%s_dst = sshrl.u32 [(%pred),] src_x, src_shift` | scalar logical left/right shift | u32；shift amount 越界行为需单测 | scalar-grid 与地址脚手架 |
| `smin.u32` | `%s_dst = smin.u32 [(%pred),] src_y, src_x` | unsigned scalar minimum | u32 | `scalar_arith` |
| `scvt.s32.f32` | `%s_dst = scvt.s32.f32 %s_x` | signed scalar integer 转 f32 | `s32 → f32` | scalar-grid |
| `smov` | `%s_dst = smov [#allocationN]` | scalar copy/materialized constant | 本轮实测为 allocation form；32-bit scalar payload | case 与通用脚手架 |
| `sphi` | `%s_dst = sphi %s_init, %s_backedge` | scalar SSA phi/loop merge | 控制流伪/合流语义，不是数值 primitive | scalar-grid |
| `scmp.eq.s32.totalorder`, `scmp.ne.s32.totalorder` | `%p_dst = scmp.eq.s32.totalorder src_y, src_x`<br>`%p_dst = scmp.ne.s32.totalorder src_y, src_x` | signed scalar equal/not-equal compare：`src,src → %p` | s32 total order | scalar-grid；部分也是 bounds check |
| `scmp.lt.s32.totalorder`, `scmp.ge.s32.totalorder`, `scmp.gt.s32.totalorder` | `%p_dst = scmp.lt.s32.totalorder src_y, src_x`<br>`%p_dst = scmp.ge.s32.totalorder src_y, src_x`<br>`%p_dst = scmp.gt.s32.totalorder src_y, src_x` | signed scalar relational compare | s32 total order | scalar-grid；部分也是 bounds check |
| `scmp.lt.u32.totalorder` | `%p_dst = scmp.lt.u32.totalorder src_y, src_x` | scalar unsigned less-than | u32 total order | bounds/control 与 scalar-grid |
| `pneg` | `%p_dst = pneg %p_x` | predicate NOT：`%p → %p` | scalar predicate | scalar-grid |
| `pnand` | `%p_dst = pnand %p_y, %p_x` | predicate NAND：`%p,%p → %p` | AND 常可由 De Morgan 组合实现 | scalar-grid 与 bounds check |
| `por` | `%p_dst = por %p_y, %p_x` | predicate OR：`%p,%p → %p` | scalar predicate | scalar-grid 与 bounds check |

资源判断：scalar 指令占 scalar slot，predicate 指令占 predicate/相关 scalar 控制资源；同一 bundle 中能否与 VPU、load/store、EUP/XLU 并发，以实际 `;;` 排布为准。高频 `smov/sshll/scmp/pnand/por` 多数来自 Pallas 调用、地址计算和 bounds check。它们仍包含在全局 112-mnemonic inventory 中，但已从向量 primitive 的 source-attribution 表中剔除。

## 6. 实测指令参考：VPU、compare 与 mask

### 6.1 算术、bitwise、shift 与 unary

除另行说明外，向量操作逐 element 作用于完整 vreg；`x/y` 可以是 vreg 或可编码的立即数/Y-source，结果为 vreg。

| final mnemonic | 完整 final-bundle pattern | 语义；operand → result | 数据类型/特别约束 |
|---|---|---|---|
| `vadd.f32`, `vadd.bf16`, `vadd.s32` | `%v_dst = vadd.f32 src_y, src_x`<br>`%v_dst = vadd.bf16 src_y, src_x`<br>`%v_dst = vadd.s32 src_y, src_x` | 逐 element `x + y → %v_dst` | f32/bf16/s32 |
| `vsub.f32`, `vsub.bf16`, `vsub.s32` | `%v_dst = vsub.f32 src_y, src_x`<br>`%v_dst = vsub.bf16 src_y, src_x`<br>`%v_dst = vsub.s32 src_y, src_x` | 逐 element `x - y → %v_dst` | f32/bf16/s32 |
| `vmul.f32`, `vmul.bf16`, `vmul.u32` | `%v_dst = vmul.f32 src_y, src_x`<br>`%v_dst = vmul.bf16 src_y, src_x`<br>`%v_dst = vmul.u32 src_y, src_x` | 逐 element multiplication；u32 返回低 32-bit | f32/bf16/u32 |
| `vmul.u32.u64.low` | `%v_low = vmul.u32.u64.low %v_y, %v_x` | u32×u32 的低 32-bit，并形成 wide-product pair | 必须与 high 形式关联 |
| `vmul.u32.u64.high` | `%v_high = vmul.u32.u64.high /*lhs_vy=*/%v_y, /*rhs_vx=*/%v_x, /*low=*/%v_low` | 从两操作数和 low value 形成高 32-bit | bundle 原文有显式 `low=` dependency |
| `vmin.f32`, `vmin.bf16`, `vmin.u32` | `%v_dst = vmin.f32 src_y, src_x`<br>`%v_dst = vmin.bf16 src_y, src_x`<br>`%v_dst = vmin.u32 src_y, src_x` | 逐 element minimum | s32 min 本轮由 compare+`vsel` 合成 |
| `vmax.f32`, `vmax.bf16` | `%v_dst = vmax.f32 src_y, src_x`<br>`%v_dst = vmax.bf16 src_y, src_x` | 逐 element maximum | s32 max 本轮由 compare+`vsel` 合成 |
| `vand.u32`, `vor.u32`, `vxor.u32` | `%v_dst = vand.u32 src_y, src_x`<br>`%v_dst = vor.u32 src_y, src_x`<br>`%v_dst = vxor.u32 src_y, src_x` | 逐 bit AND/OR/XOR | u32 bit pattern；也用于 f32 sign/exponent manipulation |
| `vshll.u32`, `vshrl.u32`, `vshra.s32` | `%v_dst = vshll.u32 %v_x, src_shift`<br>`%v_dst = vshrl.u32 %v_x, src_shift`<br>`%v_dst = vshra.s32 %v_x, src_shift` | logical-left、logical-right、arithmetic-right shift | 大 shift amount 边界待测 |
| `vclz`, `vpcnt` | `%v_dst = vclz %v_x`<br>`%v_dst = vpcnt %v_x` | count-leading-zero/population-count | 输入按 32-bit word 解释 |
| `vceil.f32`, `vfloor.f32`, `vtrunc.f32` | `%v_dst = vceil.f32 %v_x`<br>`%v_dst = vfloor.f32 %v_x`<br>`%v_dst = vtrunc.f32 %v_x` | ceil/floor/toward-zero truncate | f32 → integral-valued f32 |
| `vround.rtna.f32`, `vround.rtne.f32` | `%v_dst = vround.rtna.f32 %v_x`<br>`%v_dst = vround.rtne.f32 %v_x` | round-to-nearest，tie away/tie even | f32 |
| `vmov` | `%v_dst = vmov src` | vreg copy/materialized vector constant | 32-bit lane payload |
| `vclamps-f32` | `%v_dst = vclamps-f32 %v_x, imm_bound` | 对称 clamp：`x → clamp(x,-bound,+bound)` | f32；实测 bound=1.0；其他 bounds 形式待测 |
| `vweird.f32` | `%vm_dst = vweird.f32 %v_x` | 产生 f32 classification mask | `%v → %vm`；NaN/Inf/subnormal truth table 待测 |
| `vc.u32` | `%vm_dst = vc.u32 %v_y, %v_x` | wide-add carry/进位 mask | `u32,u32 → %vm`；对应 compare/add-carry 族 |
| `vlaneseq` | `%v_dst = vlaneseq` | 生成 lane sequence，无显式 source | 结果必须为 integer/index vector；不能直接生成 f32 |

### 6.2 compare、select 与 vector mask

| final mnemonic | 完整 final-bundle pattern | 语义；operand → result | 数据类型/顺序/约束 |
|---|---|---|---|
| `vcmp.eq.f32.partialorder`, `vcmp.ne.f32.partialorder`, `vcmp.lt.f32.partialorder`, `vcmp.le.f32.partialorder`, `vcmp.gt.f32.partialorder`, `vcmp.ge.f32.partialorder` | `%vm_dst = vcmp.eq.f32.partialorder src_y, src_x`<br>`%vm_dst = vcmp.ne.f32.partialorder src_y, src_x`<br>`%vm_dst = vcmp.lt.f32.partialorder src_y, src_x`<br>`%vm_dst = vcmp.le.f32.partialorder src_y, src_x`<br>`%vm_dst = vcmp.gt.f32.partialorder src_y, src_x`<br>`%vm_dst = vcmp.ge.f32.partialorder src_y, src_x` | 逐 element f32 compare：`src,src → %vm_dst` | `partialorder` 涉及 NaN；精确 NaN truth table 待专项验证 |
| `vcmp.eq.s32.totalorder`, `vcmp.ne.s32.totalorder`, `vcmp.lt.s32.totalorder`, `vcmp.le.s32.totalorder`, `vcmp.gt.s32.totalorder`, `vcmp.ge.s32.totalorder` | `%vm_dst = vcmp.eq.s32.totalorder src_y, src_x`<br>`%vm_dst = vcmp.ne.s32.totalorder src_y, src_x`<br>`%vm_dst = vcmp.lt.s32.totalorder src_y, src_x`<br>`%vm_dst = vcmp.le.s32.totalorder src_y, src_x`<br>`%vm_dst = vcmp.gt.s32.totalorder src_y, src_x`<br>`%vm_dst = vcmp.ge.s32.totalorder src_y, src_x` | signed 32-bit compare：`src,src → %vm_dst` | s32 total order |
| `vcmp.gt.u32.totalorder`, `vcmp.ge.u32.totalorder` | `%vm_dst = vcmp.gt.u32.totalorder src_y, src_x`<br>`%vm_dst = vcmp.ge.u32.totalorder src_y, src_x` | unsigned 32-bit compare：`src,src → %vm_dst` | u32 total order；大量出现在 software integer div/rem |
| `vsel` | `%v_dst = vsel /*vm=*/%vm_mask, /*on_true_vy=*/src_true, /*on_false_vx=*/src_false` | `vm ? true : false → %v_dst` | mask 与数据逻辑 shape 一致；true/false 可为 vreg/立即数 |
| `vmand`, `vmor`, `vmxor` | `%vm_dst = vmand %vm_y, %vm_x`<br>`%vm_dst = vmor %vm_y, %vm_x`<br>`%vm_dst = vmxor %vm_y, %vm_x` | vector-mask AND/OR/XOR：`%vm,%vm → %vm` | 使用独立 mask register file，不占普通 `%v` result |
| `vmneg` | `%vm_dst = vmneg %vm_x` | vector-mask NOT：`%vm → %vm` | mask unary op |
| `vmmov` | `%vm_dst = vmmov src_mask_or_imm` | mask copy/materialized constant | source 可为 mask 或可编码 constant；result 是 `%vm`，不是立即数 |
| `vcmask` | `%vm_dst = vcmask packed_bounds_imm /* [s_lo:s_hi,l_lo:l_hi] */` | 从 packed immediate 创建二维矩形 mask | 本轮 `1043456 /* [0:3,0:127] */`；dump 的 end bound 为 inclusive |

v7x predicate 资源限制与 `%vm` mask 寄存器是两层概念：bundle predicate 控制整条 vector op 是否执行，`%vm` 控制 vreg 内哪些 element 参与。一个 bundle 内过多不同 active predicate 可能无法编码，即使还有空 VALU slot。

## 7. 实测指令参考：convert、pack、EUP 与 XLU

### 7.1 Convert / pack / unpack

| final mnemonic | 完整 final-bundle pattern | 语义；operand → result | 数据类型/布局/约束 |
|---|---|---|---|
| `vcvt.s32.f32` | `%v_dst = vcvt.s32.f32 %v_x` | vector conversion：`s32 → f32` | iota case 先生成 integer sequence，再转 f32 |
| `vcvt.f32.s32` | `%v_dst = vcvt.f32.s32 %v_x` | vector conversion：`f32 → s32` | 与 `vtrunc.f32` 一起出现；source conversion 不是单条 lowering |
| `vcvt.sr.f32.bf16` | `%v_dst = vcvt.sr.f32.bf16 %v_random_bits, %v_f32` | 使用显式 random bits 做 stochastic f32→bf16 rounding | 两个 vector operands；本轮未做 deterministic reference check |
| `vpack.c.bf16` | `%v_dst = vpack.c.bf16 %v_y, %v_x` | compact/convert 两组 f32 lane payload 为 packed bf16 | f32→bf16 与 bf16 reduction output 均触发 |
| `vpack.c.b16` | `%v_dst = vpack.c.b16 %v_y, %v_x` | 将两个 b16 half 组合为 32-bit container | stochastic-round output path |
| `vpack.i.bf16` | `%v_dst = vpack.i.bf16 %v_y, %v_x` | bf16 interleaving pack | `pltpu.pack_elementwise`；两个 unpacked values → packed u32 container |
| `vunpack.c.l.bf16` | `%v_dst = vunpack.c.l.bf16 %v_x` | 从 compact bf16 取 low half并扩为 f32 | bf16→f32 与 bf16 reduction input path |
| `vunpack.i.l.bf16` | `%v_dst = vunpack.i.l.bf16 %v_x` | 从 interleaved packed bf16 取 low element | `pltpu.unpack_elementwise(index=0)`；其他 index 尚未触发 |

直接 `lax.bitcast_convert_type(f32,u32)` 没有留下计算 mnemonic：它是相同 32-bit payload 的类型重解释，被编译器优化为无指令。这是“optimized away”，不是硬件不支持。

### 7.2 EUP deferred-result pipeline

| final mnemonic | 完整 final-bundle pattern | 数学语义；operand → result | Pipeline/资源约束 | 本轮 primitive |
|---|---|---|---|---|
| `vtanh.f32` | `%token = vtanh.f32 %v_x` | 发起 `tanh(x)`；`%v_x → %token` | issue 不产生数值 vreg；结果由后续 `vpop.eup` 收集 | `lax.tanh` |
| `vpow2.f32` | `%token = vpow2.f32 %v_x` | 发起 `2^x`；`%v_x → %token` | deferred EUP result | `lax.exp`、`lax.exp2`、logistic、pow |
| `vrcp.f32` | `%token = vrcp.f32 %v_x` | 发起 `1/x`；`%v_x → %token` | deferred EUP result | f32 div/rem、tan、logistic |
| `vlog2.f32` | `%token = vlog2.f32 %v_x` | 发起 `log2(x)`；`%v_x → %token` | deferred EUP result | log/log1p、pow、erf_inv |
| `vrsqrt.f32` | `%token = vrsqrt.f32 %v_x` | 发起 `1/sqrt(x)`；`%v_x → %token` | deferred EUP result | sqrt/rsqrt、erf_inv |
| `vsinq.f32`, `vcosq.f32` | `%token = vsinq.f32 %v_x`<br>`%token = vcosq.f32 %v_x` | 发起 quadrant-reduced sine/cosine core | 输入需先完成 range reduction；deferred EUP result | sin/cos/tan 复合 lowering |
| `vpop.eup` | `%v_dst = vpop.eup %token` | 从 EUP result FIFO 取对应 issue 的数值结果：`%token → %v_dst` | 必须遵守 issue/token/FIFO 顺序；本轮共 17 次 | 所有上述 EUP paths |

关键约束：不能把 issue 与 pop 当成一个不可拆分指令来做调度或性能计数。实测 `div_f32` 在 bundle `0x2` 发 `vrcp.f32`，到 `0x8` 才 `vpop.eup`，最后以 `vmul.f32` 完成 `x * reciprocal(y)`。参考 cost 页面给出的 EUP latency 目前是低置信度估计，本文不把它当作 v7x 已验证延迟。

### 7.3 Reduction / permute / transpose XLU

| final mnemonic | 完整 final-bundle pattern | 语义；operand → result | 资源/形状/约束 |
|---|---|---|---|
| `vadd.xlane.f32.xlu0`, `vadd.xlane.f32.xlu1` | `%token = vadd.xlane.f32.xlu0 %v_x`<br>`%token = vadd.xlane.f32.xlu1 %v_x` | 发起跨 lane f32 sum reduction：`%v_x → %token` | 占 mnemonic 指定的 XLU；数值结果由匹配的 `vpop.xlane` 收集 |
| `vmin.xlane.f32.xlu0`, `vmin.xlane.f32.xlu1` | `%token = vmin.xlane.f32.xlu0 %v_x`<br>`%token = vmin.xlane.f32.xlu1 %v_x` | 发起跨 lane f32 minimum | deferred XLU result |
| `vmax.xlane.f32.xlu0`, `vmax.xlane.f32.xlu1` | `%token = vmax.xlane.f32.xlu0 %v_x`<br>`%token = vmax.xlane.f32.xlu1 %v_x` | 发起跨 lane f32 maximum | deferred XLU result |
| `vmin.xlane.bf16.xlu0`, `vmax.xlane.bf16.xlu0` | `%token = vmin.xlane.bf16.xlu0 %v_x`<br>`%token = vmax.xlane.bf16.xlu0 %v_x` | 发起 bf16 min/max reduction | 本轮输出前后可见 pack/unpack/mask helper |
| `vmin.index.xlane.f32.xlu0`, `vmax.index.xlane.f32.xlu0` | `%token = vmin.index.xlane.f32.xlu0 %v_x`<br>`%token = vmax.index.xlane.f32.xlu0 %v_x` | 同时归约 value 与 index，用于 argmin/argmax | tie-breaking 规则需重复值专项 case |
| `vpop.xlane.xlu0`, `vpop.xlane.xlu1` | `%v_dst = vpop.xlane.xlu0 %token`<br>`%v_dst = vpop.xlane.xlu1 %token` | 从对应 XLU result FIFO 收集 reduction result：`%token → %v_dst` | issue 到 `xluN` 必须从同一 `xluN` pop |
| `vrot.lane.b32.xlu0` | `%token = vrot.lane.b32.xlu0 %v_x, %s_or_imm_amount` | 发起 lane 维 32-bit rotate/permutation | `pltpu.roll`；返回 permute token |
| `vpop.permute.xlu0` | `%v_dst = vpop.permute.xlu0 %token` | 收集 rotate/permute result | `%token → %v_dst`；受 permute FIFO/order 约束 |
| `vxpose.xlu0.b32.start` | `%token = vxpose.xlu0.b32.start [1/N] /*vx=*/%v_x, /*width=*/width` | 开始 N-chunk b32 transpose sequence | 本轮 width=128、N=16 |
| `vxpose.xlu0.b32.cont` | `%token = vxpose.xlu0.b32.cont [i/N] /*vx=*/%v_x, /*width=*/width` | 提交第 `2..N-1` 个 transpose input chunk | 本轮 `[2/16]..[15/16]`，共 14 条 |
| `vxpose.xlu0.b32.end` | `%token = vxpose.xlu0.b32.end [N/N] /*vx=*/%v_x, /*width=*/width` | 提交最后一个 transpose input chunk | 本轮 `[16/16]` |
| `vpop.trf.xlu0` | `%v_dst = vpop.trf.xlu0` | 从 transpose result FIFO 逐 chunk 取结果 | final text 无显式 token operand；本轮 16 条对应 128×128 f32 tile |

Probe 为保持输出 shape 与输入一致，把单行 reduction result broadcast 回 `(8,128)`；因此 bundle 展示的是“reduction + broadcasted materialization”的完整 kernel，不应把所有 helper 都算成纯 reduction latency。

5–7 章 pattern 表达的是本轮 final-bundle text 中观察到的语法，不保证任意 `src` 组合都能在同一个 slot 编码；register window、Y-source、immediate slot、dual-predicate pool、XLU source bus 和 result FIFO 仍可能迫使 compiler 插入 move 或拆分 bundle。表中不列 `vld/vst/dma/vsync/scalar_lea/shalt` 等非计算脚手架，但它们仍完整保留在原始 bundle。新增 probe 若产生新 mnemonic，应同时更新对应表行、primitive mapping 和约束证据。

## 8. JAX/Pallas primitive → final LLO 摘要

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

## 9. 最小可复现观测

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

## 10. 硬件支持范围与待补 probe

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

## 11. 性能分析使用规则

1. 按 final bundle 序号分析 issue，而不是按源码 primitive 数量。
2. 同一 `{}` 内由 `;;` 分隔的指令是可并发 slot issue，不应把条数直接相加为 cycle。
3. EUP/XLU 的 push→pop bundle distance 是依赖可见距离，但不等同于硬件 latency；中间可能有 scheduler 隐藏的独立工作和 reservation 约束。
4. software integer div/rem、trig、erf_inv、nextafter 等复合 lowering 必须用实际 bundle 数/slot occupancy 分析。
5. load/store、DMA、bounds check 与 source compute 分开统计；但做 end-to-end kernel 性能时仍需全部计入。
6. `bitcast` 等 optimized-away primitive 不产生计算指令；不要强行为它分配一个虚构 opcode 成本。
7. 数字 enum 只用于 LLO IR 对照；最终硬件约束判断以对应 generation 的 bundle encoder 和实测 scheduled mnemonic 为准。

## 12. 参考与证据等级

- LloOpcode enum 与家族范围：<https://gh.evko.io/crucible-notes/libtpu/isa/llo-opcode-enum.html>
- **Observed**：本轮 v7x final bundle 直接出现，且关联 case 成功。
- **Observed + checked**：除 bundle 证据外，输出与 JAX reference 相符。
- **Reference**：来自上述 reverse-engineered enum/ISA 页面，尚未由本轮 case 单独验证。
- **Inferred**：由指令组合或上下文推断；本文对 `vweird.f32`、NaN ordering、地址对齐等未完成项明确保留，不将其写成已确认事实。
