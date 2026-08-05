# Non-MXU Pallas LLO Final-Bundle Report
- Cases: 53
- Cases with named final bundle: 53
- Unique compute mnemonics: 112
- MXU mnemonics (must be empty): `{}`

## Opcode counts

| LLO mnemonic | Family | Count |
|---|---|---:|
| `pnand` | predicate | 114 |
| `pneg` | predicate | 4 |
| `por` | predicate | 61 |
| `sadd.s32` | spu | 13 |
| `sand.u32` | spu | 5 |
| `scmp.eq.s32.totalorder` | predicate | 6 |
| `scmp.ge.s32.totalorder` | predicate | 6 |
| `scmp.gt.s32.totalorder` | predicate | 1 |
| `scmp.lt.s32.totalorder` | predicate | 112 |
| `scmp.lt.u32.totalorder` | predicate | 57 |
| `scmp.ne.s32.totalorder` | predicate | 110 |
| `scvt.s32.f32` | spu | 2 |
| `sdivrem.u32` | spu | 1 |
| `smin.u32` | spu | 1 |
| `smov` | spu | 76 |
| `smul.u32` | spu | 2 |
| `smulhi.u32` | spu | 1 |
| `sor.u32` | spu | 1 |
| `sphi` | spu | 8 |
| `spop.drf` | spu | 1 |
| `sshll.u32` | spu | 62 |
| `sshrl.u32` | spu | 2 |
| `ssub.s32` | spu | 5 |
| `sxor.u32` | spu | 1 |
| `vadd.bf16` | vpu | 3 |
| `vadd.f32` | vpu | 38 |
| `vadd.s32` | vpu | 152 |
| `vadd.xlane.f32.xlu0` | xlu_reduce | 2 |
| `vadd.xlane.f32.xlu1` | xlu_reduce | 2 |
| `vand.u32` | vpu | 26 |
| `vc.u32` | vpu | 1 |
| `vceil.f32` | vpu | 1 |
| `vclamps-f32` | vpu | 1 |
| `vclz` | vpu | 2 |
| `vcmask` | vector_mask | 1 |
| `vcmp.eq.f32.partialorder` | vector_mask | 17 |
| `vcmp.eq.s32.totalorder` | vector_mask | 7 |
| `vcmp.ge.f32.partialorder` | vector_mask | 1 |
| `vcmp.ge.s32.totalorder` | vector_mask | 1 |
| `vcmp.ge.u32.totalorder` | vector_mask | 128 |
| `vcmp.gt.f32.partialorder` | vector_mask | 3 |
| `vcmp.gt.s32.totalorder` | vector_mask | 4 |
| `vcmp.gt.u32.totalorder` | vector_mask | 1 |
| `vcmp.le.f32.partialorder` | vector_mask | 2 |
| `vcmp.le.s32.totalorder` | vector_mask | 1 |
| `vcmp.lt.f32.partialorder` | vector_mask | 7 |
| `vcmp.lt.s32.totalorder` | vector_mask | 15 |
| `vcmp.ne.f32.partialorder` | vector_mask | 4 |
| `vcmp.ne.s32.totalorder` | vector_mask | 2 |
| `vcosq.f32` | eup | 1 |
| `vcvt.f32.s32` | convert_pack | 8 |
| `vcvt.s32.f32` | convert_pack | 6 |
| `vcvt.sr.f32.bf16` | convert_pack | 1 |
| `vfloor.f32` | vpu | 2 |
| `vlaneseq` | vpu | 1 |
| `vlog2.f32` | eup | 4 |
| `vmand` | vector_mask | 6 |
| `vmax.bf16` | vpu | 1 |
| `vmax.f32` | vpu | 1 |
| `vmax.index.xlane.f32.xlu0` | xlu_reduce | 1 |
| `vmax.xlane.bf16.xlu0` | xlu_reduce | 1 |
| `vmax.xlane.f32.xlu0` | xlu_reduce | 2 |
| `vmax.xlane.f32.xlu1` | xlu_reduce | 1 |
| `vmin.bf16` | vpu | 1 |
| `vmin.f32` | vpu | 1 |
| `vmin.index.xlane.f32.xlu0` | xlu_reduce | 1 |
| `vmin.u32` | vpu | 6 |
| `vmin.xlane.bf16.xlu0` | xlu_reduce | 1 |
| `vmin.xlane.f32.xlu0` | xlu_reduce | 3 |
| `vmmov` | vector_mask | 1 |
| `vmneg` | vector_mask | 1 |
| `vmor` | vector_mask | 5 |
| `vmov` | vpu | 19 |
| `vmul.bf16` | vpu | 3 |
| `vmul.f32` | vpu | 37 |
| `vmul.u32` | vpu | 10 |
| `vmul.u32.u64.high` | vpu | 2 |
| `vmul.u32.u64.low` | vpu | 2 |
| `vmxor` | vector_mask | 3 |
| `vor.u32` | vpu | 136 |
| `vpack.c.b16` | convert_pack | 1 |
| `vpack.c.bf16` | convert_pack | 2 |
| `vpack.i.bf16` | convert_pack | 1 |
| `vpcnt` | vpu | 1 |
| `vpop.eup` | eup | 17 |
| `vpop.permute.xlu0` | xlu_permute | 1 |
| `vpop.trf.xlu0` | xlu_permute | 16 |
| `vpop.xlane.xlu0` | xlu_reduce | 11 |
| `vpop.xlane.xlu1` | xlu_reduce | 3 |
| `vpow2.f32` | eup | 4 |
| `vrcp.f32` | vpu | 4 |
| `vrot.lane.b32.xlu0` | xlu_permute | 1 |
| `vround.rtna.f32` | vpu | 1 |
| `vround.rtne.f32` | vpu | 1 |
| `vrsqrt.f32` | eup | 2 |
| `vsel` | vector_mask | 339 |
| `vshll.u32` | vpu | 263 |
| `vshra.s32` | vpu | 2 |
| `vshrl.u32` | vpu | 140 |
| `vsinq.f32` | eup | 1 |
| `vsub.bf16` | vpu | 1 |
| `vsub.f32` | vpu | 5 |
| `vsub.s32` | vpu | 141 |
| `vtanh.f32` | eup | 1 |
| `vtrunc.f32` | vpu | 3 |
| `vunpack.c.l.bf16` | convert_pack | 2 |
| `vunpack.i.l.bf16` | convert_pack | 1 |
| `vweird.f32` | vpu | 2 |
| `vxor.u32` | vpu | 8 |
| `vxpose.xlu0.b32.cont` | xlu_permute | 14 |
| `vxpose.xlu0.b32.end` | xlu_permute | 1 |
| `vxpose.xlu0.b32.start` | xlu_permute | 1 |

## Primitive-correlated observations

| Case | JAX/Pallas primitives | Observed compute LLO | Status |
|---|---|---|---|
| `abs_neg_sign_f32` | `lax.abs`, `lax.neg`, `lax.sign` | `vadd.f32`×2, `vand.u32`×2, `vcmp.gt.f32.partialorder`×1, `vmul.f32`×1, `vor.u32`×1, `vsel`×1, `vsub.f32`×1 | succeeded |
| `abs_neg_sign_s32` | `lax.abs`, `lax.neg`, `lax.sign` | `vadd.s32`×2, `vcmp.gt.s32.totalorder`×1, `vcmp.lt.s32.totalorder`×1, `vmin.u32`×1, `vmov`×1, `vmul.u32`×1, `vsel`×2, `vsub.s32`×2 | succeeded |
| `add_bf16` | `lax.add` | `vadd.bf16`×1 | succeeded |
| `add_f32` | `lax.add` | `vadd.f32`×1 | succeeded |
| `add_s32` | `lax.add` | `vadd.s32`×1 | succeeded |
| `argminmax_f32` | `lax.argmin`, `lax.argmax` | `vadd.s32`×1, `vmax.index.xlane.f32.xlu0`×1, `vmin.index.xlane.f32.xlu0`×1, `vmul.u32`×1, `vpop.xlane.xlu0`×2 | succeeded |
| `bf16_to_f32` | `lax.convert_element_type` | `vunpack.c.l.bf16`×1 | succeeded |
| `bitcast_f32_u32` | `lax.bitcast_convert_type` | — | succeeded |
| `bitwise_u32` | `lax.and`, `lax.or`, `lax.xor` | `vand.u32`×1, `vor.u32`×1, `vxor.u32`×1 | succeeded |
| `clz_popcount` | `lax.clz`, `lax.population_count` | `vadd.s32`×1, `vclz`×1, `vpcnt`×1 | succeeded |
| `compare_select_f32` | `lax.eq`, `lax.ne`, `lax.lt`, `lax.le`, `lax.gt`, `lax.ge`, `lax.select_n` | `vadd.f32`×5, `vcmp.eq.f32.partialorder`×1, `vcmp.ge.f32.partialorder`×1, `vcmp.gt.f32.partialorder`×1, `vcmp.le.f32.partialorder`×1, `vcmp.lt.f32.partialorder`×1, `vcmp.ne.f32.partialorder`×1, `vmul.f32`×5, `vsel`×6 | succeeded |
| `compare_select_s32` | `lax.eq`, `lax.ne`, `lax.lt`, `lax.le`, `lax.gt`, `lax.ge`, `lax.select_n` | `vadd.s32`×5, `vcmp.ge.s32.totalorder`×1, `vcmp.gt.s32.totalorder`×1, `vcmp.le.s32.totalorder`×1, `vcmp.lt.s32.totalorder`×1, `vmul.u32`×5, `vsel`×4 | succeeded |
| `div_f32` | `lax.div` | `vmul.f32`×1, `vpop.eup`×1, `vrcp.f32`×1 | succeeded |
| `div_s32` | `lax.div` | `vadd.s32`×32, `vcmp.ge.u32.totalorder`×32, `vcmp.lt.s32.totalorder`×2, `vcmp.ne.s32.totalorder`×1, `vmand`×1, `vmin.u32`×2, `vmxor`×1, `vor.u32`×31, `vsel`×64, `vshll.u32`×63, `vshrl.u32`×32, `vsub.s32`×34 | succeeded |
| `div_u32` | `lax.div` | `vadd.s32`×32, `vcmp.ge.u32.totalorder`×32, `vor.u32`×31, `vsel`×63, `vshll.u32`×63, `vshrl.u32`×32, `vsub.s32`×31 | succeeded |
| `erf_inv` | `lax.erf_inv` | `vadd.f32`×12, `vand.u32`×3, `vcmp.eq.f32.partialorder`×3, `vcmp.lt.f32.partialorder`×2, `vlog2.f32`×1, `vmov`×9, `vmul.f32`×15, `vpop.eup`×2, `vrsqrt.f32`×1, `vsel`×14, `vsub.f32`×2 | succeeded |
| `exp_exp2` | `lax.exp`, `lax.exp2` | `vadd.f32`×1, `vmul.f32`×1, `vpop.eup`×2, `vpow2.f32`×2 | succeeded |
| `f32_to_bf16` | `lax.convert_element_type` | `vpack.c.bf16`×1 | succeeded |
| `f32_to_s32` | `lax.convert_element_type` | `vcvt.f32.s32`×1, `vtrunc.f32`×1 | succeeded |
| `iota_f32` | `lax.iota` | `vadd.f32`×1, `vand.u32`×1, `vcvt.s32.f32`×1, `vlaneseq`×1 | succeeded |
| `isfinite_clamp` | `lax.is_finite`, `lax.clamp` | `vclamps-f32`×1, `vmmov`×1, `vmxor`×1, `vsel`×1, `vweird.f32`×1 | succeeded |
| `log_log1p` | `lax.log`, `lax.log1p` | `vadd.f32`×3, `vand.u32`×1, `vcmp.lt.f32.partialorder`×1, `vlog2.f32`×2, `vmul.f32`×4, `vpop.eup`×2, `vsel`×1 | succeeded |
| `minmax_bf16` | `lax.min`, `lax.max` | `vadd.bf16`×1, `vmax.bf16`×1, `vmin.bf16`×1, `vmul.bf16`×1 | succeeded |
| `minmax_f32` | `lax.min`, `lax.max` | `vadd.f32`×1, `vmax.f32`×1, `vmin.f32`×1, `vmul.f32`×1 | succeeded |
| `minmax_s32` | `lax.min`, `lax.max` | `vadd.s32`×1, `vcmp.gt.s32.totalorder`×1, `vcmp.lt.s32.totalorder`×1, `vmul.u32`×1, `vsel`×2 | succeeded |
| `mul_bf16` | `lax.mul` | `vmul.bf16`×1 | succeeded |
| `mul_f32` | `lax.mul` | `vmul.f32`×1 | succeeded |
| `mul_u32` | `lax.mul` | `vmul.u32`×1 | succeeded |
| `pack_f32_bf16` | `pltpu.pack_elementwise` | `vpack.i.bf16`×1 | succeeded |
| `pow_nextafter` | `lax.pow`, `lax.nextafter` | `vadd.f32`×1, `vadd.s32`×1, `vand.u32`×5, `vcmp.eq.f32.partialorder`×8, `vcmp.eq.s32.totalorder`×3, `vcmp.gt.f32.partialorder`×1, `vcmp.gt.u32.totalorder`×1, `vcmp.lt.f32.partialorder`×3, `vcmp.lt.s32.totalorder`×1, `vcmp.ne.f32.partialorder`×3, `vcmp.ne.s32.totalorder`×1, `vcvt.f32.s32`×1, `vlog2.f32`×1, `vmand`×5, `vmneg`×1, `vmor`×5, `vmov`×3, `vmul.f32`×1, `vmxor`×1, `vor.u32`×1, `vpop.eup`×2, `vpow2.f32`×1, `vsel`×19, `vtrunc.f32`×2, `vxor.u32`×2 | succeeded |
| `reduce_minmax_bf16` | `lax.reduce_min`, `lax.reduce_max` | `vadd.bf16`×1, `vcmask`×1, `vmax.xlane.bf16.xlu0`×1, `vmin.xlane.bf16.xlu0`×1, `vmul.bf16`×1, `vpop.xlane.xlu0`×2, `vsel`×2 | succeeded |
| `reduce_minmax_f32` | `lax.reduce_min`, `lax.reduce_max` | `vadd.f32`×1, `vmax.xlane.f32.xlu0`×1, `vmin.xlane.f32.xlu0`×1, `vmul.f32`×1, `vpop.xlane.xlu0`×2 | succeeded |
| `reduce_s32` | `lax.reduce_sum`, `lax.reduce_min`, `lax.reduce_max` | `vadd.s32`×5, `vadd.xlane.f32.xlu1`×2, `vand.u32`×1, `vcmp.eq.f32.partialorder`×2, `vcvt.f32.s32`×6, `vcvt.s32.f32`×3, `vmax.xlane.f32.xlu0`×1, `vmax.xlane.f32.xlu1`×1, `vmin.xlane.f32.xlu0`×2, `vpop.xlane.xlu0`×3, `vpop.xlane.xlu1`×3, `vsel`×2, `vshll.u32`×3, `vshra.s32`×1, `vshrl.u32`×1 | succeeded |
| `reduce_sum_bf16` | `lax.reduce_sum` | `vadd.xlane.f32.xlu0`×1, `vpack.c.bf16`×1, `vpop.xlane.xlu0`×1, `vunpack.c.l.bf16`×1 | succeeded |
| `reduce_sum_f32` | `lax.reduce_sum` | `vadd.xlane.f32.xlu0`×1, `vpop.xlane.xlu0`×1 | succeeded |
| `rem_f32` | `lax.rem` | `vand.u32`×4, `vcmp.eq.f32.partialorder`×1, `vfloor.f32`×1, `vmul.f32`×2, `vor.u32`×1, `vpop.eup`×1, `vrcp.f32`×1, `vsel`×1, `vsub.f32`×1 | succeeded |
| `rem_s32` | `lax.rem` | `vadd.s32`×31, `vcmp.ge.u32.totalorder`×32, `vcmp.lt.s32.totalorder`×1, `vmin.u32`×2, `vor.u32`×31, `vsel`×64, `vshll.u32`×62, `vshrl.u32`×32, `vsub.s32`×35 | succeeded |
| `rem_u32` | `lax.rem` | `vadd.s32`×31, `vcmp.ge.u32.totalorder`×32, `vor.u32`×31, `vsel`×63, `vshll.u32`×62, `vshrl.u32`×32, `vsub.s32`×32 | succeeded |
| `roll_f32` | `pltpu.roll` | `vpop.permute.xlu0`×1, `vrot.lane.b32.xlu0`×1 | succeeded |
| `round_ceil_floor` | `lax.round`, `lax.ceil`, `lax.floor` | `vadd.f32`×3, `vceil.f32`×1, `vfloor.f32`×1, `vround.rtna.f32`×1, `vround.rtne.f32`×1 | succeeded |
| `s32_to_f32` | `lax.convert_element_type` | `vcvt.s32.f32`×1 | succeeded |
| `scalar_arith` | `pl.program_id`, `lax.add`, `lax.sub`, `lax.mul`, `lax.div`, `lax.rem`, `lax.min`, `lax.max` | `pnand`×6, `pneg`×2, `por`×5, `sadd.s32`×8, `sand.u32`×2, `scmp.eq.s32.totalorder`×3, `scmp.ge.s32.totalorder`×3, `scmp.gt.s32.totalorder`×1, `scmp.lt.s32.totalorder`×5, `scmp.lt.u32.totalorder`×3, `scmp.ne.s32.totalorder`×4, `scvt.s32.f32`×1, `sdivrem.u32`×1, `smin.u32`×1, `smov`×12, `smul.u32`×1, `smulhi.u32`×1, `sphi`×4, `spop.drf`×1, `sshll.u32`×5, `sshrl.u32`×1, `ssub.s32`×4, `vadd.f32`×1 | succeeded |
| `scalar_logic_compare` | `pl.program_id`, `lax.and`, `lax.or`, `lax.xor`, `lax.shift_left`, `lax.shift_right_logical`, `lax.lt`, `lax.select_n` | `pnand`×6, `pneg`×2, `por`×5, `sadd.s32`×5, `sand.u32`×3, `scmp.eq.s32.totalorder`×3, `scmp.ge.s32.totalorder`×3, `scmp.lt.s32.totalorder`×5, `scmp.lt.u32.totalorder`×3, `scmp.ne.s32.totalorder`×4, `scvt.s32.f32`×1, `smov`×10, `smul.u32`×1, `sor.u32`×1, `sphi`×4, `sshll.u32`×6, `sshrl.u32`×1, `ssub.s32`×1, `sxor.u32`×1, `vadd.f32`×1 | succeeded |
| `shifts` | `lax.shift_left`, `lax.shift_right_logical`, `lax.shift_right_arithmetic` | `vshll.u32`×1, `vshra.s32`×1, `vshrl.u32`×1, `vxor.u32`×2 | succeeded |
| `sin_cos_tan` | `lax.sin`, `lax.cos`, `lax.tan` | `vadd.f32`×2, `vadd.s32`×9, `vand.u32`×7, `vc.u32`×1, `vclz`×1, `vcmp.eq.s32.totalorder`×4, `vcmp.gt.s32.totalorder`×1, `vcmp.le.f32.partialorder`×1, `vcmp.lt.s32.totalorder`×8, `vcosq.f32`×1, `vcvt.s32.f32`×1, `vmin.u32`×1, `vmov`×6, `vmul.f32`×2, `vmul.u32`×1, `vmul.u32.u64.high`×2, `vmul.u32.u64.low`×2, `vor.u32`×8, `vpop.eup`×3, `vrcp.f32`×1, `vsel`×28, `vshll.u32`×9, `vshrl.u32`×10, `vsinq.f32`×1, `vsub.s32`×6, `vweird.f32`×1, `vxor.u32`×3 | succeeded |
| `sqrt_rsqrt` | `lax.sqrt`, `lax.rsqrt` | `vadd.f32`×1, `vand.u32`×1, `vcmp.eq.f32.partialorder`×2, `vmul.f32`×1, `vpop.eup`×1, `vrsqrt.f32`×1, `vsel`×2 | succeeded |
| `stochastic_bf16` | `pltpu.stochastic_round` | `vcvt.sr.f32.bf16`×1, `vpack.c.b16`×1 | succeeded |
| `sub_bf16` | `lax.sub` | `vsub.bf16`×1 | succeeded |
| `sub_f32` | `lax.sub` | `vsub.f32`×1 | succeeded |
| `sub_s32` | `lax.sub` | `vsub.s32`×1 | succeeded |
| `tanh_logistic` | `lax.tanh`, `lax.logistic` | `vadd.f32`×2, `vmul.f32`×1, `vpop.eup`×3, `vpow2.f32`×1, `vrcp.f32`×1, `vtanh.f32`×1 | succeeded |
| `transpose_f32` | `lax.transpose` | `vpop.trf.xlu0`×16, `vxpose.xlu0.b32.cont`×14, `vxpose.xlu0.b32.end`×1, `vxpose.xlu0.b32.start`×1 | succeeded |
| `unpack_bf16_f32` | `pltpu.unpack_elementwise` | `vunpack.i.l.bf16`×1 | succeeded |

## Attribution rule

Vector cases exclude scalar bounds-check, address, and DMA-loop scaffolding from source attribution. A one-primitive case supports an isolated source→LLO observation. A compound case only establishes that the listed LLO mnemonics occur in that source primitive set; it does not prove a 1:1 mapping. Scalar-grid probes remain compound correlations because loop control and the tested SPU operations share the scalar slot. Final scheduled bundles are authoritative for emitted instructions, while unsupported/optimized-away enum members remain unobserved.
