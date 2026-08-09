## Phase 0 host test — atom record data checks.
## Validates the 4-kind MmaAtom contract: the tf32 m16n8k8 NVIDIA atom in
## depth (fragment value counts, layout cosizes, thread counts) and one
## instantiation per other kind (AMX, x86 SIMD, ARM SIMD) as data.
## Run with: nim cpp -r workspace/ceramic/experiments/wip_mma_gemm/test_atoms_host.nim
import std/strformat
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/atoms
import workspace/ceramic/src/atoms_nvidia

const atom = SM80_16x8x8_F32TF32TF32F32_TN

doAssert atom.mnk == (m: 16, n: 8, k: 8), "mnk"
doAssert atom.kind == bkGPU_TensorCore
doAssert atom.instr == "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
doAssert atom.aType == mdtTF32 and atom.bType == mdtTF32 and atom.cType == mdtF32
doAssert atom.scaleMode == smNone

# cosize checks: (T,V) layouts must tile the operand tile exactly
#   A: (16·8)/32 = 4 values/thread, B: (8·8)/32 = 2, C: (16·8)/32 = 4
let vA = atom.valuesPerThread(opA).toIntVal()
let vB = atom.valuesPerThread(opB).toIntVal()
let vC = atom.valuesPerThread(opC).toIntVal()
doAssert vA == 4, &"A vals: {vA}"
doAssert vB == 2, &"B vals: {vB}"
doAssert vC == 4, &"C vals: {vC}"

# layout cosize == tile size (each layout covers the whole operand tile)
doAssert cosize(atom.aLayout).toIntVal() == 16 * 8, &"A cosize {cosize(atom.aLayout)}"
doAssert cosize(atom.bLayout).toIntVal() == 8 * 8,  &"B cosize {cosize(atom.bLayout)}"
doAssert cosize(atom.cLayout).toIntVal() == 16 * 8, &"C cosize {cosize(atom.cLayout)}"

# T-mode = 32 threads for all three layouts
doAssert atom.threadCount(opA).toIntVal() == 32
doAssert atom.threadCount(opB).toIntVal() == 32
doAssert atom.threadCount(opC).toIntVal() == 32

echo "  OK — tf32 atom record data checks passed"

# ── Other kinds: prove all 4 kinds are expressible as data ──
import workspace/ceramic/src/atoms_x86_amx
import workspace/ceramic/src/atoms_x86_simd
import workspace/ceramic/src/atoms_arm_simd

const amx = AMX_16x16x16_TDPBF16PS
doAssert amx.kind == bkCPU_X86_AMX
doAssert amx.instr == "tdpbf16ps"
doAssert amx.threadCount(opA).toIntVal() == 1, &"AMX T: {amx.threadCount(opA)}"   # T=1
doAssert amx.valuesPerThread(opC).toIntVal() == 256

const simd = X86_AVX512_SGEMM_14x32
doAssert simd.kind == bkCPU_SIMD
doAssert simd.isa == siAVX512
doAssert simd.nbScalars == 16 and simd.nbVecsNr == 2
doAssert simd.conversionPoint == cpEndOfK

const vnni = X86_AVX512_VNNI_DPBSSD
doAssert vnni.cType == mdtInt32
doAssert vnni.scaleMode == smSoftware and vnni.blockSize == 32

const arm = ARM_NEON_SDOT_8x8x4
doAssert arm.kind == bkCPU_SIMD and arm.isa == siSDOT
doAssert arm.cType == mdtInt32 and arm.conversionPoint == cpPerBlock

echo "  OK — all 4 atom kinds expressible as data"
