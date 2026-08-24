## Shared benchmark utilities for tattletale/ceramic CPU benchmarks.

import std/[math, strformat, strutils, typetraits]
import workspace/ceramic/src/macros/static_for

# ═══════════════════════════════════════════════════════════════════════════
#  XOR hash — exact bit-level fingerprint
# ═══════════════════════════════════════════════════════════════════════════

proc xorHash*(data: openArray[float32]): uint32 =
  result = 0
  for i in 0 ..< data.len:
    result = result xor cast[uint32](data[i])

# ═══════════════════════════════════════════════════════════════════════════
#  allClose — tolerance-based comparison
# ═══════════════════════════════════════════════════════════════════════════

type CloseResult* = object
  ok*: bool
  maxAbsErr*: float32
  maxRelErr*: float32

proc allClose*(
    a, b: openArray[float32];
    rtol: float32 = 1e-4;
    atol: float32 = 1e-4
  ): CloseResult =
  ## Element-wise comparison with relative and absolute tolerance.
  ## Returns max error and whether all entries are within tolerance.
  doAssert a.len == b.len
  result.maxAbsErr = 0
  result.maxRelErr = 0
  for i in 0 ..< a.len:
    let diff = abs(a[i] - b[i])
    result.maxAbsErr = max(result.maxAbsErr, diff)
    result.maxRelErr = max(result.maxRelErr, diff / max(abs(a[i]), max(abs(b[i]), 1e-12'f32)))
  result.ok = result.maxAbsErr <= atol and result.maxRelErr <= rtol

# ═══════════════════════════════════════════════════════════════════════════
#  Scalar GEMM reference
# ═══════════════════════════════════════════════════════════════════════════

proc gemm_reference*(
    M, N, K: int, alpha: float32,
    A: openArray[float32], rsA, csA: int,
    B: openArray[float32], rsB, csB: int,
    beta: float32, C: var openArray[float32], rsC, csC: int) =
  ## Scalar triple-loop GEMM: `C = beta*C + alpha*A·B` for strided f32
  ## views, the correctness reference for the tuned kernels.
  ##
  ## Expected input:
  ##   - A: M×K view, element strides (rsA, csA)
  ##   - B: K×N view, element strides (rsB, csB)
  ##   - C: M×N view, element strides (rsC, csC), prior values read
  ##
  ## Output: C updated in place to `beta*C + alpha*A·B`. B elements equal
  ## to 0 skip their rank-1 update.
  for j in 0 ..< N:
    for i in 0 ..< M:
      let ci = i * rsC + j * csC
      C[ci] = if beta == 0.0'f32: 0.0'f32
              elif beta != 1.0'f32: C[ci] * beta
              else: C[ci]
  for j in 0 ..< N:
    for k in 0 ..< K:
      let bv = B[k * rsB + j * csB]
      if bv != 0.0'f32:
        for i in 0 ..< M:
          C[i * rsC + j * csC] += alpha * A[i * rsA + k * csA] * bv

# ═══════════════════════════════════════════════════════════════════════════
#  Median
# ═══════════════════════════════════════════════════════════════════════════

proc median*(v: openArray[float64]): float64 =
  let n = v.len
  if n == 0: return 0.0
  if n mod 2 == 1: v[n div 2]
  else: (v[n div 2 - 1] + v[n div 2]) * 0.5

# ═══════════════════════════════════════════════════════════════════════════
#  Theoretical peak (float32, single-core)
# ═══════════════════════════════════════════════════════════════════════════
#
#  Formula: freq(GHz) × vectorWidth × instrPerCycle × FLOP/instr
#  AVX+FMA: 8 floats × 2 FMA/cycle × 2 FLOP/FMA = 32 FLOP/cycle
#  AVX-512: 16 floats × 2 FMA/cycle × 2 FLOP/FMA = 64 FLOP/cycle

type CpuArch* = enum
  archAVX_FMA
  archAVX512

const AnchorFreqs* = [4.0, 4.5, 5.0, 5.5]

proc theoreticalPeak*(arch: CpuArch; freq: float64): float64 =
  let vecWidth = case arch
    of archAVX_FMA:  8.0
    of archAVX512:  16.0
  let instrCycle = 2.0    # 2 FMAs per cycle (Intel: ports 0 & 1)
  let flopInstr  = 2.0    # FMA = 1 mul + 1 add
  freq * vecWidth * instrCycle * flopInstr

proc printPeakTable*() =
  ## Print the theoretical peak reference table.
  echo "  Theoretical peak (float32, single-core):"
  echo "  " & spaces(4) & "4.0 GHz".align(9) & "4.5 GHz".align(9) & "5.0 GHz".align(9) & "5.5 GHz".align(9)
  echo "  " & "-".repeat(46)
  for arch in [archAVX_FMA, archAVX512]:
    let name = if arch == archAVX_FMA: "AVX+FMA" else: "AVX-512"
    echo &"  {name:<8} {int(theoreticalPeak(arch, 4.0)):>5d} {int(theoreticalPeak(arch, 4.5)):>5d} {int(theoreticalPeak(arch, 5.0)):>5d} {int(theoreticalPeak(arch, 5.5)):>5d} GFLOP/s"
  echo ""
  echo &"  Formula: freq × vecWidth × 2 FMA/cycle × 2 FLOP/FMA"
  echo &"    AVX+FMA: 8 floats  — peak = freq × 32"
  echo &"    AVX-512: 16 floats — peak = freq × 64"
  echo ""

# ═══════════════════════════════════════════════════════════════════════════
#  toArray — convert tuple to array
# ═══════════════════════════════════════════════════════════════════════════

func toArray*(t: tuple): auto =
  ## Convert a tuple to array[tupleLen, int].
  const N = tupleLen(typeof(t))
  var a: array[N, int]
  staticFor i, 0, N:
    a[i] = t[i].toIntVal()
  a

# ═══════════════════════════════════════════════════════════════════════════
#  Matrix shape helpers (Laser convention)
# ═══════════════════════════════════════════════════════════════════════════

type MatrixShape* = tuple[M, N: int]

func gemm_out_shape*(a, b: MatrixShape): MatrixShape =
  doAssert a.N == b.M
  result.M = a.M
  result.N = b.N

func gemm_required_ops*(a, b: MatrixShape): int =
  doAssert a.N == b.M
  result = a.M * a.N * b.N * 2   # 1 mul + 1 add per element

func gemm_required_data*(a, b: MatrixShape): int =
  doAssert a.N == b.M
  result = a.M * a.N + b.M * b.N
