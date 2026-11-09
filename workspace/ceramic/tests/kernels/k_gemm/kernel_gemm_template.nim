# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[math, random, times]
import workspace/math
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_compiletime
import workspace/ceramic/src/layout_algebra/ism_coord_strides
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/src/kernels/k_tile_gemm
import workspace/ceramic/src/kernels/k_layout_fillwith_cpu

export int_tuples, int_tuples_streams
export layout_algebra
export tensors
export tile_algebra
export k_tile_gemm

# ═══════════════════════════════════════════════════════════════
#  fp32 host reference
# ═══════════════════════════════════════════════════════════════

func gemm_ref*(refOut: distinct HostTensor, alpha: float32, A, B: distinct HostTensor, beta: float32, C: distinct HostTensor) =
  ## Host reference: refOut = alpha * A * B^T + beta * C, BLAS B^T form
  let
    M = A.layout.shape[0]
    N = B.layout.shape[0]
    K = A.layout.shape[1]
  for m in 0 ..< M:
    for n in 0 ..< N:
      var acc = 0.0'f32
      for k in 0 ..< K:
        acc += to(A[m, k], float32) * to(B[n, k], float32)
      refOut[m, n] = alpha * acc + beta * to(C[m, n], float32)

# ═══════════════════════════════════════════════════════════════
#  Case type
# ═══════════════════════════════════════════════════════════════

type GemmCase* = tuple
  alpha: float32
  A: tuple[shape, strides: (int, int)]
  B: tuple[shape, strides: (int, int)]
  beta: float32
  C: tuple[shape, strides: (int, int)]

const GemmHighlights* = [
  # stride permutations at the atom tile size, (32, 32, 32)
  (alpha: 2.0'f32, A: ((32, 32), (32, 1)), B: ((32, 32), (32, 1)), beta: 1.0'f32, C: ((32, 32), (32, 1))),
  (alpha: 2.0'f32, A: ((32, 32), (1, 32)), B: ((32, 32), (1, 32)), beta: 1.0'f32, C: ((32, 32), (1, 32))),
  (alpha: 1.0'f32, A: ((32, 32), (1, 32)), B: ((32, 32), (32, 1)), beta: 0.0'f32, C: ((32, 32), (1, 32))),
  (alpha: 1.0'f32, A: ((32, 32), (32, 1)), B: ((32, 32), (1, 32)), beta: 2.0'f32, C: ((32, 32), (32, 1))),
  # non-contiguous skips, strided-apart lines and inner skips
  (alpha: 1.0'f32, A: ((48, 40), (121, 3)), B: ((24, 40), (1, 80)), beta: 1.0'f32, C: ((48, 24), (24, 1))),
  (alpha: 2.0'f32, A: ((64, 32), (65, 1)), B: ((48, 32), (2, 96)), beta: 1.0'f32, C: ((64, 48), (48, 1))),
  (alpha: 1.0'f32, A: ((50, 30), (1, 60)), B: ((40, 30), (31, 1)), beta: 1.0'f32, C: ((50, 40), (1, 50))),
  (alpha: 1.0'f32, A: ((33, 33), (2, 67)), B: ((33, 33), (67, 2)), beta: 1.0'f32, C: ((33, 33), (1, 33))),
  (alpha: 1.0'f32, A: ((128, 64), (64, 1)), B: ((128, 64), (1, 128)), beta: 0.0'f32, C: ((128, 128), (128, 1))),
  # ragged odd shapes
  (alpha: 1.0'f32, A: ((17, 23), (23, 1)), B: ((19, 23), (1, 19)), beta: 1.0'f32, C: ((17, 19), (19, 1))),
  (alpha: 1.0'f32, A: ((65, 62), (62, 1)), B: ((63, 62), (63, 1)), beta: 2.0'f32, C: ((65, 63), (63, 1))),
  (alpha: 2.0'f32, A: ((24, 96), (96, 1)), B: ((24, 96), (1, 24)), beta: 1.0'f32, C: ((24, 24), (24, 1))),
  (alpha: 1.0'f32, A: ((40, 96), (1, 96)), B: ((96, 96), (96, 1)), beta: 1.0'f32, C: ((40, 96), (96, 1))),
  # sub-tile shapes, tensors smaller than one mma tile
  (alpha: 1.0'f32, A: ((8, 8), (8, 1)), B: ((8, 8), (8, 1)), beta: 0.0'f32, C: ((8, 8), (8, 1))),
  (alpha: 1.0'f32, A: ((12, 28), (28, 1)), B: ((20, 28), (1, 20)), beta: 1.0'f32, C: ((12, 20), (20, 1))),
  (alpha: 2.0'f32, A: ((2, 16), (16, 1)), B: ((130, 16), (1, 130)), beta: 1.0'f32, C: ((2, 130), (130, 1))),
  # multi-tile shapes: a tile grid that needs grid-stride loops
  (alpha: 2.0'f32, A: ((70, 100), (100, 1)), B: ((50, 100), (1, 50)), beta: 1.0'f32, C: ((70, 50), (50, 1))),
  (alpha: 1.0'f32, A: ((96, 96), (1, 96)), B: ((96, 96), (96, 1)), beta: 0.5'f32, C: ((96, 96), (1, 96))),
  # alpha scaling and beta accumulation
  (alpha: 0.5'f32, A: ((48, 40), (40, 1)), B: ((24, 40), (1, 24)), beta: 2.0'f32, C: ((48, 24), (24, 1))),
  (alpha: 1.0'f32, A: ((17, 23), (23, 1)), B: ((19, 23), (19, 1)), beta: 1.0'f32, C: ((17, 19), (19, 1)))]

proc checkGemm*[E: HwEngine, TAcc, TIn](
    rng: var Rand, engine: E,
    _: typedesc[TAcc], _: typedesc[TIn],
    MT, NT, KT: static int, cs: GemmCase, lang: static string = "spirv") =
  let (aShape, aStrides) = cs.A
  let (bShape, bStrides) = cs.B
  let (cShape, cStrides) = cs.C
  let (M, N, K) = (aShape[0], bShape[0], aShape[1])
  when TIn is float16:
    type Stg = float16
  elif TIn is bfloat16:
    type Stg = bfloat16
  else:
    type Stg = TIn
  var
    aHost = rng.randomHostTensor(Stg, aShape, aStrides)
    bHost = rng.randomHostTensor(Stg, bShape, bStrides)
    cHost = rng.randomHostTensor(float32, cShape, cStrides)
    dHost = makeHostTensor(float32, cShape, cStrides)
    refHost = makeHostTensor(float32, (M, N), (N, 1))
  var dView = dHost.view()
  fillWith_cpu(dView, -12345.0'f32) # every D cell must be written
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    cDev = engine.bridge(cHost)
    dDev = engine.bridge(dHost)
  # the blocks loop over the tile grid, any launch grid completes it
  template launch(langLit: untyped) =
    engine.launchDeviceFn gemm(tiledMma(TAcc, TIn, TIn, MT, NT, KT),
        dDev, cs.alpha, aDev, bDev, cs.beta, cDev) do:
      grid = (rng.rand(1 .. 2 * ceil_div(M, MT)),
              rng.rand(1 .. 2 * ceil_div(N, NT)))
      blk = (32, 1)
      lang = `langLit`
  when lang == "spirv":
    launch("spirv")
  elif lang == "glsl":
    launch("glsl")
  elif lang == "hlsl":
    launch(hlsl)
  else:
    {.error: "checkGemm: lang expects spirv, glsl or hlsl, got " & lang.}
  engine.copyToHostFromDevice(dHost.data.toOpenArray(0, dHost.len - 1), dDev.data)
  gemm_ref(refHost, cs.alpha, aHost, bHost, cs.beta, cHost)
  assertAllClose(dHost.view(), refHost.view(), rtol = 1e-5, abstol = 1e-3,
    msg = "gemm " & $M & "x" & $N & "x" & $K & " strides " & $aStrides & " " &
          $bStrides & " " & $cStrides)

proc runGemmChecks*[E: HwEngine, TAcc, TIn](
    rng: var Rand, engine: E, _: typedesc[TAcc], _: typedesc[TIn],
    MT, NT, KT: static int, cases: openArray[GemmCase], Iters: static int,
    lang: static string = "spirv") =
  ## Random checks over one case list, one dtype pairing.
  ## `lang` selects the ingestion route: `spirv` (words, the default) or `glsl` (glslangValidator).
  for iter in 0 ..< Iters:
    for cs in cases:
      rng.checkGemm(engine, TAcc, TIn, MT, NT, KT, cs, lang)

proc runGemmFuzz*[E: HwEngine, TAcc, TIn](
    rng: var Rand, engine: E, _: typedesc[TAcc], _: typedesc[TIn],
    MT, NT, KT: static int, cases: openArray[GemmCase],
    lang: static string = "spirv") =
  ## Random checks over the generated case list, one operand leg
  for cs in cases:
    rng.checkGemm(engine, TAcc, TIn, MT, NT, KT, cs, lang)

# ═══════════════════════════════════════════════════════════════
#  Case generation
# ═══════════════════════════════════════════════════════════════

func genGemmStridePerms[Sh: tuple](shape: Sh): seq[Sh] =
  func rowmajor(dims: seq[int]): seq[int] =
    result = newSeq[int](dims.len)
    var p = 1
    for i in countdown(dims.len - 1, 0):
      result[i] = p
      p *= dims[i]

  func asTuple(vals: seq[int]): Sh =
    var i = 0
    for f in fields(result):
      f = vals[i]
      i.inc

  var dims: seq[int]
  for f in fields(shape):
    dims.add(f)
  let rm = rowmajor(dims)
  result.add(asTuple(rm))
  var cm: seq[int]
  var p = 1
  for i in 0 ..< dims.len:
    cm.add(p)
    p *= dims[i]
  result.add(asTuple(cm))
  var skip = rm
  skip[0] *= 2
  result.add(asTuple(skip))
  skip = cm
  skip[dims.len - 1] *= 2
  result.add(asTuple(skip))

func genGemmCases*[Sh: tuple](shapes: varargs[Sh]): seq[GemmCase] =
  for shape in shapes:
    let (m, n, k) = (shape[0], shape[1], shape[2])
    for sA in genGemmStridePerms((m, k)):
      for sB in genGemmStridePerms((n, k)):
        for sC in genGemmStridePerms((m, n)):
          result.add((1.0'f32,
            (shape: (m, k), strides: sA),
            (shape: (n, k), strides: sB),
            0.0'f32,
            (shape: (m, n), strides: sC)))

