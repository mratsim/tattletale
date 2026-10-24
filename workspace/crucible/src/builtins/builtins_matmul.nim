# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Cooperative-matrix matmul builtins (Vulkan KHR, `GL_KHR_cooperative_matrix`).
##
## One fragment is a per-subgroup `coopmat<T, gl_ScopeSubgroup, M, N, Use>` object,
## the component distribution is driver-defined and inaccessible.
##
## - data enters and leaves only through the builtins and the fragment
##   operators
## - staging, packing, and threadmaps are bypassed
## - supported component types: float32, float16, bfloat16, accumulating
##   in float32
## - requires SPIR-V 1.3 (`--target-env vulkan1.1`, wired in vulkan_compile)

import ./builtins_pragmas

type
  CoopmatUse* {.builtin.} = enum
    ## Operand role of a fragment, the GLSL `gl_MatrixUse*` template parameter.
    cmA
    cmB
    cmAcc

  coopmatA*[T, M, N: static int] {.builtin.} = object
    ## (M, N) fragment of the A operand, `gl_MatrixUseA`.
  coopmatB*[T, M, N: static int] {.builtin.} = object
    ## (M, N) fragment of the B operand, `gl_MatrixUseB`.
  coopmatAcc*[T, M, N: static int] {.builtin.} = object
    ## (M, N) accumulator fragment, `gl_MatrixUseAccumulator`.

const
  cmLayoutRowMajor* = 0
    ## GLSL `gl_CooperativeMatrixLayoutRowMajor`: element (r, c) at `buf[element + r*stride + c]`
  cmLayoutColumnMajor* = 1
    ## GLSL `gl_CooperativeMatrixLayoutColumnMajor`: element (r, c) at `buf[element + c*stride + r]`

# ═══════════════════════════════════════════════════════════════════════
#  Builtins
# ═══════════════════════════════════════════════════════════════════════

func coopMatLoadA*[T](
      M, N: static int, src: ptr UncheckedArray[T],
      element, stride: int, matrixLayout: int
    ): coopmatA[T, M, N] {.builtin.} = discard
  ## Loads an (M, N) A fragment from memory.
  ##
  ## Lowered at the assignment site:
  ## - the GLSL builtin is void with an out param, the loaded fragment
  ##   sits in the assignment's lhs
  ## - the shape travels as static ints (dropped in the IR), the fragment
  ##   type instantiates through the return type The layout is a runtime `int`,
  ## `cmLayoutRowMajor`/`cmLayoutColumnMajor`.

func coopMatLoadB*[T](
      M, N: static int, src: ptr UncheckedArray[T],
      element, stride: int, matrixLayout: int
    ): coopmatB[T, M, N] {.builtin.} = discard
  ## Loads an (M, N) B fragment from memory, same lowering as the A overload.

func coopMatLoadAcc*[T](
      M, N: static int, src: ptr UncheckedArray[T],
      element, stride: int, matrixLayout: int
    ): coopmatAcc[T, M, N] {.builtin.} = discard
  ## Loads an (M, N) accumulator fragment, same lowering as the A overload.

proc coopMatStore*[T, M, N: static int](
      src: coopmatAcc[T, M, N], dst: ptr UncheckedArray[T],
      element, stride: int, matrixLayout: int) {.builtin.} = discard
  ## Stores an accumulator fragment to memory.
  ##
  ## GLSL: `coopMatStore(src, dst.x, element, stride, matrixLayout)`.

func coopMatMulAdd*[TA, TB, TC, M, N, K: static int](
      a: coopmatA[TA, M, K], b: coopmatB[TB, K, N], c: coopmatAcc[TC, M, N]
    ): coopmatAcc[TC, M, N] {.builtin.} = discard
  ## D = A·B + C on opaque fragments, the only arithmetic a fragment supports.
  ## The accumulator component type is independent of the operand types
  ## (f16/bf16 operands accumulate in f32).
  ##
  ## GLSL: `coopMatMulAdd(a, b, c)` with the same argument order.

func coopMatFill*[T](M, N: static int, v: T): coopmatAcc[T, M, N] {.builtin.} = discard
  ## Splat-constructs an (M, N) accumulator fragment, every component set to `v`.
  ##
  ## - the shape travels as static ints (dropped in the IR)
  ## - the fragment type instantiates through the return type
  ##
  ## GLSL: `coopmat<T, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(v)`.

func coopMatScale*[T, M, N: static int](
      m: coopmatAcc[T, M, N], s: T): coopmatAcc[T, M, N] {.builtin.} = discard
  ## s·m, the GLSL scalar·matrix fragment operator
  ## (glslang lowers to OpMatrixTimesScalar). The alpha leg of the epilogue.

func coopMatAdd*[T, M, N: static int](
      a, b: coopmatAcc[T, M, N]): coopmatAcc[T, M, N] {.builtin.} = discard
  ## a + b, the GLSL matrix+matrix fragment operator
  ## (glslang lowers to OpFAdd). The beta leg of the epilogue.
