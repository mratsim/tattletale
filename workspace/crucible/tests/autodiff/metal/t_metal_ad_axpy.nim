# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## bwd_diff pair artifact on Metal and GLSL: the axpy kernel pair at expansion
##
## Test that `cgMetal.bwd_diff` and `cgGlsl.bwd_diff` derive the axpy pair under
## the backward signature rule, and that a declared backward validates against it:
##
## - derived pair: kernel name, workgroup, metadata, workspace, reduce kernel
## - GLSL: per-kernel push-constant blocks, a 1x1x1 reduce workgroup
## - declared pair: signature validation passes, no reduce kernels, no workspaces
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/metal/t_metal_ad_axpy.nim

import std/strutils
import workspace/crucible
import workspace/crucible/src/codegen/autodiff_types

proc axpy(dst, a, b: ptr UncheckedArray[float32], alpha: float32) {.global, workgroup: (1, 1).} =
  for i in 0 ..< 8:
    dst[i] = alpha * a[i] + b[i]

proc axpy_bwd(d_dst, d_a, d_b, d_alpha: ptr UncheckedArray[float32],
             dst, a, b: ptr UncheckedArray[float32], alpha: float32) {.global, workgroup: (1, 1).} =
  # Hand-written declared counterpart of the derived axpy backward, one signature,
  # scalar gradient accumulated in-kernel, no scratch and no reduce kernel.
  for i in 0 ..< 8:
    d_a[i] += d_dst[i] * alpha
    d_b[i] += d_dst[i]
    d_alpha[0] += d_dst[i] * a[i]

proc coordIndexTP(dst, a: ptr UncheckedArray[float32]) {.global, workgroup: (8, 1).} =
  # Thread gx owns slice `j = gx + i * 8`, a strided map the thread-local write analysis approves.
  let gx = thread_position_in_threadgroup.x
  for i in 0 ..< 8:
    let j = gx + i * 8
    dst[j] = a[j] * 2.0'f32

const axpyKernels = cgMetal.bwd_diff(axpy)
const axpyKernelsGlsl = cgGlsl.bwd_diff(axpy)
const axpyDeclaredKernels = cgMetal.bwd_diff(axpy, axpy_bwd)
const coordIndexTPKernels = cgMetal.bwd_diff(coordIndexTP)

static:
  block derivedArtifact:
    doAssert axpyKernels.kernelName == "axpy"
    doAssert axpyKernels.workgroup.x == 1 and axpyKernels.workgroup.y == 1 and
      axpyKernels.workgroup.z == 1, "the pair bakes the (1,1,1) launch geometry"
    doAssert axpyKernels.fwdSource.contains("kernel void axpy("),
      "the forward prints as a standalone Metal kernel, got:\n" & axpyKernels.fwdSource
    doAssert axpyKernels.bwdSource.contains("kernel void axpy_bwd("),
      "the backward prints as a standalone Metal kernel, got:\n" & axpyKernels.bwdSource

  block derivedMetadata:
    doAssert axpyKernels.params.len == 4, "one record per forward parameter"
    doAssert axpyKernels.params[0].name == "dst" and
      axpyKernels.params[0].kind == pkResidual and
      axpyKernels.params[0].gradKind == gkBuffer and
      axpyKernels.params[0].gradName == "d_dst" and
      axpyKernels.params[0].extent == xeWrapperField,
      "the written buffer records as a residual with a same-shape gradient"
    doAssert axpyKernels.params[1].name == "a" and
      axpyKernels.params[1].kind == pkBuffer and
      axpyKernels.params[1].gradKind == gkBuffer and
      axpyKernels.params[1].gradName == "d_a"
    doAssert axpyKernels.params[2].name == "b" and
      axpyKernels.params[2].kind == pkBuffer and
      axpyKernels.params[2].gradKind == gkBuffer and
      axpyKernels.params[2].gradName == "d_b"
    doAssert axpyKernels.params[3].name == "alpha" and
      axpyKernels.params[3].kind == pkScalar and
      axpyKernels.params[3].gradKind == gkBuffer1 and
      axpyKernels.params[3].gradName == "d_alpha"

  block derivedScalarGradient:
    doAssert axpyKernels.scalarWorkspaces.len == 1,
      "one workspace per scalar gradient"
    let ws = axpyKernels.scalarWorkspaces[0]
    doAssert ws.gradName == "d_alpha" and ws.workspaceName == "d_alpha_partials"
    doAssert ws.tripCount == 8 and ws.tripCountParam == "" and
      not ws.tripCount64 and ws.threadCount == 1,
      "the static trip count comes off the loop bound"
    doAssert axpyKernels.reduceSources.len == 1
    doAssert axpyKernels.reduceSources[0].name == "axpy_bwd_d_alpha_reduce"
    doAssert axpyKernels.reduceSources[0].source.contains(
      "kernel void axpy_bwd_d_alpha_reduce("),
      "the reduce kernel prints standalone, got:\n" & axpyKernels.reduceSources[0].source
    doAssert axpyKernels.bwdSource.contains("d_alpha_partials"),
      "the backward's scalar gradient accumulates into its partials workspace"

  block derivedBackwardParams:
    doAssert axpyKernels.bwdKernelParams ==
      @["d_dst", "d_a", "d_b", "a", "alpha", "d_alpha_partials"],
      "the backward's call-site parameters in printed order, coordinate " &
      "builtins excluded, got: " & $axpyKernels.bwdKernelParams

  block glslArtifact:
    doAssert axpyKernelsGlsl.kernelName == "axpy"
    doAssert axpyKernelsGlsl.fwdSource.contains("void axpy()"),
      "the forward prints as a standalone GLSL compute shader, got:\n" &
      axpyKernelsGlsl.fwdSource
    doAssert axpyKernelsGlsl.bwdSource.contains("void axpy_bwd()")
    # per-kernel push-constant shape: each singleton declares its own block,
    # and the multi-kernel shared-block validation never sees the pair
    doAssert axpyKernelsGlsl.fwdSource.contains("uniform KernelParams")
    doAssert axpyKernelsGlsl.bwdSource.contains("uniform KernelParams")
    doAssert axpyKernelsGlsl.bwdSource.contains("Ref_float d_alpha_partials;"),
      "the backward's workspace is a parameter of its own push-constant block"
    doAssert axpyKernelsGlsl.reduceSources[0].source.contains("local_size_x = 1"),
      "the reduce kernel bakes a single-thread workgroup"

  block threadParallelArtifact:
    doAssert coordIndexTPKernels.workgroup.x == 8 and
      coordIndexTPKernels.workgroup.y == 1 and
      coordIndexTPKernels.workgroup.z == 1,
      "the multi-thread annotation survives, every write is proven thread-local"
    doAssert coordIndexTPKernels.bwdSource.contains("kernel void coordIndexTP_bwd("),
      "the backward derives with the forward's thread mapping"
    doAssert coordIndexTPKernels.bwdSource.contains("d_a"),
      "the backward scatters the gradient through the strided index"

  block declaredArtifact:
    doAssert axpyDeclaredKernels.kernelName == "axpy"
    doAssert axpyDeclaredKernels.bwdSource.contains("kernel void axpy_bwd("),
      "the declared backward prints under the pair's backward name"
    doAssert axpyDeclaredKernels.params.len == 4, "one record per forward parameter"
    doAssert axpyDeclaredKernels.params[0].name == "dst" and
      axpyDeclaredKernels.params[0].kind == pkResidual and
      axpyDeclaredKernels.params[0].gradKind == gkBuffer and
      axpyDeclaredKernels.params[0].gradName == "d_dst"
    doAssert axpyDeclaredKernels.params[3].name == "alpha" and
      axpyDeclaredKernels.params[3].gradKind == gkBuffer1 and
      axpyDeclaredKernels.params[3].gradName == "d_alpha",
      "the hand-written kernel writes the 1-element scalar gradient itself"
    doAssert axpyDeclaredKernels.scalarWorkspaces.len == 0,
      "the declared pair has no reduce scratch"
    doAssert axpyDeclaredKernels.reduceSources.len == 0,
      "the declared pair generates no reduce kernels"
    doAssert axpyDeclaredKernels.bwdKernelParams ==
      @["d_dst", "d_a", "d_b", "d_alpha", "dst", "a", "b", "alpha"],
      "the declared backward's mirror signature in printed order, got: " &
      $axpyDeclaredKernels.bwdKernelParams

when isMainModule:
  # Every check above ran at expansion, since the artifact is a compile-time value
  discard axpyKernels
