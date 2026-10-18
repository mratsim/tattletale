## WGSL: nested device fns with pointer-field structs inline per call site.
##
## Test that a nested callee inlines and its pointer-field accesses resolve
## through the buffers passed at the call, with and without name collisions.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/functions/t_webgpu_ptr_struct_inline.nim

import std/[strutils, unittest]
import workspace/crucible

type AxpyView = object
  p: ptr UncheckedArray[float32]
  n: int

proc innerSwap(y, x: AxpyView) =
  y.p[0] = x.p[0] + 1

proc innerPlain(dst, src: AxpyView) =
  dst.p[0] = src.p[0] + 100

proc outerSwap(y: AxpyView, x: AxpyView) {.global, workgroup: (1, 1, 1).} =
  innerSwap(x, y)

proc outerPlain(y: AxpyView, x: AxpyView) {.global, workgroup: (1, 1, 1).} =
  innerPlain(x, y)

const swapSrc = webgpu:
  proc outerSwap(y: AxpyView, x: AxpyView) {.global, workgroup: (1, 1, 1).} =
    innerSwap(x, y)

const plainSrc = webgpu:
  proc outerPlain(y: AxpyView, x: AxpyView) {.global, workgroup: (1, 1, 1).} =
    innerPlain(x, y)

proc runTest() =
  suite "WebGPU - pointer-field struct params inline per call site":
    test "swap: the callee inlines and binds y.p to the buffer passed as first arg":
      doAssert "fn innerSwap" notin swapSrc,
        "the callee must inline, not stay a real call with display-name wiring:\n" & swapSrc
      doAssert "(&x_p)[0] = ((&y_p)[0] + 1)" in swapSrc,
        "inner's y.p must resolve to x_p (the first call arg) and x.p to y_p:\n" & swapSrc
    test "no collision: dst/src resolve through the call's arguments, not the type-keyed fallback":
      doAssert "fn innerPlain" notin plainSrc,
        "the callee must inline:\n" & plainSrc
      doAssert "(&x_p)[0] = ((&y_p)[0] + 100)" in plainSrc,
        "dst=x must receive from src=y, not from the last-constructed buffer:\n" & plainSrc

runTest()
