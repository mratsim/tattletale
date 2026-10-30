## WGSL: `{.exportc.}` entry-point name override at macroexpansion.
##
## Test that `{.exportc: "my_custom_entry".}` renames the emitted WGSL
## `@compute` entry to `fn my_custom_entry`. A bare `{.exportc.}` kernel keeps
## its proc name.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/compiletime/t_webgpu_exportc_entry.nim

import std/strutils
import workspace/crucible

const renamedWgsl = webgpu:
  proc renamedKernel(o, a: ptr UncheckedArray[uint32]) {.global, exportc: "my_custom_entry".} =
    o[0] = a[0] + 1

const bareWgsl = webgpu:
  proc bareKernel(o, a: ptr UncheckedArray[uint32]) {.global, exportc.} =
    o[0] = a[0] + 2

static:
  # compile-time signals: a rename regression fails compilation here
  doAssert "fn my_custom_entry(" in renamedWgsl
  doAssert "fn renamedKernel(" notin renamedWgsl
  doAssert "fn bareKernel(" in bareWgsl
  doAssert "fn my_custom_entry(" notin bareWgsl
