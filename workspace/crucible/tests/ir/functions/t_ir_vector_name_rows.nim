## Hand-built CIRv2: vector type-name composition through the wgsl and hlsl rows.
##
## Test that a legal vector prints the row's name form and a lane count outside
## the row refuses at the composition site.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_vector_name_rows.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/wgsl_lang
import workspace/crucible/src/codegen/backends/hlsl_lang
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/callgraph_analysis

static:
  let f32 = tFloat(32)
  let outT = tAptr(caDevice, f32)

  proc vectorKernel(n: int): CirModule =
    let kern = procD("k", @[param("o", outT)], tVoid(), @[
      varD("v", tVec(n, f32), fltLit(1.0, f32)),
      retS(),
    ])
    kern.isKernel = true
    result = moduleD("probeVec" & $n, @[], @[kern])

  var reg = PassRegistry.new()
  reg.registerHlslCallGraphPass()

  # ═══ a 4-wide float vector prints the row's name form ═══
  let legal = vectorKernel(4)
  doAssert legal.runPasses(reg).len == 0, "call-graph analysis refused"
  let wgsl = printWgslModule(legal)
  doAssert wgsl.err.len == 0, "wgsl printer refused: " & wgsl.err
  doAssert wgsl.src.contains("vec4<f32>"),
    "wgsl prints the row's generic-element form, got:\n" & wgsl.src
  let hlsl = printHlslModule(legal)
  doAssert hlsl.err.len == 0, "hlsl printer refused: " & hlsl.err
  doAssert hlsl.src.contains("float4 v"),
    "hlsl prints the row's prefixed form, got:\n" & hlsl.src

  # ═══ a lane count outside the row refuses at the composition site ═══
  let wide = vectorKernel(5)
  doAssert wide.runPasses(reg).len == 0, "call-graph analysis refused"
  let wgslWide = printWgslModule(wide)
  doAssert wgslWide.err.contains("vector of f32x5 has no WGSL type name"),
    "wgsl refuses a lane count outside its row, got: " & wgslWide.err
  let hlslWide = printHlslModule(wide)
  doAssert hlslWide.err.contains("vector of floatx5 has no HLSL type name"),
    "hlsl refuses a lane count outside its row, got: " & hlslWide.err
