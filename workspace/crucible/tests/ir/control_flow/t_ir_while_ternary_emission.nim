## Hand-built CIRv2: while-condition and ternary emission across the text targets.
##
## Test that a conditioned loop prints while (cond), a ternary prints the conditional-value form, a nil loop keeps the conditionless shape.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/control_flow/t_ir_while_ternary_emission.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/ir/cirv2_pretty
import workspace/crucible/src/codegen/backends/cuda_lang
import workspace/crucible/src/codegen/backends/msl_lang
import workspace/crucible/src/codegen/backends/opencl_lang
import workspace/crucible/src/codegen/backends/glsl_lang
import workspace/crucible/src/codegen/backends/hlsl_lang
import workspace/crucible/src/codegen/backends/wgsl_lang
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/callgraph_analysis

static:
  let u32 = tUint(32)
  let outT = tAptr(caDevice, u32)

  # kernel: output[i] = (i < 1 ? 100 : i) while i < limit, limit is a local
  let condKern = procD("condKernel", @[param("output", outT)], tVoid(), @[
    varD("i", u32, intLit(0, u32)),
    varD("limit", u32, intLit(3, u32)),
    whileS(ltE(sym("i", u32), sym("limit", u32)), @[
      asgnS(patE(sym("output", outT), sym("i", u32)),
        ternaryE(u32, ltE(sym("i", u32), intLit(1, u32)),
          intLit(100, u32), sym("i", u32))),
      asgnS(sym("i", u32),
        addE(u32, sym("i", u32), intLit(1, u32))),
    ]),
    retS(),
  ])
  condKern.isKernel = true

  # kernel: the conditionless loop keeps the label-free form
  let plainKern = procD("plainKernel", @[param("output", outT)], tVoid(), @[
    varD("i", u32, intLit(0, u32)),
    loopS(@[
      asgnS(patE(sym("output", outT), sym("i", u32)), sym("i", u32)),
      asgnS(sym("i", u32),
        addE(u32, sym("i", u32), intLit(1, u32))),
    ]),
    retS(),
  ])
  plainKern.isKernel = true

  # kernel: the loop condition itself is a ternary, while ((i < 2) ? (i < 3) : false)
  let ternCondKern = procD("ternCondKernel", @[param("output", outT)], tVoid(), @[
    varD("i", u32, intLit(0, u32)),
    whileS(ternaryE(tBool(),
      ltE(sym("i", u32), intLit(2, u32)),
      ltE(sym("i", u32), intLit(3, u32)),
      falseE()), @[
      asgnS(sym("i", u32), addE(u32, sym("i", u32), intLit(1, u32))),
    ]),
    retS(),
  ])
  ternCondKern.isKernel = true

  # kernel: {.unroll.} on a conditioned loop prints its directive in front of the while head
  let unrollKern = procD("unrollKernel", @[param("output", outT)], tVoid(), @[
    varD("i", u32, intLit(0, u32)),
    varD("limit", u32, intLit(3, u32)),
    whileS(ltE(sym("i", u32), sym("limit", u32)), @[
      asgnS(sym("i", u32), addE(u32, sym("i", u32), intLit(1, u32))),
    ], true),
    retS(),
  ])
  unrollKern.isKernel = true

  # HLSL takes one kernel per module, so the loops live in separate modules
  let m = moduleD("probe", @[], @[condKern])
  let plainM = moduleD("probePlain", @[], @[plainKern])
  let ternCondM = moduleD("probeTernCond", @[], @[ternCondKern])
  let unrollM = moduleD("probeUnroll", @[], @[unrollKern])

  # HLSL classifies read-write buffer parameters from the stored call-graph record, so the analysis must run before printing
  var reg = PassRegistry.new()
  reg.registerHlslCallGraphPass()
  let perr = m.runPasses(reg)
  doAssert perr.len == 0, "call-graph analysis refused: " & perr
  # plainM gets its own analysis so the hlsl print classifies the output
  # buffer read-write from the stored record
  let perrPlain = plainM.runPasses(reg)
  doAssert perrPlain.len == 0, "call-graph analysis refused: " & perrPlain
  let perrUnroll = unrollM.runPasses(reg)
  doAssert perrUnroll.len == 0, "call-graph analysis refused: " & perrUnroll

  # ═══ the debug renderer shows the new nodes ═══
  let pretty = renderModule(m)
  doAssert pretty.contains("(loop while (lt (u +32) i limit)"),
    "the conditioned loop renders its condition, got:\n" & pretty
  doAssert pretty.contains("(tern (u +32) (lt (u +32) i 1) 100 i)"),
    "the ternary renders cond then else, got:\n" & pretty
  doAssert renderModule(plainM).contains("(loop\n"),
    "the conditionless loop keeps its rendering"

  # ═══ every text target emits while and its conditional-value form ═══
  # Every condition prints in the target's while head.
  # Every ternary prints the target's conditional-value form, WGSL as select(else, then, cond).
  # printExpr parenthesizes comparisons, so a comparison condition in a head is doubly parenthesized like the if head's condition.
  type Printed = tuple[name, src, err: string]
  var targets: seq[Printed]
  let cudaP = printCudaModule(m)
  let mslP = printMslModule(m)
  let openclP = printOpenclModule(m)
  let glslP = printGlslModule(m)
  let hlslP = printHlslModule(m)
  let wgslP = printWgslModule(m)
  targets.add (name: "cuda", src: cudaP.src, err: cudaP.err)
  targets.add (name: "msl", src: mslP.src, err: mslP.err)
  targets.add (name: "opencl", src: openclP.src, err: openclP.err)
  targets.add (name: "glsl", src: glslP.src, err: glslP.err)
  targets.add (name: "hlsl", src: hlslP.src, err: hlslP.err)
  targets.add (name: "wgsl", src: wgslP.src, err: wgslP.err)
  for t in targets:
    doAssert t.err.len == 0,
      t.name & " printer refused: " & t.err
    doAssert t.src.contains("while ((i < limit))"),
      t.name & " must emit the loop condition, got:\n" & t.src
  for t in targets[0 ..< 4]:
    doAssert t.src.contains("((i < 1U) ? 100U : i)"),
      t.name & " must emit the ternary, got:\n" & t.src
  doAssert targets[4].src.contains("((i < 1u) ? 100u : i)"),
    "hlsl must emit the ternary, got:\n" & targets[4].src
  doAssert targets[5].src.contains("select(i, 100u, (i < 1u))"),
    "wgsl must emit the ternary as select(else, then, cond), got:\n" &
      targets[5].src

  # ═══ the conditionless loop keeps the label-free form ═══
  doAssert printCudaModule(plainM).src.contains("for (;;) {"),
    "cuda keeps the conditionless loop form"
  doAssert printMslModule(plainM).src.contains("for (;;) {"),
    "msl keeps the conditionless loop form"
  doAssert printOpenclModule(plainM).src.contains("for (;;) {"),
    "opencl keeps the conditionless loop form"
  doAssert printGlslModule(plainM).src.contains("for (;;) {"),
    "glsl keeps the conditionless loop form"
  doAssert printHlslModule(plainM).src.contains("for (;;)\n"),
    "hlsl keeps the conditionless loop head without a brace, the brace is on its own line"
  doAssert printWgslModule(plainM).src.contains("loop {"),
    "wgsl keeps the conditionless loop form"

  # ═══ a ternary in the loop condition ═══
  doAssert renderModule(ternCondM).contains("(loop while (tern (bool)"),
    "the debug renderer shows the ternary loop condition"
  let ternCuda = printCudaModule(ternCondM)
  doAssert ternCuda.err.len == 0, "cuda printer refused: " & ternCuda.err
  doAssert ternCuda.src.contains("while (((i < 2U) ? (i < 3U) : false))"),
    "cuda emits the ternary loop condition in the while head, got:\n" & ternCuda.src
  let ternWgsl = printWgslModule(ternCondM)
  doAssert ternWgsl.err.len == 0, "wgsl printer refused: " & ternWgsl.err
  doAssert ternWgsl.src.contains("while (select(false, (i < 3u), (i < 2u)))"),
    "wgsl emits the ternary loop condition as select(else, then, cond), got:\n" & ternWgsl.src

  # ═══ {.unroll.} on a conditioned loop ═══
  # Each target prints its own directive in front of the while head.
  type UnrollPin = tuple[name, marker: string]
  let pins: seq[UnrollPin] = @[
    (name: "cuda", marker: "#pragma unroll\n"),
    (name: "msl", marker: "#pragma clang loop unroll(full)\n"),
    (name: "opencl", marker: "#pragma unroll\n"),
    (name: "glsl", marker: "[[unroll]]\n"),
    (name: "hlsl", marker: "[unroll]\n"),
    (name: "wgsl", marker: "// {.unroll.} requested but not honored on WebGPU\n"),
  ]
  for pin in pins:
    let printed = case pin.name
      of "cuda": printCudaModule(unrollM)
      of "msl": printMslModule(unrollM)
      of "opencl": printOpenclModule(unrollM)
      of "glsl": printGlslModule(unrollM)
      of "hlsl": printHlslModule(unrollM)
      else: printWgslModule(unrollM)
    doAssert printed.err.len == 0, pin.name & " printer refused: " & printed.err
    doAssert printed.src.contains(pin.marker),
      pin.name & " must emit its unroll directive in front of the while head, got:\n" &
        printed.src
