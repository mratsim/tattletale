## Metal: `{.emit.}` raw MSL injection through the GPU DSL, end to end.
##
## Covered:
## - the argument-form matrix: single string, comma form, backticks,
##   triple-quoted multiline
## - the not-silently-dropped regression for both emit spellings
## - the insertResult exclusion
## - generic-instantiation cloning
## - `#define`-style payload byte-exactness
## - a multi-emit pragma rendering every item in order
## - a non-emit statement pragma mapping to gpuDiscard
##
## Emits are supported inside proc bodies and render at the statement position.
## Top-level emits (outside any proc body) are out of scope.
## Every kernel runs through `engine.run()` and the outputs are asserted,
## never print-only.
##
## The emit is self-terminating: the printer appends no `;`, so the injected text
## must carry its own terminator where MSL requires one.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_emit.nim

import std/[strutils, unittest]
import workspace/crucible

# ── single string literal, injected as raw MSL ──────────────────────────────

const t1Msl = metal:
  proc t1Kernel(C: ptr UncheckedArray[uint32];
                A: ptr UncheckedArray[uint32];
                B: ptr UncheckedArray[uint32]) {.global.} =
    {.emit: "C[0] = A[0] + B[0];".}

# ── comma form, one interpolated ident (the `A` param) ──────────────────────

const t2Msl = metal:
  proc t2Kernel(C: ptr UncheckedArray[uint32];
                A: ptr UncheckedArray[uint32]) {.global.} =
    {.emit: ["C[0] = ", A, "[0] + 1;"].}

# ── comma form, interleaved ident + call + arithmetic ───────────────────────

const t3Msl = metal:
  proc doubleIt(v: uint32): uint32 =
    v * 2

  proc t3Kernel(C: ptr UncheckedArray[uint32];
                A: ptr UncheckedArray[uint32]) {.global.} =
    let off: uint32 = 2
    {.emit: ["C[0] = ", doubleIt(A[0]), " + ", off, ";"].}

# ── backtick form on a local variable ───────────────────────────────────────

const t4Msl = metal:
  proc t4Kernel(C: ptr UncheckedArray[uint32];
                A: ptr UncheckedArray[uint32]) {.global.} =
    let tmp: uint32 = A[0] + 1
    {.emit: "C[0] = `tmp` + 1;".}

# ── triple-quoted multiline string ──────────────────────────────────────────

const t5Msl = metal:
  proc t5Kernel(C: ptr UncheckedArray[uint32];
                A: ptr UncheckedArray[uint32];
                B: ptr UncheckedArray[uint32]) {.global.} =
    {.emit: """C[0] = A[0] + B[0];
C[1] = A[1] + B[1];""".}

# ── not silently dropped — the emit must override the DSL write ─────────────

const t6Msl = metal:
  proc t6Kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    C[0] = 0
    {.emit: "C[0] = 42;".}

# ── backtick form on a kernel parameter ─────────────────────────────────────

const t7Msl = metal:
  proc t7Kernel(C: ptr UncheckedArray[uint32];
                x: ptr UncheckedArray[uint32]) {.global.} =
    {.emit: "C[0] = `x`[0] + 1;".}

# ── comma form on a kernel parameter ────────────────────────────────────────

const t8Msl = metal:
  proc t8Kernel(C: ptr UncheckedArray[uint32];
                x: ptr UncheckedArray[uint32]) {.global.} =
    {.emit: ["C[0] = ", x, "[0] + 1;"].}

# ── insertResult exclusion — a non-void device fn with only a trailing emit
#    must not become `result = <text>` ─────────────────────────────────────────

const t10Msl = metal:
  proc emit42(): uint32 =
    {.emit: "return 42;".}

  proc t10Kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    C[0] = emit42()

# ── generic device fn (pointer param) with an emit — exercises the clone path
#    through generic instantiation ────────────────────────────────────────────

const t11Msl = metal:
  proc emitInto[T](dst: ptr UncheckedArray[T]; v: T) =
    {.emit: ["*", dst, " = ", v, ";"].}

  proc t11Kernel(C: ptr UncheckedArray[uint32];
                 A: ptr UncheckedArray[uint32]) {.global.} =
    emitInto(C, A[0])

# ── call-form emit `{.emit("...").}` is not silently dropped ────────────────

const t12Msl = metal:
  proc t12Kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    C[0] = 0
    {.emit("C[0] = 42;").}

# ── a #define-style directive must pass through byte-exact. A `;` appended by the printer
#    would corrupt the macro (used below) ──────────────────────────────────────

const t14Msl = metal:
  proc t14Kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    {.emit: "#define MSL_EMIT_TILE 32".}
    {.emit: "C[0] = MSL_EMIT_TILE + 1;".}

# ── duplicate emit items in one pragma — every item renders, in order ────────

const t15Msl = metal:
  proc t15Kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    C[0] = 0
    {.emit: "C[0] = 1;", emit: "C[0] = 2;".}

# ── non-emit statement pragma inside a kernel stays a gpuDiscard ──────────────

const t16Msl = metal:
  proc t16Kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    {.warning: "this statement pragma must be discarded, not emitted".}
    C[0] = 42

# ── Host code ────────────────────────────────────────────────────────────────

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - {.emit.} raw MSL injection":

    test "single string literal emits the sum":
      var engine = bkMetal.init()
      engine.ingest(t1Msl)
      var a: array[1, uint32] = [40'u32]
      var b: array[1, uint32] = [2'u32]
      var res: array[1, uint32]
      engine.run("t1Kernel", res, (a, b))
      check res[0] == 42

    test "comma form with one interpolated ident":
      var engine = bkMetal.init()
      engine.ingest(t2Msl)
      var a: array[1, uint32] = [41'u32]
      var res: array[1, uint32]
      engine.run("t2Kernel", res, (a,))
      check res[0] == 42

    test "comma form with ident + call + arithmetic":
      var engine = bkMetal.init()
      engine.ingest(t3Msl)
      var a: array[1, uint32] = [20'u32]
      var res: array[1, uint32]
      engine.run("t3Kernel", res, (a,))
      check res[0] == 42

    test "backtick form on a local variable":
      var engine = bkMetal.init()
      engine.ingest(t4Msl)
      var a: array[1, uint32] = [40'u32]
      var res: array[1, uint32]
      engine.run("t4Kernel", res, (a,))
      check res[0] == 42

    test "triple-quoted multiline string":
      var engine = bkMetal.init()
      engine.ingest(t5Msl)
      var a: array[2, uint32] = [40'u32, 10'u32]
      var b: array[2, uint32] = [2'u32, 3'u32]
      var res: array[2, uint32]
      engine.run("t5Kernel", res, (a, b))
      check res[0] == 42
      check res[1] == 13

    test "emit is not silently dropped (must override the DSL write)":
      var engine = bkMetal.init()
      engine.ingest(t6Msl)
      var res: array[1, uint32]
      engine.run("t6Kernel", res, ())
      check res[0] == 42

    test "backtick form on a kernel parameter":
      var engine = bkMetal.init()
      engine.ingest(t7Msl)
      var x: array[1, uint32] = [41'u32]
      var res: array[1, uint32]
      engine.run("t7Kernel", res, (x,))
      check res[0] == 42

    test "comma form on a kernel parameter":
      var engine = bkMetal.init()
      engine.ingest(t8Msl)
      var x: array[1, uint32] = [41'u32]
      var res: array[1, uint32]
      engine.run("t8Kernel", res, (x,))
      check res[0] == 42

    test "trailing emit in a non-void device fn survives insertResult":
      var engine = bkMetal.init()
      engine.ingest(t10Msl)
      var res: array[1, uint32]
      engine.run("t10Kernel", res, ())
      check res[0] == 42

    test "emit survives generic instantiation (clone path)":
      var engine = bkMetal.init()
      engine.ingest(t11Msl)
      var a: array[1, uint32] = [42'u32]
      var res: array[1, uint32]
      engine.run("t11Kernel", res, (a,))
      check res[0] == 42

    test "call-form emit is not silently dropped (must override the DSL write)":
      var engine = bkMetal.init()
      engine.ingest(t12Msl)
      var res: array[1, uint32]
      engine.run("t12Kernel", res, ())
      check res[0] == 42

    test "#define-style payload is byte-exact (no `;` appended)":
      var engine = bkMetal.init()
      engine.ingest(t14Msl)
      let msl = engine.getArtifact()
      check "#define MSL_EMIT_TILE 32" in msl
      check "#define MSL_EMIT_TILE 32;" notin msl
      var res: array[1, uint32]
      engine.run("t14Kernel", res, ())
      check res[0] == 33

    test "every emit item in one pragma renders, in order, on separate lines":
      # One gpuEmit per item: the generated MSL must not concatenate the
      # two statements onto one line (a line-oriented payload would corrupt).
      doAssert "C[0] = 1;C[0] = 2;" notin t15Msl,
        "multi-item emit rendered on a single line:\n" & t15Msl
      doAssert "C[0] = 1;" in t15Msl and "C[0] = 2;" in t15Msl,
        "multi-item emit dropped an item:\n" & t15Msl
      var engine = bkMetal.init()
      engine.ingest(t15Msl)
      var res: array[1, uint32]
      engine.run("t15Kernel", res, ())
      check res[0] == 2

    test "non-emit statement pragma stays a gpuDiscard (kernel compiles and runs)":
      var engine = bkMetal.init()
      engine.ingest(t16Msl)
      var res: array[1, uint32]
      engine.run("t16Kernel", res, ())
      check res[0] == 42

when isMainModule:
  runTest()
