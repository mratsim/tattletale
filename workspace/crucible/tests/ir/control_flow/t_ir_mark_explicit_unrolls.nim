## markExplicitUnrolls: `{.unroll.}` comments become fUnroll flags
## Test that the filterPragmas pass sets fUnroll on loops carrying the `{.unroll.}` pragma
##
## Run:
##   cd tattletale && ##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/ir/control_flow/t_ir_mark_explicit_unrolls.nim

import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_normalizations {.all.}
import workspace/crucible/tests/ir/ir_testutils

proc ident(name: string): GpuAst =
  GpuAst(kind: gpuIdent, symbol: newSymbol(name))

proc litU32(v: string): GpuAst =
  GpuAst(kind: gpuLit, lType: GpuType(kind: gtUint32), lValue: v)

proc loopStmt(varName: string, statements: varargs[GpuAst]): GpuAst =
  GpuAst(kind: gpuFor,
         fVar: ident(varName),
         fStart: litU32("0"), fEnd: litU32("4"),
         fStep: litU32("1"),
         fBody: GpuAst(kind: gpuBlock, statements: @statements),
         fRangeKind: rkExclusive)

proc comment(text: string): GpuAst =
  GpuAst(kind: gpuComment, comment: text)

proc noopAssign(varName: string): GpuAst =
  GpuAst(kind: gpuAssign,
         aLeft: ident(varName), aOp: "=", aRight: litU32("1"))

static:
  # a loop whose body opens with `{.unroll.}`: marked, comment consumed
  let marked = loopStmt("i", comment("{.unroll.}"), noopAssign("x"))
  let unmarked = loopStmt("j", noopAssign("y"))
  let stray = comment("{.line.}")
  let fn = GpuAst(kind: gpuProc,
                  pName: ident("f"),
                  pRetType: GpuType(kind: gtVoid),
                  pBody: GpuAst(kind: gpuBlock,
                                statements: @[marked, unmarked, stray]))
  let outFn = registerFilterPragmas.runPass(fn)

  let loops = outFn.pBody.statements
  doAssert loops.len == 3, "the pass consumes only the unroll comment, got " & $loops.len
  doAssert loops[0].kind == gpuFor and loops[0].fUnroll,
    "the loop carrying {.unroll.} is marked fUnroll"
  doAssert loops[0].fBody.statements.len == 1 and
           loops[0].fBody.statements[0].kind == gpuAssign,
    "the unroll comment is consumed, the body statement survives"
  doAssert loops[1].kind == gpuFor and not loops[1].fUnroll,
    "an unmarked loop keeps fUnroll false"
  doAssert loops[1].fBody.statements.len == 1,
    "an unmarked loop body is untouched"
  doAssert loops[2].kind == gpuComment and loops[2].comment == "{.line.}",
    "a statement pragma with no GPU meaning stays a comment"

  # a pragma in a nested loop's body marks the inner loop only
  let outer = loopStmt("m",
    loopStmt("n", comment("{.unroll.}"), noopAssign("z")))
  let fn2 = GpuAst(kind: gpuProc,
                   pName: ident("g"),
                   pRetType: GpuType(kind: gtVoid),
                   pBody: GpuAst(kind: gpuBlock, statements: @[outer]))
  let outFn2 = registerFilterPragmas.runPass(fn2)
  let outerLoop = outFn2.pBody.statements[0]
  doAssert outerLoop.kind == gpuFor and not outerLoop.fUnroll,
    "the outer loop is not marked by the inner body's pragma"
  let innerLoop = outerLoop.fBody.statements[0]
  doAssert innerLoop.kind == gpuFor and innerLoop.fUnroll,
    "the inner loop carrying the pragma is marked"
  doAssert innerLoop.fBody.statements.len == 1,
    "the inner pragma is consumed"

echo "  OK - t_ir_mark_explicit_unrolls"
