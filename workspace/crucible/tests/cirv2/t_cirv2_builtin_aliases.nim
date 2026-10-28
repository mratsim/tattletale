# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Builtin alias canonicalization: any column of the public contract table resolves to the canonical MSL name in the IR.
##
## The test pipeline runs lowering plus the legalization and normalization stages, then renders the module:
##
## - the CUDA kernel (`threadIdx`, ...) and the GLSL kernel (`gl_GlobalInvocationID`, ...) render under their canonical MSL names
## - a helper proc reading an alias name splices into its caller: the alias pass runs before the inline splice
## - the kernel-only-reads check in the splice pass must see canonical names
## - a name outside the contract table is undeclared, Nim sem refuses it
##
## Run:
##   nim c --hints:off --warnings:off \
##     --nimcache:nimcache/tests/cirv2 --outdir:build/tests/cirv2 \
##     workspace/crucible/tests/cirv2/t_cirv2_builtin_aliases.nim

import std/[macros, strutils]
import
  workspace/crucible/src/builtins/builtins_pragmas,
  workspace/crucible/src/crucible_gpu_builtins,
  workspace/crucible/src/codegen/ir/cirv2_lang,
  workspace/crucible/src/codegen/ir/cirv2_pretty,
  workspace/crucible/src/codegen/frontend/nim_to_crucible,
  workspace/crucible/src/codegen/passes/pass_datatypes,
  workspace/crucible/src/codegen/passes/passes_legalizations,
  workspace/crucible/src/codegen/passes/passes_normalizations

macro traceAfterPassesImpl(body: typed): string =
  ## Args:
  ## - `body`: the kernel body, lowered through the full test pipeline
  ##
  ## Returns the module render as a compile-time string, a refusal anywhere in the chain is a compile-time error.
  let (m, err) = tryLowerCirv2(body)
  if err.len > 0:
    error(err)
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  result = newLit(renderModule(m))

macro traceAfterPasses*(body: untyped): string =
  ## Args:
  ## - `body`: the kernel body to lower
  ##
  ## Returns the module render after the legalization and normalization stages.
  ##
  ## Example:
  ##   const trace = traceAfterPasses:
  ##     proc k(o: ptr UncheckedArray[uint32]) {.global.} =
  ##       o[0] = 1'u32
  result = newCall(bindSym"traceAfterPassesImpl", body)

macro aliasSpellingRefuses*(body: untyped): bool =
  ## Args:
  ## - `body`: the kernel body under refusal test
  ##
  ## Returns true when `body` fails to lower through the test pipeline.
  ## The macro forwards `body` to `compiles`, a colon call cannot sit
  ## inside `compiles`, so the negation lives here.
  result = newCall(bindSym"not",
    newCall(bindSym"compiles", newCall(bindSym"traceAfterPassesImpl", body)))

proc collectBuiltinKinds(e: CirExpr, entries: var seq[string]) =
  ## Collects one `name|coord|synchro|reduction|atomics` line per builtin
  ## symbol under `e`, the alias pass's kind assignment made visible.
  if e.isNil:
    return
  if e.kind == eSym and (e.coordBuiltin != cbkNone or e.synchroBuiltin != sbkNone or
      e.reductionBuiltin != rbkNone or e.atomicsBuiltin != abkNone):
    entries.add e.name & "|" & $e.coordBuiltin & "|" & $e.synchroBuiltin & "|" &
      $e.reductionBuiltin & "|" & $e.atomicsBuiltin
    return
  for ch in getExprChildren(e):
    collectBuiltinKinds(ch, entries)

proc collectBuiltinKinds(ss: seq[CirStmt], entries: var seq[string]) =
  for s in ss:
    case s.kind
    of sVar, sConst: collectBuiltinKinds(s.init, entries)
    of sAsgn:
      collectBuiltinKinds(s.lhs, entries)
      collectBuiltinKinds(s.rhs, entries)
    of sIte:
      collectBuiltinKinds(s.cond, entries)
      collectBuiltinKinds(s.thenBody, entries)
      collectBuiltinKinds(s.elseBody, entries)
    of sLoop: collectBuiltinKinds(s.loopBody, entries)
    of sScope: collectBuiltinKinds(s.scopeBody, entries)
    of sAssume: collectBuiltinKinds(s.assumeCond, entries)
    of sRet: collectBuiltinKinds(s.retVal, entries)
    of sDiscard: collectBuiltinKinds(s.discardVal, entries)
    of sCall:
      collectBuiltinKinds(s.callCallee, entries)
      for a in s.callArgs:
        collectBuiltinKinds(a, entries)
    else: discard

macro kindReportImpl(body: typed): string =
  ## Args:
  ## - `body`: the kernel body, lowered through the full test pipeline
  ##
  ## Returns one `name|coord|synchro|reduction|atomics` line per builtin
  ## symbol after the pass chain, a refusal anywhere is a compile-time error.
  let (m, err) = tryLowerCirv2(body)
  if err.len > 0:
    error(err)
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error(perr)
  var entries: seq[string]
  for fn in m.procs:
    collectBuiltinKinds(fn.body, entries)
  result = newLit(entries.join("\n"))

macro kindReport*(body: untyped): string =
  ## Args:
  ## - `body`: the kernel body to lower
  ##
  ## Returns the builtin-kind report after the legalization and normalization stages,
  ## one `name|coord|synchro|reduction|atomics` line per builtin symbol.
  ##
  ## Example:
  ##   const report = kindReport:
  ##     proc k(y: ptr UncheckedArray[float32]) {.global.} =
  ##       y[threadIdx.x] = 1.0'f32
  result = newCall(bindSym"kindReportImpl", body)

proc cudaKernel(y: ptr UncheckedArray[float32]) {.global.} =
  let gid = threadIdx.x + blockIdx.x * blockDim.x
  let wg = gridDim.x
  y[gid] = float32(gid) + float32(wg)

const cudaTrace = traceAfterPasses:
  cudaKernel

static:
  # every CUDA name resolves to its canonical MSL name
  doAssert "thread_position_in_threadgroup" in cudaTrace
  doAssert "threadgroup_position_in_grid" in cudaTrace
  doAssert "threads_per_threadgroup" in cudaTrace
  doAssert "threadgroups_per_grid" in cudaTrace
  doAssert "threadIdx" notin cudaTrace
  doAssert "blockIdx" notin cudaTrace
  doAssert "blockDim" notin cudaTrace
  doAssert "gridDim" notin cudaTrace

const cudaKinds = kindReport:
  cudaKernel

static:
  # alias pass assigns the typed kinds, the markers stay on the syms
  doAssert "thread_position_in_threadgroup|cbkThreadPositionInThreadgroup|sbkNone|rbkNone|abkNone" in cudaKinds
  doAssert "threadgroup_position_in_grid|cbkThreadgroupPositionInGrid|sbkNone|rbkNone|abkNone" in cudaKinds
  doAssert "threads_per_threadgroup|cbkThreadsPerThreadgroup|sbkNone|rbkNone|abkNone" in cudaKinds
  doAssert "threadgroups_per_grid|cbkThreadgroupsPerGrid|sbkNone|rbkNone|abkNone" in cudaKinds

proc glslKernel(y: ptr UncheckedArray[float32]) {.global.} =
  let gid = gl_GlobalInvocationID.x
  let li = gl_LocalInvocationIndex
  y[gid] = float32(gid + li)

const glslTrace = traceAfterPasses:
  glslKernel

static:
  # GLSL names resolve to the canonical vector name and the canonical flat index
  doAssert "thread_position_in_grid" in glslTrace
  doAssert "thread_index_in_threadgroup" in glslTrace
  doAssert "gl_GlobalInvocationID" notin glslTrace
  doAssert "gl_LocalInvocationIndex" notin glslTrace

const glslKinds = kindReport:
  glslKernel

static:
  doAssert "thread_position_in_grid|cbkThreadPositionInGrid|sbkNone|rbkNone|abkNone" in glslKinds
  doAssert "thread_index_in_threadgroup|cbkThreadIndexInThreadgroup|sbkNone|rbkNone|abkNone" in glslKinds

proc readTid(): int = threadIdx.x

proc helperKernel(y: ptr UncheckedArray[int32]) {.global.} =
  y[0] = int32(readTid())

const helperTrace = traceAfterPasses:
  helperKernel

const helperKinds = kindReport:
  helperKernel

static:
  # alias pass runs before the inline splice, the kernel-only-reads check sees the canonical name, the helper proc splices into the kernel
  doAssert "thread_position_in_threadgroup" in helperTrace
  # kernel-only-reads check keys on the kind recorded for the spliced read
  doAssert "thread_position_in_threadgroup|cbkThreadPositionInThreadgroup|sbkNone|rbkNone|abkNone" in helperKinds
  doAssert "(call readTid_" notin helperTrace
  doAssert "(proc readTid" notin helperTrace
  doAssert "threadIdx" notin helperTrace

static:
  # a name outside the contract table is undeclared, Nim sem refuses it
  # (bound through a let: a colon call under `doAssert` types the kernel body outside the compiles guard)
  let unknownRefused = aliasSpellingRefuses:
    proc badKernel(y: ptr UncheckedArray[float32]) {.global.} =
      y[0] = float32(vk_GlobalInvocationID.x)
  doAssert unknownRefused
