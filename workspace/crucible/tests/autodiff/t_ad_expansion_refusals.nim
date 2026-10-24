# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## bwd_diff expansion refusals of the launch model
##
## Launch-model contracts refuse at the `bwd_diff` macro expansion.
## The refusal carries the `bwd_diff:` prefix and names the construct:
##
## - coordinate-builtin reads imply thread parallelism, the subset is single-thread
## - multi-thread workgroup annotations would race the backward's accumulations
## - sibling top-level loops mis-size the scalar-cotangent scratch (v1 is one index loop)
## - a computed loop bound leaves the reduce trip count underivable
##
## A macro `error()` is a compile error, so each case compiles a standalone
## snippet through `nim check` and must fail with the prefix and construct.
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/t_ad_expansion_refusals.nim

import std/[os, osproc, strutils]

proc repoRoot(): string =
  ## This file lives five levels below the repository root.
  result = currentSourcePath()
  for i in 0 ..< 5:
    result = result.parentDir()

proc runCase(name, src: string): tuple[output: string, exitCode: int] =
  ## Compiles one standalone snippet with `nim check` and returns the combined
  ## compiler output. A shared nimcache keeps the crucible sources compiled
  ## once across cases.
  let dir = repoRoot() / "build" / "ad_expansion_refusals"
  createDir(dir)
  let path = dir / ("case_" & name & ".nim")
  writeFile(path, src)
  let cmd = "nim check --hints:off --warnings:off --verbosity:0 " &
            "--maxLoopIterationsVM:100000000 --nimcache:build/nimcache_ad_expansion_refusals " &
            "--path:. " & path
  result = execCmdEx(cmd, options = {poStdErrToStdOut, poUsePath},
                     workingDir = repoRoot())

proc expectRefusal(name, src, construct: string) =
  let (output, code) = runCase(name, src)
  doAssert code != 0,
    "case '" & name & "' must refuse to compile, it compiled cleanly"
  doAssert "bwd_diff:" in output,
    "case '" & name & "' refusal must carry the bwd_diff: prefix, got:\n" & output
  doAssert construct in output,
    "case '" & name & "' refusal must name the construct '" & construct &
    "', got:\n" & output
proc entryBody(params, body, backend: string, pragmas = "global"): string =
  ## Standalone snippet driving the `bwd_diff` macro entry over one kernel:
  ## the kernel typechecks at its definition, then the pair expansion runs
  ## the full ingestion pipeline.
  result = """
import workspace/crucible

proc kernel(""" & params & """) {.""" & pragmas & """.} =
""" & body & """

const adPair = """ & backend & """.bwd_diff(kernel)
"""

expectRefusal("coord_builtin", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  """  let gx = thread_position_in_grid.x
  dst[gx] = a[gx] * 2.0'f32""", "bkMetal"), "gpuDot")

expectRefusal("workgroup_threads", entryBody(
  "dst, a: ptr UncheckedArray[float32], b: ptr UncheckedArray[float32], alpha: float32",
  """  for i in 0 ..< 8:
    dst[i] = alpha * a[i] + b[i]""", "bkMetal", pragmas = "global, workgroup: (4, 4)"), "gpuProc")

expectRefusal("sibling_loops", entryBody(
  "dst, a, b: ptr UncheckedArray[float32], n1, n2: int32, w: float32",
  """  for i in 0 ..< n1:
    dst[i] = a[i]
  for j in 0 ..< n2:
    dst[j] = b[j] * w""", "bkMetal"), "gpuFor")

expectRefusal("binop_bound", entryBody(
  "dst: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], alpha: float32, n: int32",
  "  for i in 0 ..< (2 * n):\n    dst[i] = alpha * a[i]", "bkMetal"), "gpuFor")

echo "[OK] bwd_diff launch-model refusals: 4 cases"
