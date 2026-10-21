# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## P3 test: verification pragma extraction to Lean spec scaffolds
##
## Test that pragma payloads extract into Lean scaffolds with an honest
## proven/unproven verdict:
##
## - correct properties elaborate under warningAsError, recorded proven
## - a wrong property is downgraded to a sorry-backed scaffold, recorded unproven
##   in the manifest with its Lean error
## - wrong modules fail the warningAsError elaboration, out-of-scope payloads are refused loudly
##
## Run:
##   cd workspace/crucible
##   nim c -r --hints:off --warnings:off \
##     -o:build/wip/t_p3_pragmas --nimcache:nimcache/wip poc/lean/t_p3_pragmas.nim

import std/[assertions, json, os, osproc, strutils]
import workspace/crucible/src/crucible_internal_compiler
import lean_pragmas
import nim_to_lean

const leanPocDir = "poc/lean"
const genSpecRelPath = leanPocDir & "/generated/add_spec.lean"
const genWrongPostRelPath = leanPocDir & "/generated/add_spec_wrong_post.lean"
const genWrongInvRelPath = leanPocDir & "/generated/add_spec_wrong_inv.lean"

proc elaborate(relPath: string): tuple[output: string, exitCode: int] =
  ## Elaborates one emitted Lean module through the lake environment,
  ## warnings escalated to errors so a sorry fails
  execCmdEx("lake env lean -DwarningAsError=true " & relPath,
    workingDir = leanPocDir)

proc scaffoldEntry(manifest: JsonNode, kind: string): JsonNode =
  ## Manifest scaffold entry of the given kind
  for e in manifest["scaffolds"].items:
    if e["kind"].getStr() == kind:
      return e
  doAssert false, "no " & kind & " scaffold in the manifest"

proc runTest() =
  block: # correct pragmas: proven manifest, module elaborates under warningAsError
    let k = irVulkanLean:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, precondition: "tid < n", invariant: "tid < n", postcondition: "output tid = a tid + b tid".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]

    let emitted = emitLeanModuleWithSpecs(k.gpu.statements[0], k.specs, genSpecRelPath)
    doAssert "theorem addBody_invariant" in emitted, "invariant scaffold emitted"
    doAssert "theorem addBody_postcondition" in emitted, "postcondition scaffold emitted"
    doAssert "sorry" notin emitted, "no scaffold needed the downgrade"

    let manifest = parseJson(readFile(genSpecRelPath.changeFileExt("manifest.json")))
    doAssert manifest["kernel"].getStr() == "addKernel", "manifest names the kernel"
    doAssert manifest["module"].getStr() == "generated/add_spec.lean"
    let pre = scaffoldEntry(manifest, "precondition")
    doAssert pre["payload"].getStr() == "tid < n"
    doAssert pre["proven"].kind == JNull, "a precondition is assumed, not proved"
    let inv = scaffoldEntry(manifest, "invariant")
    doAssert inv["name"].getStr() == "addBody_invariant"
    doAssert inv["proven"].getBool(), "guard invariant proved by the guard"
    doAssert inv["error"].kind == JNull
    let post = scaffoldEntry(manifest, "postcondition")
    doAssert post["payload"].getStr() == "output tid = a tid + b tid"
    doAssert post["name"].getStr() == "addBody_postcondition"
    doAssert post["proven"].getBool(), "add-shape postcondition proved by simp"

    let (outp, code) = elaborate("generated/add_spec.lean")
    doAssert code == 0, "Lean rejected the proven module:\n" & outp
    doAssert "sorry" notin outp, "no sorry under warningAsError"
    echo "  correct pragmas: proven manifest + warningAsError elaboration: ok"

  block: # wrong postcondition: sorry downgrade, unproven manifest, warningAsError rejection
    let k = irVulkanLean:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, precondition: "tid < n", postcondition: "output tid = a tid + b tid + 1".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]

    discard emitLeanModuleWithSpecs(k.gpu.statements[0], k.specs, genWrongPostRelPath)
    let manifest = parseJson(readFile(genWrongPostRelPath.changeFileExt("manifest.json")))
    let post = scaffoldEntry(manifest, "postcondition")
    doAssert post["proven"].getBool() == false, "wrong postcondition recorded unproven"
    doAssert post["error"].getStr().len > 0, "Lean error excerpt recorded"
    echo "  wrong postcondition manifest entry: ", post.pretty

    let emitted = readFile(genWrongPostRelPath)
    doAssert "sorry" in emitted, "proof downgraded to a sorry-backed scaffold"
    let (_, code) = elaborate("generated/add_spec_wrong_post.lean")
    doAssert code != 0, "the sorry-backed module must fail under warningAsError"
    echo "  wrong postcondition: unproven manifest + warningAsError rejection: ok"

  block: # wrong invariant (not implied by the guard): same downgrade + rejection
    let k = irVulkanLean:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "tid ≤ n".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]

    discard emitLeanModuleWithSpecs(k.gpu.statements[0], k.specs, genWrongInvRelPath)
    let manifest = parseJson(readFile(genWrongInvRelPath.changeFileExt("manifest.json")))
    let inv = scaffoldEntry(manifest, "invariant")
    doAssert inv["proven"].getBool() == false, "wrong invariant recorded unproven"
    doAssert inv["error"].getStr().len > 0, "Lean error excerpt recorded"
    let (_, code) = elaborate("generated/add_spec_wrong_inv.lean")
    doAssert code != 0, "the sorry-backed module must fail under warningAsError"
    echo "  wrong invariant: unproven manifest + warningAsError rejection: ok"

  block: # loud refusals on out-of-scope payloads
    let unparseable = irVulkanLean:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, postcondition: "output tid =".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    doAssertRaises(ValueError):
      discard emitLeanModuleWithSpecs(unparseable.gpu, unparseable.specs, genSpecRelPath)

    let unknownIdent = irVulkanLean:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, postcondition: "bogus tid = a tid".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    doAssertRaises(ValueError):
      discard emitLeanModuleWithSpecs(unknownIdent.gpu, unknownIdent.specs, genSpecRelPath)

    let badPrecondition = irVulkanLean:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, precondition: "n < tid".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    doAssertRaises(ValueError):
      discard emitLeanModuleWithSpecs(badPrecondition.gpu, badPrecondition.specs, genSpecRelPath)

    let bufferInvariant = irVulkanLean:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global, invariant: "a tid < n".} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    doAssertRaises(ValueError):
      discard emitLeanModuleWithSpecs(bufferInvariant.gpu, bufferInvariant.specs, genSpecRelPath)

    let plain = irVulkanLean:
      proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
        let i = thread_position_in_grid.x
        if i < 2:
          output[i] = a[i] + b[i]
    doAssertRaises(ValueError):
      discard emitLeanModuleWithSpecs(plain.gpu.statements[0],
        @[RawSpec(name: "sidecondition", payload: "tid < n")], genSpecRelPath)
    echo "  refusals on out-of-scope payloads: ok"

when isMainModule:
  runTest()
  echo "t_p3_pragmas: all checks passed"
