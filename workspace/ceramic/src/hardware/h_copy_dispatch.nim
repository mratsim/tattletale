## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Copy dispatch, bridging the copy-atom catalog (h_copy_registry.nim) with the hardware copy instructions.

import std/macros
import ./h_copy_registry
import ./h_copy_properties
import workspace/crucible

{.experimental: "dynamicBindSym".}

# ═════════════════════════════════════════════════════════════════════════
#  The async copy atom (the tile-copy chunk of the GEMM pipeline)
# ═════════════════════════════════════════════════════════════════════════

func getCopyAsyncAtom*(T: typedesc): static auto {.inline.} =
  ## The backend's async copy atom for element type `T`
  when ccGetBackend() == ctCuda:
    SM80_CP_ASYNC_CG_16B_ZFILL
  else:
    {.error: "getCopyAsyncAtom: no async copy atom for backend " & $ccGetBackend().}

# ═════════════════════════════════════════════════════════════════════════
#  copy_unpack: instruction emission from a registry entry
# ═════════════════════════════════════════════════════════════════════════

proc copyBody(atomName: string;
              dstView, srcView: NimNode; srcSize: NimNode;
              zfill: bool; chunkBytes: int): NimNode =
  let instr = bindSym(atomName & "_instr").getImpl()[2].strVal
  if instr == "":
    doAssert not zfill,
      "copyIf: the universal copy has no zero-fill fold"
    let i = ident("i")
    result = newTree(nnkForStmt, i,
      newTree(nnkInfix, bindSym".. <", newIntLitNode(0),
        newCall(bindSym"size", dstView)),
      newAssignment(newCall(dstView, i), newCall(srcView, i)))
  else:
    var operandStr = " :: \"r\"(`smemInt`), \"l\"(`gmemPtr`), \"n\"(" &
      $chunkBytes & ")"
    if zfill:
      # the src-size operand is the chunk width, or 0 to zero-fill the chunk
      operandStr = " :: \"r\"(`smemInt`), \"l\"(`gmemPtr`), \"n\"(" &
        $chunkBytes & "), \"r\"(`srcSize`)"
    let asmStr = "\"" & instr & (if zfill: ", %3" else: "") &
      ";\"" & operandStr & " : \"memory\""
    result = newStmtList(
      newLetStmt(ident"srcSize", srcSize),
      newLetStmt(ident"smemInt",
        newCall(bindSym"cvtaGenericToShared", newDotExpr(dstView, ident"data"))),
      newLetStmt(ident"gmemPtr",
        newDotExpr(srcView, ident"data")),
      newTree(nnkAsmStmt, newEmptyNode(), newLit(asmStr)))

macro copyIf*(atom: static CopyAtom;
              dstView, srcView: untyped; pred: untyped): untyped =
  ## Prepare a cp.async copy from global memory (gmem)
  ##   to the per-warp shared memory (smem)
  ## Contract, one predication behavior per atom capability
  ## - issued asynchronously with other prepared copies in the same commit_group,
  ##   waited for with wait_group
  ## - a zero-fill-capable atom with a false predicate zero-fills the chunk
  ##   (the instruction's src-size operand is the chunk width or 0)
  ## - other atoms guard the chunk copy with a runtime if
  let name = $atom
  let zfill = bindSym(name & "_zeroFill").getImpl()[2].strVal == "true"
  let chunkB = bindSym(name & "_vecBytes").getImpl()[2].intVal
  if zfill:
    # the predicate folds into the instruction (srcSize 0 zero-fills the chunk)
    let srcSize = newTree(nnkIfExpr,
      newTree(nnkElifExpr, pred, newIntLitNode(chunkB)),
      newTree(nnkElseExpr, newIntLitNode(0)))
    result = copyBody(name, dstView, srcView, srcSize, zfill, chunkB)
  else:
    # branch predication around the chunk copy
    let body = copyBody(name, dstView, srcView, newIntLitNode(chunkB), zfill, chunkB)
    result = newTree(nnkIfStmt, newTree(nnkElifBranch, pred, body))

# ═════════════════════════════════════════════════════════════════════════
#  commit / wait slots
# ═════════════════════════════════════════════════════════════════════════

macro commit_group*(atom: static CopyAtom): untyped =
  ## Commit the cp.async copies prepared since the previous commit
  let name = $atom
  let kind = bindSym(name & "_kind").getImpl()[2].strVal
  let slot = (if kind == "commit": name
              elif kind == "copy": bindSym(name & "_commitAtom").getImpl()[2].strVal
              else: "")
  if slot == "":
    error("commit_group: the atom must be a copy or commit-kind atom")
  let instr = bindSym(slot & "_instr").getImpl()[2].strVal
  if instr == "":
    result = newEmptyNode()
  else:
    result = newTree(nnkAsmStmt, newEmptyNode(), newLit("\"" & instr & ";\" :: : \"memory\""))

macro wait_group*(atom: static CopyAtom; depth: static int): untyped =
  ## Block until all but N of the recent copy groups are fully copied to shared memory.
  ## The buffering depth is N + 1 stages
  ## - N = 0, single-buffered
  ## - N = 1, double-buffered
  ## - N = 2, triple-buffered
  let name = $atom
  let kind = bindSym(name & "_kind").getImpl()[2].strVal
  let slot = (if kind == "wait": name
              elif kind == "copy": bindSym(name & "_waitAtom").getImpl()[2].strVal
              else: "")
  if slot == "":
    error("wait_group: the atom must be a copy or wait-kind atom")
  let instr = bindSym(slot & "_instr").getImpl()[2].strVal
  if instr == "":
    result = newEmptyNode()
  else:
    let depthCap = bindSym(slot & "_waitDepth").getImpl()[2].intVal
    doAssert depth >= 0 and depth <= depthCap,
      "wait_group: depth " & $depth & " exceeds the wait atom's supported depth " & $depthCap
    result = newTree(nnkAsmStmt, newEmptyNode(),
      newLit("\"" & instr & ";\" :: \"n\"(" & $depth & ") : \"memory\""))
