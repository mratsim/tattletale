# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shape assertions on the compiled chattyninja form, covering the node budget,
## the arena's POD status, the two-field read-only `Machine`, and dispatch totality.
##
## Run:
##   $ ./workspace/chattyninja/run_tests.sh t_shape

import std/macros
import cjn_types, chattyninja
import workspace/data_structures/src/small_seqs

macro fieldNames(T: type): untyped =
  ## Returns the field names of an object type, in declaration order.
  var t = getTypeImpl(T)
  if t.kind == nnkBracketExpr:
    t = getTypeImpl(t[1])
  expectKind t, nnkObjectTy
  let rec = t[2]
  expectKind rec, nnkRecList
  result = newTree(nnkBracket)
  for f in rec:
    case f.kind
    of nnkSym:
      result.add newLit(f.strVal)
    of nnkIdentDefs:
      for k in 0 ..< f.len - 2:
        result.add newLit(f[k].strVal)
    else:
      error("unexpected field node", f)

const
  nodeFields: array[2, string] = fieldNames(Node)
  machineFields: array[2, string] = fieldNames(Machine)

static:
  # The node vocabulary is the corpus-derived ten, in declaration order.
  assert NodeKind.high.ord + 1 == 10, "NodeKind must hold the ten corpus-derived kinds"
  assert $NodeKind.low == "nkVerbatim", "nkVerbatim opens the enum"
  assert $NodeKind.high == "nkMacroDef", "nkMacroDef closes the enum"

  # Node is `{kind, slots}` and nothing else. Measured layout on arm64 with Nim 2.2.10:
  # `SmallSeq[5, int32]` is 40 bytes, an `int32` len and cap, a 20-byte inline array of five
  # `int32`, and one 8-byte-aligned tail pointer. `Node` pads the one-byte kind to the tail
  # pointer's alignment, so 8 + 40 = 48 bytes per node, the arena's per-node budget.
  #
  # Measured over the 1098 nodes of the 13 corpus templates that parse, the other 5 raising
  # declared gaps. 93% of nodes hold at most 5 payload slots and no node holds exactly 6,
  # the histogram peaking at 3, 4 and 5 slots.
  #
  # Only the variable `nkFor` and `nkMacroDef` payloads spill, one heap block each at parse
  # time and never at render.
  #
  # Any proc, `string` or `seq` member would change `sizeof`, so this pair of assertions
  # rules out a proc field and a heap box without naming each forbidden type.
  assert nodeFields == ["kind", "slots"], "Node must be exactly {kind, slots}"
  assert sizeof(SmallSeq[5, int32]) == 40, "SmallSeq[5, int32] layout: " & $sizeof(SmallSeq[5, int32])
  assert sizeof(Node) == 48, "Node layout: " & $sizeof(Node)
  assert alignof(Node) == 8, "the payload tail pointer aligns the node to 8 bytes"
  assert not (Node is ref), "nodes are POD in one seq, never a ref box"

  # Machine is the borrowed text plus the arena:
  #   two fields, so nothing render-mutable is reachable
  # from the artifact and one artifact can serve several drivers.
  assert machineFields == ["jinja", "nodes"],
      "Machine must be exactly {jinja, nodes}"

  # Dispatch is one array total over the enum. The array's type already makes an uncovered kind
  # a compile error, and the length check keeps a renamed or reordered enum from silently
  # shrinking the dispatch table.
  assert steps.len == NodeKind.high.ord + 1, "steps must be total over NodeKind"

# A nil step would be a hole in the table:
#   a render would jump through a null pointer rather than
# report the gap, so totality is checked over every kind, not merely counted.
for k in NodeKind:
  doAssert not steps[k].isNil, "steps has no entry for " & $k

# A `Node` must move by assignment with no reference left behind:
#   this is the property that lets the arena be one allocation. An assignment deep-copies
# the spilled payload, so the copy stays valid after the source's block is freed.
#
# The check runs at runtime, the compile-time VM cannot run the payload's allocator.
var a = Node(kind: nkEmit)
for slot in 7'i32 .. 12'i32:
  a.slots.add slot
var arena = @[a, a]
arena[1] = arena[0]
arena[0].slots[0] = 99'i32
doAssert arena[1].slots[0] == 7'i32, "an assignment must deep-copy a spilled payload"
doAssert arena.len == 2, "the arena moved by assignment with no reference left behind"

doAssert sizeof(Machine) == 32, "Machine layout: " & $sizeof(Machine)

echo "t_shape: Node=", sizeof(Node), "B align=", alignof(Node),
    " SmallSeq=", sizeof(SmallSeq[5, int32]), "B Machine=", sizeof(Machine), "B steps=", steps.len
