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
  nodeFields: array[6, string] = fieldNames(Node)
  machineFields: array[2, string] = fieldNames(Machine)

static:
  # The node vocabulary is the corpus-derived ten, in declaration order.
  assert NodeKind.high.ord + 1 == 10, "NodeKind must hold the ten corpus-derived kinds"
  assert $NodeKind.low == "nkVerbatim", "nkVerbatim opens the enum"
  assert $NodeKind.high == "nkMacroDef", "nkMacroDef closes the enum"

  # Node is `{kind, lo, hi, succ, child, alt}` and nothing else. The six names are
  # spelled out explicitly, and the layout proof is arithmetic, five `int32` links
  # and payload fields plus a one-byte tag padded to a four-byte alignment, 24 bytes.
  # Any proc, pointer, `string` or `seq` member would raise `alignof` to 8 and change
  # `sizeof`, so this pair of assertions
  # rules out a proc field and a heap box without naming each forbidden type.
  assert nodeFields == ["kind", "lo", "hi", "succ", "child", "alt"],
      "Node must be exactly {kind, lo, hi, succ, child, alt}"
  assert sizeof(Node) == 24, "Node layout: " & $sizeof(Node)
  assert alignof(Node) == 4, "a pointer-sized member would lift alignment to 8"
  assert sizeof(Node) <= 32, "the arena's per-node budget is 32 bytes"
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

# A `Node` must move by assignment with no reference held anywhere:
#   this is the property that lets
# the arena be one allocation and the artifact be serializable.
static:
  let a = Node(kind: nkEmit, lo: 1'i32, hi: 2'i32, succ: 3'i32, child: 4'i32, alt: 5'i32)
  var arena = @[a, a]
  arena[1] = arena[0]
  assert arena[1].succ == 3'i32
  assert arena.len == 2

doAssert sizeof(Machine) == 32, "Machine layout: " & $sizeof(Machine)

echo "t_shape: Node=", sizeof(Node), "B align=", alignof(Node),
    " Machine=", sizeof(Machine), "B steps=", steps.len
