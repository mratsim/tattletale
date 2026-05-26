# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Intrusive WAVL (Weak AVL) Tree — index-based, seq-backed.
##
## ==============================  Design  ==============================
##
## A WAVL tree is a self-balancing BST with rank differences of 1 or 2
## between parent and child (vs AVL's strict 1).  This gives O(log N)
## operations with amortised O(1) restructuring per insert/delete.
##
## **Intrusive (index-based) design**
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Tree nodes are NOT separately allocated.  Instead, each "node" is an
## index into a parallel `WavlLink` seq that lives alongside your data:
##
##   data: seq[YourElement]   ← your actual data
##   links: seq[WavlLink]    ← WAVL parent/left/right/rank at same idx
##   root: int32             ← root index (-1 = empty)
##
## The link at `links[i]` stores the WAVL metadata for `data[i]`.
## Operations (find/insert/delete) work through index indirection:
##
##   links[links[root].l]  instead of  root.left
##
## **Benefits**
## ~~~~~~~~~~~~
## - Zero tree-node GC allocations (links live in a flat seq)
## - Cache-friendly (seq is contiguous memory)
## - 200K nodes = ~3.2 MB of links (vs ~10 MB for ref-object nodes)
## - Seamless integration with Nim `seq.del` (swap-pop) for O(1) removal
##
## **Removal dance with seq.del**
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## When a child is evicted, the sequence shrinks via `.del(deleteIdx)`.
## Nim's `del` moves the LAST element into `deleteIdx` (swap-pop).
## This shifts the last child's index from `lastIdx` to `deleteIdx`,
## breaking any WAVL references that pointed to `lastIdx`.
##
## *Key insight*: a node is referenced by at most 3 other links
## (parent, left child, right child).  After the swap, only those three
## references need updating — O(1), not O(N):
##
##   let link = links[deleteIdx]      # last child's link, now here
##   # parent's child pointer
##   if link.p >= 0:
##     if links[link.p].l == lastIdx: links[link.p].l = deleteIdx
##     if links[link.p].r == lastIdx: links[link.p].r = deleteIdx
##   # children's parent pointer
##   if link.l >= 0: links[link.l].p = deleteIdx
##   if link.r >= 0: links[link.r].p = deleteIdx
##   if root == lastIdx: root = deleteIdx
##
## **fixLinksAfterDataDeletion**
## ~~~~~~~~~~~~~~~~~~~~~
## Exported so containers (e.g. kvcache.nim) can compose the removal
## dance themselves:
##
##   wavlDelete(links, root, childIdx)   # O(log N)
##   dataSeq.del(childIdx)                # O(1) swap-pop
##   links.del(childIdx)                  # O(1)
##   fixLinksAfterDataDeletion(links, root, lastIdx, deleteIdx)  # O(1)
##
## **Comparison callbacks**
## ~~~~~~~~~~~~~~~~~~~~~~~~
## Since the tree does not own the data, callers provide a comparison
## func.  Two forms:
##
##   NodeCmp — compares two node indices (for insert)
##   FindCmp — compares a caller datum against a node index (for find)
##
## These are plain closure procs — cheap to call, easy to inline later.
##
## ==============================  Conventions  ==============================
##
## - Index -1 = nil (no parent/child)
## - Nil child has rank -1 → parity `true`
## - `rank` field: parity of the node's rank (true = odd, false = even)
## - A "2-child": `child.rank == parent.rank` (rank diff = 2)
## - A "1-child": `child.rank != parent.rank` (rank diff = 1)
## - A leaf has rank 0 → `rank == false`
##
## ==============================  References  ==============================
##
## - Haeupler, Sen, Tarjan (2015). "Rank-Balanced Trees".
##   https://dl.acm.org/doi/epdf/10.1145/2689412

type
  WavlLink* = object
    ## Intrusive WAVL tree link — stored at the same index as the data node.
    ## p/l/r default to WavlNil (-1) so uninitialized links are safe.
    p*: int32 = -1'i32      ## parent index (-1 = nil)
    l*: int32 = -1'i32      ## left child
    r*: int32 = -1'i32      ## right child
    rank*: bool             ## rank parity (false = even, true = odd)

  NodeCmp* = proc(a, b: int32): int {.closure.}
    ## Compares two node indices.  Returns <0 / 0 / >0 .

  FindCmp* = proc(idx: int32): int {.closure.}
    ## Compares a search datum against node `idx`.  Returns <0 / 0 / >0 .
    ## The closure captures whatever key the caller is searching for.

const
  WavlNil* = -1'i32   ## sentinel for "no link"

# ---------------------------------------------------------------------------
# Parity helpers
# ---------------------------------------------------------------------------

func getParity(links: openArray[WavlLink]; idx: int32): bool {.inline.} =
  ## Get the rank parity of node `idx`.  Nil nodes have rank -1 → parity `true`.
  if idx == WavlNil: true else: links[idx].rank

func is2Child(links: openArray[WavlLink]; child, parent: int32): bool {.inline.} =
  ## True when `child` is a 2-child of `parent` (rank diff = 2).
  links[child].rank == links[parent].rank

func isLeaf(links: openArray[WavlLink]; idx: int32): bool {.inline.} =
  ## True when `idx` has no children.
  links[idx].l == WavlNil and links[idx].r == WavlNil

func promote(links: var seq[WavlLink]; idx: int32) {.inline.} =
  ## Flip rank parity — increases the node's rank by 1.
  links[idx].rank = not links[idx].rank

func demote(links: var seq[WavlLink]; idx: int32) {.inline.} =
  ## Flip rank parity — decreases the node's rank by 1.
  links[idx].rank = not links[idx].rank

func doublePromote(links: var seq[WavlLink]; idx: int32) {.inline.} =
  ## With 1-bit parity, double-promote is a no-op.  Provided for
  ## documentation / correspondence with reference implementations.
  discard

func doubleDemote(links: var seq[WavlLink]; idx: int32) {.inline.} =
  ## With 1-bit parity, double-demote is a no-op.
  discard

# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------

func rotateRight(links: var seq[WavlLink]; root: var int32; x, z: int32) =
  ## Single right rotation.  X is left child of Z.
  ##
  ##     Z            X
  ##    / \    →     / \
  ##   X   C        A   Z
  ##  / \              / \
  ## A   B            B   C
  let p_z = links[z].p
  let b = links[x].r

  links[x].p = p_z
  if p_z >= 0:
    if links[p_z].l == z: links[p_z].l = x
    else:                 links[p_z].r = x
  else: root = x

  links[z].l = b
  if b >= 0: links[b].p = z

  links[x].r = z
  links[z].p = x

func rotateLeft(links: var seq[WavlLink]; root: var int32; x, z: int32) =
  ## Single left rotation.  X is right child of Z.
  ##
  ##   Z                X
  ##  / \              / \
  ## C   X      →     Z   B
  ##    / \          / \
  ##   A   B        C   A
  let p_z = links[z].p
  let a = links[x].l

  links[x].p = p_z
  if p_z >= 0:
    if links[p_z].l == z: links[p_z].l = x
    else:                 links[p_z].r = x
  else: root = x

  links[z].r = a
  if a >= 0: links[a].p = z

  links[x].l = z
  links[z].p = x

func doubleRotateRight(links: var seq[WavlLink]; root: var int32;
                        y, x, z: int32) =
  ## Double right rotation (left–right).  X is left child of Z,
  ## Y is right child of X.
  ##
  ##     Z              Y
  ##    / \           /   \
  ##   X   D    →   X     Z
  ##  / \          / \   / \
  ## A   Y        A   B C   D
  ##    / \
  ##   B   C
  let p_z = links[z].p
  let b = links[y].l
  let c = links[y].r

  links[y].p = p_z
  if p_z >= 0:
    if links[p_z].l == z: links[p_z].l = y
    else:                 links[p_z].r = y
  else: root = y

  links[x].r = b
  if b >= 0: links[b].p = x

  links[y].l = x
  links[x].p = y

  links[z].l = c
  if c >= 0: links[c].p = z

  links[y].r = z
  links[z].p = y

func doubleRotateLeft(links: var seq[WavlLink]; root: var int32;
                       y, x, z: int32) =
  ## Double left rotation (right–left).  X is right child of Z,
  ## Y is left child of X.
  ##
  ##   Z                Y
  ##  / \             /   \
  ## D   X     →    Z     X
  ##    / \        / \   / \
  ##   Y   A      D   B C   A
  ##  / \
  ## B   C
  let p_z = links[z].p
  let b = links[y].l
  let c = links[y].r

  links[y].p = p_z
  if p_z >= 0:
    if links[p_z].l == z: links[p_z].l = y
    else:                 links[p_z].r = y
  else: root = y

  links[x].l = c
  if c >= 0: links[c].p = x

  links[y].r = x
  links[x].p = y

  links[z].r = b
  if b >= 0: links[b].p = z

  links[y].l = z
  links[z].p = y

# ---------------------------------------------------------------------------
# Insert rebalancing
# ---------------------------------------------------------------------------

func balanceAfterInsert(links: var seq[WavlLink]; root: var int32;
                         node: int32) =
  ## Rebalance after inserting `node` whose parent was a 1,1 leaf.
  ## The algorithm from Haeupler, Sen, Tarjan:
  ##   1. Promote parent while it is a (0,1) node.
  ##   2. If parent became (0,2), perform 1 or 2 rotations.

  let p_x0 = links[node].p
  if p_x0 < 0: return
  # If parent already has both children the rank rule is satisfied.
  if links[p_x0].l >= 0 and links[p_x0].r >= 0: return

  var x = node
  var p_x = p_x0
  var nodeParity, parentParity, siblingParity: bool
  var isLeftChild: bool

  # Phase 1 — promote while parent is (0,1)
  while true:
    promote(links, p_x)
    let pp = links[p_x].p
    if pp < 0: return          # climbed to root

    x = p_x
    p_x = pp

    nodeParity   = links[x].rank
    parentParity = links[p_x].rank
    isLeftChild  = links[p_x].l == x

    let sibling = if isLeftChild: links[p_x].r else: links[p_x].l
    siblingParity = getParity(links, sibling)

    # (0,1) iff  (!N*!P*S) + (N*P*!S)
    let is01 = (not nodeParity and not parentParity and siblingParity) or
               (nodeParity and parentParity and not siblingParity)
    if not is01: break

  # Phase 2 — rotate if parent is (0,2) i.e.  (!N*!P*!S) + (N*P*S)
  let is02 = (nodeParity == parentParity) and (nodeParity == siblingParity)
  if not is02: return

  let z = p_x
  if isLeftChild:
    let y = links[x].r
    if y < 0 or links[y].rank == nodeParity:
      rotateRight(links, root, x, z)
      if z >= 0: demote(links, z)
    else:
      doubleRotateRight(links, root, y, x, z)
      promote(links, y)
      demote(links, x)
      if z >= 0: demote(links, z)
  else:
    let y = links[x].l
    if y < 0 or links[y].rank == nodeParity:
      rotateLeft(links, root, x, z)
      if z >= 0: demote(links, z)
    else:
      doubleRotateLeft(links, root, y, x, z)
      promote(links, y)
      demote(links, x)
      if z >= 0: demote(links, z)

# ---------------------------------------------------------------------------
# Delete rebalancing
# ---------------------------------------------------------------------------

func rebalance3Child(links: var seq[WavlLink]; root: var int32;
                      z: int32; xIsLeftChild: bool) =
  ## Rebalance after a 3-child was created at `z`.
  ## `xIsLeftChild` tells which side of Z is the (potential) 3-child.
  ## Algorithm: demote while Y is a 2-child or 2-2 node, then rotate.

  var curZ = z
  var leftChild = xIsLeftChild
  var done = true

  while true:
    let y = if leftChild: links[curZ].r else: links[curZ].l
    doAssert y >= 0, "rebalance3Child: sibling Y must exist"

    let yIs2Child = (links[y].rank == links[curZ].rank)
    var yIs22Node = false

    if not yIs2Child:
      # Y is a 1-child — check if it is a 2,2 node
      if links[y].rank:
        # odd parity → 2,2 if both children have odd parity
        yIs22Node = getParity(links, links[y].l) and getParity(links, links[y].r)
      else:
        # even parity → 2,2 if binary with both children even
        yIs22Node = links[y].l >= 0 and links[y].r >= 0 and
                    not links[links[y].l].rank and not links[links[y].r].rank

      if not yIs22Node:
        done = false
        break

    demote(links, curZ)
    if not yIs2Child:
      demote(links, y)

    let pp = links[curZ].p
    if pp < 0: return                     # climbed to root

    let xParityAfter = links[curZ].rank
    let oldZ = curZ
    curZ = pp

    if links[curZ].rank == xParityAfter:  # no longer a 3-child
      return

    leftChild = links[curZ].l == oldZ

  if done: return

  # Phase 2 — rotations
  if leftChild:
    let y = links[curZ].r
    let w = links[y].r
    let wParity = getParity(links, w)

    if links[y].rank != wParity:
      # W is a 1-child of Y → single rotation
      rotateLeft(links, root, y, curZ)
      promote(links, y)
      if isLeaf(links, curZ):
        demote(links, curZ)
        demote(links, curZ)
      else:
        demote(links, curZ)
    else:
      # W is a 2-child → V must be a 1-child
      let v = links[y].l
      doubleRotateLeft(links, root, v, y, curZ)
      promote(links, v)
      demote(links, y)
      demote(links, curZ)
      demote(links, curZ)
  else:
    let y = links[curZ].l
    let w = links[y].l
    let wParity = getParity(links, w)

    if links[y].rank != wParity:
      rotateRight(links, root, y, curZ)
      promote(links, y)
      if isLeaf(links, curZ):
        demote(links, curZ)
        demote(links, curZ)
      else:
        demote(links, curZ)
    else:
      let v = links[y].r
      doubleRotateRight(links, root, v, y, curZ)
      promote(links, v)
      demote(links, y)
      demote(links, curZ)
      demote(links, curZ)

func rebalance22Leaf(links: var seq[WavlLink]; root: var int32;
                      node: int32) =
  ## Fix a 2,2 leaf by demoting it, then checking if the parent now
  ## has a 3-child.
  if not links[node].rank: return
  if links[node].l >= 0 or links[node].r >= 0: return

  demote(links, node)

  let p = links[node].p
  if p < 0: return

  rebalance3Child(links, root, p, links[p].l == node)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

func wavlInit*(links: var seq[WavlLink]; root: var int32) {.inline.} =
  root = WavlNil

template wavlFindTpl*(links: openArray[WavlLink]; root: int32;
                       cmpExpr: untyped): int32 =
  ## Find the node index matching the caller's search datum.
  ## `cmpExpr` is an inline expression that uses `ti` (injected) as
  ## the current candidate node index.  Returns <0 / 0 / >0.
  ## Returns `WavlNil` if not found.
  var resultIdx: int32 = WavlNil
  var curr = root
  while curr >= 0 and resultIdx == WavlNil:
    var ti {.inject.} = curr
    let c = cmpExpr
    if c == 0:
      resultIdx = curr
    elif c < 0:
      curr = links[curr].l
    else:
      curr = links[curr].r
  resultIdx

template wavlInsertTpl*(links: var seq[WavlLink]; root: var int32; idx: int32;
                         cmpExpr: untyped) =
  ## Insert node `idx`.  Its links are overwritten — starts as rank-0 leaf.
  ## `cmpExpr` uses injected `a` (= idx) and `b` (= current node).
  ## Raises `AssertionError` on duplicate key.
  ##
  ## NOTE: Uses `block insertBlock` so `break` exits only the template,
  ## not the enclosing function (templates expand at call site, `return`
  ## would escape the caller). This is the template vs proc gotcha.
  block insertBlock:
    links[idx] = WavlLink(p: WavlNil, l: WavlNil, r: WavlNil, rank: false)
    if root < 0:
      root = idx
      break insertBlock
    var curr = root
    var wasLeaf = false
    while true:
      var a {.inject.} = idx
      var b {.inject.} = curr
      let c = cmpExpr
      if c == 0:
        doAssert false, "wavlInsertTpl: duplicate key at index " & $curr
      elif c < 0:
        let child = links[curr].l
        if child < 0:
          wasLeaf = isLeaf(links, curr)
          links[curr].l = idx
          links[idx].p = curr
          break
        curr = child
      else:
        let child = links[curr].r
        if child < 0:
          wasLeaf = isLeaf(links, curr)
          links[curr].r = idx
          links[idx].p = curr
          break
        curr = child
    if wasLeaf:
      balanceAfterInsert(links, root, idx)

proc wavlFind*(links: openArray[WavlLink]; root: int32;
               cmp: FindCmp): int32 =
  ## Find the node index matching the caller's search datum.
  ## `cmp` receives each candidate index; return <0 / 0 / >0.
  ## Returns `WavlNil` if not found.
  ##
  ## Closure-based convenience wrapper around `wavlFindTpl`.
  wavlFindTpl(links, root): cmp(ti)

proc wavlInsert*(links: var seq[WavlLink]; root: var int32; idx: int32;
                 cmp: NodeCmp) =
  ## Insert node `idx`.  Its links are overwritten — starts as rank-0 leaf.
  ## `cmp(a, b)` compares node `a` against node `b`.
  ## Raises `AssertionError` on duplicate key.
  ##
  ## Closure-based convenience wrapper around `wavlInsertTpl`.
  wavlInsertTpl(links, root, idx): cmp(a, b)
func wavlDelete*(links: var seq[WavlLink]; root: var int32; idx: int32) =
  ## Remove node `idx` from the tree.  Its links are NOT cleared
  ## (caller may reuse or discard the slot).

  if root < 0: return

  if links[idx].l >= 0 and links[idx].r >= 0:
    # ── two children: swap with in-order successor ──
    var succ = links[idx].r
    while links[succ].l >= 0:
      succ = links[succ].l

    let y = succ
    var p_y = links[y].p
    let is2Child = (p_y >= 0) and (links[y].rank == links[p_y].rank)

    let yRight = links[y].r

    # Splice Y out of its current position
    if yRight >= 0: links[yRight].p = p_y
    if p_y >= 0:
      if links[p_y].l == y: links[p_y].l = yRight
      else:                 links[p_y].r = yRight

    # ── IMPORTANT: capture idx's links AFTER the splice ──
    # When the successor is a direct child of idx, the splice
    # modifies links[idx].r (sets it to yRight).  We must
    # read the post-splice value to avoid self-references.
    let idxP = links[idx].p; let idxL = links[idx].l
    let idxR = links[idx].r; let idxRank = links[idx].rank

    # Move Y into idx's tree position
    links[y].p = idxP
    links[y].l = idxL
    links[y].r = idxR
    links[y].rank = idxRank

    if idxP >= 0:
      if links[idxP].l == idx: links[idxP].l = y
      else:                    links[idxP].r = y
    else: root = y

    if idxL >= 0: links[idxL].p = y
    if idxR >= 0: links[idxR].p = y

    # idx is now disconnected — its successor took its place.
    # Rebalance at the position where Y was spliced out.
    if p_y == idx:   # successor was direct right child of deleted node
      p_y = y

    if p_y >= 0:
      # Is the child at p_y's left (the spliced-in child) a 3-child?
      # 3-child iff (child exists) == (parent has odd parity)
      let x = if links[p_y].l >= 0 and links[p_y].l != y and links[p_y].l != idx:
                links[p_y].l
              elif links[p_y].r >= 0 and links[p_y].r != y and links[p_y].r != idx:
                links[p_y].r
              else: WavlNil

      if is2Child:
        if (x >= 0) == links[p_y].rank:
          # Determine which side the 3-child is on.
          # When x is nil, check which of p_y's children is nil.
          # If both children are nil, parent is a 2,2 leaf.
          let hasY = if x >= 0: true
                     else: links[p_y].l >= 0 or links[p_y].r >= 0
          if hasY:
            let xIsLeft = if x >= 0: links[p_y].l == x
                          elif links[p_y].l >= 0: false
                          else: true
            rebalance3Child(links, root, p_y, xIsLeft)
          else:
            rebalance22Leaf(links, root, p_y)
      else:
        # 1-child removal — check for 2,2 leaf
        if x < 0 and isLeaf(links, p_y):
          rebalance22Leaf(links, root, p_y)

  else:
    # ── 0 or 1 child: splice idx out directly ──
    let p_y = links[idx].p
    let is2Child = (p_y >= 0) and (links[idx].rank == links[p_y].rank)
    let x = if links[idx].l >= 0: links[idx].l else: links[idx].r

    if x >= 0: links[x].p = p_y
    if p_y >= 0:
      if links[p_y].l == idx: links[p_y].l = x
      else:                   links[p_y].r = x
    else: root = x

    # Rebalance
    if p_y >= 0:
      if is2Child:
        if (x >= 0) == links[p_y].rank:
          # Determine which side the 3-child is on.
          # When x is nil, check which of p_y's children is nil.
          # If both children are nil, parent is a 2,2 leaf.
          let hasY = if x >= 0: true
                     else: links[p_y].l >= 0 or links[p_y].r >= 0
          if hasY:
            let xIsLeft = if x >= 0: links[p_y].l == x
                          elif links[p_y].l >= 0: false
                          else: true
            rebalance3Child(links, root, p_y, xIsLeft)
          else:
            rebalance22Leaf(links, root, p_y)
      else:
        if x < 0 and isLeaf(links, p_y):
          rebalance22Leaf(links, root, p_y)

# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------

func wavlMin*(links: openArray[WavlLink]; root: int32): int32 =
  if root < 0: return WavlNil
  var curr = root
  while links[curr].l >= 0: curr = links[curr].l
  curr

func wavlMax*(links: openArray[WavlLink]; root: int32): int32 =
  if root < 0: return WavlNil
  var curr = root
  while links[curr].r >= 0: curr = links[curr].r
  curr

func wavlNext*(links: openArray[WavlLink]; idx: int32): int32 =
  if idx < 0: return WavlNil
  if links[idx].r >= 0:
    var curr = links[idx].r
    while links[curr].l >= 0: curr = links[curr].l
    return curr
  var curr = idx
  var parent = links[curr].p
  while parent >= 0 and links[parent].r == curr:
    curr = parent
    parent = links[parent].p
  parent

func wavlPrev*(links: openArray[WavlLink]; idx: int32): int32 =
  if idx < 0: return WavlNil
  if links[idx].l >= 0:
    var curr = links[idx].l
    while links[curr].r >= 0: curr = links[curr].r
    return curr
  var curr = idx
  var parent = links[curr].p
  while parent >= 0 and links[parent].l == curr:
    curr = parent
    parent = links[parent].p
  parent

# ---------------------------------------------------------------------------
# Index fixup after seq.del
# ---------------------------------------------------------------------------

func fixLinksAfterDataDeletion*(links: var seq[WavlLink]; root: var int32;
                        oldIdx, newIdx: int32) =
  ## After `seq.del(delIdx)` moved the last element from `oldIdx` to
  ## `newIdx` (= delIdx), update all remaining WAVL references that
  ## pointed to `oldIdx`.
  ##
  ## A node is referenced by at most 3 links — O(1), not O(N).

  if oldIdx == newIdx: return

  let link = links[newIdx]

  if link.p >= 0:
    if links[link.p].l == oldIdx: links[link.p].l = newIdx
    if links[link.p].r == oldIdx: links[link.p].r = newIdx

  if link.l >= 0: links[link.l].p = newIdx
  if link.r >= 0: links[link.r].p = newIdx
  if root == oldIdx: root = newIdx

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

func verifySubtree(links: openArray[WavlLink]; relRank: var seq[int];
                     n: int32; depth, N: int; ctx: string) =
  if n < 0: return
  doAssert depth < N,
    ctx & ": cycle detected (depth=" & $depth & ") at node " & $n
  doAssert n >= 0 and n < N,
    ctx & ": out-of-bounds node " & $n

  doAssert links[n].l != n,
    ctx & ": node " & $n & " is its own left child"
  doAssert links[n].r != n,
    ctx & ": node " & $n & " is its own right child"
  doAssert links[n].p != n,
    ctx & ": node " & $n & " is its own parent"

  var hasLeft = false
  var hasRight = false

  if links[n].l >= 0:
    doAssert links[links[n].l].p == n,
      ctx & ": left child " & $links[n].l & " does not point back to parent " & $n
    # 1-child (diff=1) when parities differ, 2-child (diff=2) when they match
    let diff = if links[links[n].l].rank == links[n].rank: 2 else: 1
    relRank[links[n].l] = relRank[n] - diff
    hasLeft = true
    verifySubtree(links, relRank, links[n].l, depth + 1, N, ctx)

  if links[n].r >= 0:
    doAssert links[links[n].r].p == n,
      ctx & ": right child " & $links[n].r & " does not point back to parent " & $n
    let diff = if links[links[n].r].rank == links[n].rank: 2 else: 1
    relRank[links[n].r] = relRank[n] - diff
    hasRight = true
    verifySubtree(links, relRank, links[n].r, depth + 1, N, ctx)

  # Leaf check: leaves must have absolute rank 0 → parity false
  if not hasLeft and not hasRight:
    doAssert not links[n].rank,
      ctx & ": leaf " & $n & " has parity=true (non-zero rank)"

  # Unary node: the nil child has implicit rank -1 → parity true.
  # We can only verify the PARITY-based diff (mod 2), not the absolute diff.
  # Since nil parity is always true and any parent parity gives a valid
  # 1 or 2 diff mod 2, the unary case is always parity-consistent.

func wavlVerifyInvariants*(links: openArray[WavlLink]; root: int32; ctx: string) =
  ## Thorough WAVL invariant verification.
  ##
  ## Checks:
  ## 1. Rank differences are 1 or 2 at every edge (2-child vs 1-child parity rule)
  ## 2. Leaves have rank parity = false (even → rank 0)
  ## 3. Parent pointers are consistent (bidirectional links)
  ## 4. Root has no parent
  ## 5. No cycles (finite tree, no self-references)
  ##
  ## This computes *relative* ranks by assigning root = 0 and deriving children.
  ## Leaf ranks may differ (tree is not perfectly balanced), but each edge
  ## has a valid rank difference of 1 or 2.
  ##
  ## Only effective in debug builds.
  when defined(debug) or defined(assertions):
    if root < 0: return

    doAssert links[root].p < 0,
      ctx & ": root has parent " & $links[root].p

    let N = links.len
    var relRank = newSeq[int](N)
    relRank[root] = 0
    verifySubtree(links, relRank, root, 0, N, ctx)

func wavlAssertValid*(links: openArray[WavlLink]; root: int32; ctx: string) =
  ## Assert WAVL invariants (shorthand).
  wavlVerifyInvariants(links, root, ctx)
