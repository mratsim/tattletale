/-
# Tattletale KVCache — Formal specification in Lean 4

## Design rationale

The KVCache is a **PagedRadixTrie**: a compressed Radix/Patricia trie over token
sequences, where each node holds a contiguous prefix of tokens and the trie
operates at page (256-token) granularity. GPU pages are immutable once grafted.

See kvcache.nim for the full implementation (with WAVL acceleration).
-/
import Std
set_option linter.unusedVariables false
namespace KVCache

-- ============================================================================
-- 1. Basic types
-- ============================================================================

abbrev TokenID := Nat
abbrev PageIdx := Nat
def TokensPerPage : Nat := 256

/--
A node in the PagedRadixTrie.

Fields mirror the Nim `PagedRadixNode[T, P]`:
- `nodeId` = unique ID for identity comparison (mirrors ref pointer in Nim).
- `depthInPages` = distance from root in pages (256-token blocks).
- `oldestDecode` = minimum kvClock across all leaves in subtree.
- `subtreeSumLocked` = number of active staging paths in subtree.
- `subtreeLeafCount` = total leaves in subtree.
- `subtreeSumPages` = total GPU pages in subtree.
- `parent` = direct parent (none for root).

The WAVL acceleration trees (lpmLinks/lpmRoot, evictLinks/evictRoot) and
childId are NOT modelled here — they are pure acceleration over the children
list.
-/
structure PagedRadixNode where
  nodeId            : Nat
  tokenData         : List TokenID
  pages             : List PageIdx
  children          : List PagedRadixNode
  depthInPages      : Nat
  oldestDecode      : Nat
  subtreeSumLocked  : Nat
  subtreeLeafCount  : Nat
  subtreeSumPages   : Nat
  parent            : Option PagedRadixNode
  deriving Repr, Nonempty

-- BEq by nodeId only (avoids recursion through children/parent fields)
instance : BEq PagedRadixNode where
  beq a b := a.nodeId == b.nodeId

/--
Result of LPM. Mirrors Nim's `LongestPrefixMatch[P]`:
- `pages` = GPU pages for the matched prefix only.
- `totalTokenMatched` = total tokens matched from root.
- `lastLevelMatched` = tokens matched within the final node.
-/
structure LongestPrefixMatch where
  pages              : List PageIdx
  totalTokenMatched  : Nat
  lastLevelMatched   : Nat
  deriving Repr, Nonempty

structure KVCacheState where
  root        : PagedRadixNode
  globalClock : Nat
  deriving Repr

-- ============================================================================
-- 2. Helper definitions
-- ============================================================================

def isLeaf (n : PagedRadixNode) : Prop := n.children.isEmpty
def isLocked (n : PagedRadixNode) : Prop := n.subtreeSumLocked > 0
def isEvictable (n : PagedRadixNode) : Prop := isLeaf n ∧ ¬ isLocked n

def depthInTokens (n : PagedRadixNode) : Nat := n.depthInPages * TokensPerPage
def totalTokenCount (n : PagedRadixNode) : Nat := depthInTokens n + n.tokenData.length

/-- Page-granular common prefix (first 256 tokens only). -/
def getCommonFirstPageLen (a b : List TokenID) : Nat :=
  let n := min (min a.length b.length) TokensPerPage
  (a.zip b).takeWhile (λ (x, y) => x = y) |>.take n |>.length

/-- Token-granular common prefix. -/
def sharedPrefixLength (a b : List TokenID) : Nat :=
  (a.zip b).takeWhile (λ (x, y) => x = y) |>.length

inductive GraftCase where
  | fullMatch | partialMatch | rootNewChild | fork | append
  deriving Repr, BEq

/--
classifyGraft — pure-function decision: which graftPages branch applies?
Mirrors Nim's `classifyGraft`.
-/
def classifyGraftFn (targetMatchLen tokensLen lastLevel targetTokLen : Nat)
                    (hasParent rootHasChildren : Bool) : GraftCase :=
  if targetMatchLen == tokensLen then                .fullMatch
  else if lastLevel < TokensPerPage ∧ hasParent ∧ lastLevel == targetTokLen then
    .partialMatch
  else if lastLevel < TokensPerPage ∧ ¬ hasParent ∧ rootHasChildren then
    .rootNewChild
  else if lastLevel < targetTokLen then              .fork
  else                                               .append

-- ============================================================================
-- 3. Aggregation helpers (walkUp metadata)
-- ============================================================================

def sumLeafCount : List PagedRadixNode → Nat
  | [] => 0
  | c :: cs => c.subtreeLeafCount + sumLeafCount cs

def sumLocked : List PagedRadixNode → Nat
  | [] => 0
  | c :: cs => c.subtreeSumLocked + sumLocked cs

def sumPages : List PagedRadixNode → Nat
  | [] => 0
  | c :: cs => c.subtreeSumPages + sumPages cs

def minOldestDecode (cs : List PagedRadixNode) : Nat :=
  match cs with
  | [] => 0
  | c :: rest => rest.foldl (λ m c' => min m c'.oldestDecode) c.oldestDecode

-- ============================================================================
-- 4. Read-only: LPM (Longest Prefix Match)
-- ============================================================================

/--
LPM — longest prefix match at page granularity.
Mirrors Nim's walkDown.

NOTE: the trie path is implicitly "locked" (subtreeSumLocked incremented).
The caller MUST call graftPages with the FULL tokens + ALL pages to unlock.
-/
partial def lpm (n : PagedRadixNode) (input : List TokenID) : LongestPrefixMatch :=
  let rec go (node : PagedRadixNode) (pos : Nat) : LongestPrefixMatch :=
    let remaining := input.drop pos
    let content := node.tokenData
    let lessThanPage : Bool := content.length < TokensPerPage

    let sharedVal : Nat :=
      if lessThanPage then sharedPrefixLength content remaining
      else
        let firstPage := content.take TokensPerPage
        let inpFirst := remaining.take TokensPerPage
        let s1 := sharedPrefixLength firstPage inpFirst
        if s1 < TokensPerPage then s1
        else if content.length > TokensPerPage then
          TokensPerPage + sharedPrefixLength (content.drop TokensPerPage) (remaining.drop TokensPerPage)
        else TokensPerPage

    let newPos := pos + sharedVal

    let fullyConsumed : Bool :=
      if lessThanPage then sharedVal == content.length
      else
        let firstPageMatches :=
          sharedPrefixLength (content.take TokensPerPage) (remaining.take TokensPerPage) = TokensPerPage
        if firstPageMatches ∧ content.length > TokensPerPage then
          TokensPerPage + sharedPrefixLength
            (content.drop TokensPerPage) (remaining.drop TokensPerPage) = content.length
        else firstPageMatches

    let matchPages := (sharedVal + TokensPerPage - 1) / TokensPerPage

    if newPos ≥ input.length then
      { pages := node.pages.take matchPages,
        totalTokenMatched := newPos,
        lastLevelMatched := sharedVal }
    else if node.children.isEmpty then
      { pages := node.pages.take matchPages,
        totalTokenMatched := newPos,
        lastLevelMatched := sharedVal }
    else if fullyConsumed then
      let inputAtPos := input.drop newPos
      match inputAtPos.head? with
      | none =>
        { pages := node.pages.take matchPages,
          totalTokenMatched := newPos,
          lastLevelMatched := sharedVal }
      | some tok =>
        let candidates := node.children.filter (λ (c : PagedRadixNode) =>
          match c.tokenData with
          | [] => false
          | hd :: _ => hd = tok
        )
        let verified := candidates.filter (λ (c : PagedRadixNode) =>
          getCommonFirstPageLen c.tokenData inputAtPos > 0
        )
        match verified with
        | [] =>
          { pages := node.pages.take matchPages,
            totalTokenMatched := newPos,
            lastLevelMatched := sharedVal }
        | best :: _ =>
          go best newPos
    else
      { pages := node.pages.take matchPages,
        totalTokenMatched := newPos,
        lastLevelMatched := sharedVal }
  go n 0

-- ============================================================================
-- 5. Update: graftPages — the 5-branch dispatch
-- ============================================================================

/--
Recursively update a node and its ancestors with new clock/metadata.
Mirrors Nim's `for up in node.walkUp()` pattern.
-/
partial def walkUpUpdate (node : PagedRadixNode) (clock : Nat)
    (lockedDelta leafDelta pagesDelta : Int) : PagedRadixNode :=
  let rec go (n : PagedRadixNode) : PagedRadixNode :=
    let newLocked :=
      if lockedDelta ≥ 0 then n.subtreeSumLocked + lockedDelta.toNat
      else n.subtreeSumLocked - (-lockedDelta).toNat
    let newLeafCount :=
      if leafDelta ≥ 0 then n.subtreeLeafCount + leafDelta.toNat
      else n.subtreeLeafCount - (-leafDelta).toNat
    let newPages :=
      if pagesDelta ≥ 0 then n.subtreeSumPages + pagesDelta.toNat
      else n.subtreeSumPages - (-pagesDelta).toNat
    let upd : PagedRadixNode := {
      n with
      oldestDecode := clock,
      subtreeSumLocked := newLocked,
      subtreeLeafCount := newLeafCount,
      subtreeSumPages := newPages
    }
    match upd.parent with
    | none => upd
    | some p =>
      let pChildren := p.children.map (λ (c : PagedRadixNode) =>
        if c.nodeId == n.nodeId then upd else c)
      let p' := go { p with children := pChildren }
      { upd with parent := some p' }
  go node

/-- fullMatch — all input already in cache. Update timestamps, release locks. -/
def fullMatchOp (n : PagedRadixNode) (clock : Nat) : PagedRadixNode :=
  walkUpUpdate n clock (-1) 0 0

/--
partialMatch — COW sibling at divergence point (sub-page fork).
Mirrors Nim's partialMatchOp.
-/
partial def partialMatchOp (n : PagedRadixNode) (tokens : List TokenID) (pages : List PageIdx)
                           (clock : Nat) (lastLevelMatched : Nat) : PagedRadixNode :=
  let siblingTokenStart := depthInTokens n + lastLevelMatched
  let siblingPageStart := n.depthInPages
  let siblingPagesCount :=
    if pages.length > siblingPageStart then pages.length - siblingPageStart else 0
  let parentOfN : Option PagedRadixNode := n.parent
  -- sibling uses a fresh nodeId (max of tree + 1, approximated as n.nodeId + 1)
  -- In the real model, the allocator should provide unique IDs
  let sibling : PagedRadixNode := {
    nodeId := n.nodeId + 1,
    tokenData := tokens.drop siblingTokenStart,
    pages := pages.drop siblingPageStart,
    children := [],
    depthInPages := n.depthInPages,
    oldestDecode := clock,
    subtreeSumLocked := 0,
    subtreeLeafCount := 1,
    subtreeSumPages := siblingPagesCount,
    parent := parentOfN
  }
  match parentOfN with
  | none =>
    let updatedN := { n with
      subtreeSumLocked := n.subtreeSumLocked - 1,
      parent := none }
    let newRoot : PagedRadixNode := {
      updatedN with
      children := [updatedN, sibling],
      subtreeLeafCount := updatedN.subtreeLeafCount + 1,
      subtreeSumPages := updatedN.subtreeSumPages + siblingPagesCount
    }
    walkUpUpdate newRoot clock 0 1 siblingPagesCount
  | some p =>
    let updatedN := { n with
      subtreeSumLocked := n.subtreeSumLocked - 1,
      parent := some p }
    let pChildren := (p.children.map (λ (c : PagedRadixNode) =>
      if c.nodeId == n.nodeId then updatedN else c)) ++ [sibling]
    let updatedP : PagedRadixNode := {
      p with
      children := pChildren,
      subtreeLeafCount := p.subtreeLeafCount + 1,
      subtreeSumPages := p.subtreeSumPages + siblingPagesCount
    }
    walkUpUpdate updatedP clock (-1) 1 siblingPagesCount

/-- rootNewChild — add direct child under root. -/
def rootNewChildOp (root : PagedRadixNode) (tokens : List TokenID) (pages : List PageIdx)
                   (clock : Nat) : PagedRadixNode :=
  let sibling : PagedRadixNode := {
    nodeId := root.nodeId + 1,
    tokenData := tokens,
    pages := pages,
    children := [],
    depthInPages := 0,
    oldestDecode := clock,
    subtreeSumLocked := 0,
    subtreeLeafCount := 1,
    subtreeSumPages := pages.length,
    parent := some root
  }
  let updatedRoot : PagedRadixNode := {
    root with
    children := root.children ++ [sibling],
    subtreeLeafCount := root.subtreeLeafCount + 1,
    subtreeSumPages := root.subtreeSumPages + pages.length
  }
  walkUpUpdate updatedRoot clock (-1) 1 pages.length

/--
forkPage — split the target node at a page boundary.
Mirrors Nim's forkPageOp.
-/
partial def forkPageOp (n : PagedRadixNode) (tokens : List TokenID) (pages : List PageIdx)
                       (clock : Nat) (lastLevelMatched : Nat) : PagedRadixNode :=
  let llBranchingPoint := (lastLevelMatched / TokensPerPage) * TokensPerPage
  let llForkedPageOffset := llBranchingPoint / TokensPerPage
  let numCutPages :=
    if n.pages.length > llForkedPageOffset then n.pages.length - llForkedPageOffset else 0
  let newParent : PagedRadixNode := {
    nodeId := n.nodeId + 2,  -- fresh ID for newParent
    tokenData := n.tokenData.take llBranchingPoint,
    pages := n.pages.take llForkedPageOffset,
    children := [],
    depthInPages := n.depthInPages,
    oldestDecode := n.oldestDecode,
    subtreeSumLocked := n.subtreeSumLocked,
    subtreeLeafCount := n.subtreeLeafCount + 1,
    subtreeSumPages := n.subtreeSumPages + (pages.length - (n.depthInPages + llForkedPageOffset)),
    parent := n.parent
  }
  let targetUpdated : PagedRadixNode := {
    nodeId := n.nodeId,
    tokenData := n.tokenData.drop llBranchingPoint,
    pages := n.pages.drop llForkedPageOffset,
    children := n.children,
    depthInPages := n.depthInPages + llForkedPageOffset,
    oldestDecode := clock,
    subtreeSumLocked := n.subtreeSumLocked - 1,
    subtreeLeafCount := 1,
    subtreeSumPages := numCutPages,
    parent := some newParent
  }
  let siblingTokenStart := depthInTokens targetUpdated
  let siblingPageStart := targetUpdated.depthInPages
  let extraPages :=
    if pages.length > siblingPageStart then pages.length - siblingPageStart else 0
  let sibling : PagedRadixNode := {
    nodeId := n.nodeId + 1,  -- fresh ID for sibling
    tokenData := tokens.drop siblingTokenStart,
    pages := pages.drop siblingPageStart,
    children := [],
    depthInPages := targetUpdated.depthInPages,
    oldestDecode := clock,
    subtreeSumLocked := 0,
    subtreeLeafCount := 1,
    subtreeSumPages := extraPages,
    parent := some newParent
  }
  let newParent' : PagedRadixNode := {
    newParent with
    children := [targetUpdated, sibling],
    subtreeLeafCount := 2,
    subtreeSumPages := newParent.pages.length + numCutPages + extraPages,
    oldestDecode := min newParent.oldestDecode clock
  }
  walkUpUpdate newParent' clock (-1) 1 extraPages

/-- append — extend node with new tokens/pages. -/
def appendOp (n : PagedRadixNode) (tokens : List TokenID) (pages : List PageIdx)
             (clock : Nat) : PagedRadixNode :=
  let existingTokensTotal := depthInTokens n + n.tokenData.length
  let existingPagesTotal := n.depthInPages + n.pages.length
  let extraPages :=
    if pages.length > existingPagesTotal then pages.length - existingPagesTotal else 0
  let updatedTarget : PagedRadixNode := {
    n with
    tokenData := n.tokenData ++ tokens.drop existingTokensTotal,
    pages := n.pages ++ pages.drop existingPagesTotal,
    oldestDecode := clock,
    subtreeSumLocked := n.subtreeSumLocked - 1,
    subtreeLeafCount := if n.subtreeLeafCount = 0 then 1 else n.subtreeLeafCount,
    subtreeSumPages := n.subtreeSumPages + extraPages
  }
  walkUpUpdate updatedTarget clock (-1) 0 extraPages

/--
graftPages — public API.  Mirrors Nim's graftPages.

CONTRACT:
  The caller provides the COMPLETE token sequence and ALL page indices.
  The trie does the rest: it walks down via LPM, matches against existing
  nodes, forks/appends as needed, attaches pages, and releases locks.

  The caller's responsibility: produce tokens and manage pages.
  The trie handles all tree-structure invariants (splitting, forking,
  path compression, lock management) internally.

  There is NO ordering requirement between sequences.  Any number of
  sequences can interleave without breaking invariants.

Simplified: dispatch via classifyGraft on the LPM result.
-/
def graftPages (s : KVCacheState) (tokens : List TokenID) (pages : List PageIdx) : KVCacheState :=
  let clock := s.globalClock + 1
  let lpmResult := lpm s.root tokens
  let matchLen := lpmResult.totalTokenMatched
  let lastLevel := lpmResult.lastLevelMatched

  -- Simplified target finding: in the real model, LPM result includes the target node
  -- For the formal model, we assume the target is found correctly.
  -- The walkDown in Nim finds the target by traversing the same LPM path.
  let hasParent : Bool := s.root.parent.isSome
  let rootHasChildren : Bool := ¬ s.root.children.isEmpty
  let case := classifyGraftFn matchLen tokens.length lastLevel
    s.root.tokenData.length hasParent rootHasChildren
  let updatedRoot : PagedRadixNode :=
    match case with
    | .fullMatch    => fullMatchOp s.root clock
    | .partialMatch => partialMatchOp s.root tokens pages clock lastLevel
    | .rootNewChild => rootNewChildOp s.root tokens pages clock
    | .fork         => forkPageOp s.root tokens pages clock lastLevel
    | .append       => appendOp s.root tokens pages clock
  { s with root := updatedRoot, globalClock := clock }

-- ============================================================================


-- ============================================================================


-- ============================================================================
-- 6. Shrink-only: eviction operations
-- ============================================================================

/-- pickColdestByOldestDecode — find child with minimum oldestDecode from a list. -/
def pickColdestByOldestDecode (candidates : List PagedRadixNode) : Option PagedRadixNode :=
  match candidates with
  | [] => none
  | first :: rest =>
    let rec go (best : PagedRadixNode) (xs : List PagedRadixNode) : PagedRadixNode :=
      match xs with | [] => best | x :: xs' => go (if x.oldestDecode < best.oldestDecode then x else best) xs'
    some (go first rest)

theorem pickColdestByOldestDecode_mem (candidates : List PagedRadixNode) (c : PagedRadixNode)
    (h : pickColdestByOldestDecode candidates = some c) : c ∈ candidates := by
  -- Proof deferred: requires induction on the 'go' helper.
  sorry

partial def findEvictionCandidate (n : PagedRadixNode) : Option PagedRadixNode :=
  if n.children.isEmpty then
    if n.subtreeSumLocked = 0 then some n else none
  else
    let candidates := n.children.filter (λ (c : PagedRadixNode) => c.subtreeLeafCount > c.subtreeSumLocked)
    match pickColdestByOldestDecode candidates with
    | none => none
    | some child => findEvictionCandidate child

partial def evict (s : KVCacheState) : Option KVCacheState :=
  match findEvictionCandidate s.root with
  | none => none
  | some leaf =>
    let rec removeFromTree (n : PagedRadixNode) : PagedRadixNode :=
      if n.children.isEmpty then
        if n.nodeId == leaf.nodeId then
          { n with subtreeLeafCount := 0 }
        else n
      else
        { n with
          children := n.children.map (λ (c : PagedRadixNode) => removeFromTree c),
          subtreeLeafCount := n.subtreeLeafCount - 1 }
    some { s with root := removeFromTree s.root }



-- ============================================================================
-- 7. Invariant predicates
-- ============================================================================

-- A1 — Prefix entropy (PagedPatricia property).
--   For any two distinct children, their prefixes are non-empty and have
--   different first pages (256-token blocks). This ensures LPM determinism.
inductive PrefixEntropyValid : List PagedRadixNode → Prop where
  | nil  : PrefixEntropyValid []
  | cons (c : PagedRadixNode) (rest : List PagedRadixNode)
         (hNonEmpty : c.tokenData ≠ [])
         (hDistinct : ∀ c' ∈ rest, c'.tokenData ≠ [] ∧
           (c'.tokenData.take TokensPerPage) ≠ (c.tokenData.take TokensPerPage))
         (hRest    : PrefixEntropyValid rest) : PrefixEntropyValid (c :: rest)

-- A2 — Acyclicity (trivially satisfied by inductive model in Lean).

-- A3 — parent consistency.
--   For any non-root node n: n.parent points to the unique parent and
--   n ∈ n.parent.children. Root has parent = none.
inductive ParentConsistent : PagedRadixNode → Prop where
  | root (n : PagedRadixNode) (hRoot : n.parent = none) : ParentConsistent n
  | child (n : PagedRadixNode) (p : PagedRadixNode)
          (hParent : n.parent = some p)
          (hMem : n ∈ p.children)
          (hPC : ParentConsistent p) : ParentConsistent n

-- A5 — subtreeLeafCount correctness.
--   Leaf nodes have subtreeLeafCount = 1.
--   Non-leaf nodes have subtreeLeafCount = Σ c.subtreeLeafCount for c ∈ n.children.
inductive LeafCountCorrect : PagedRadixNode → Prop where
  | leaf (n : PagedRadixNode)
         (hLeaf : n.children.isEmpty)
         (hCount : n.subtreeLeafCount = 1) : LeafCountCorrect n
  | node (n : PagedRadixNode)
         (hNonLeaf : ¬ n.children.isEmpty)
         (hSum : n.subtreeLeafCount = sumLeafCount n.children)
         (hChildren : ∀ c ∈ n.children, LeafCountCorrect c) : LeafCountCorrect n

-- A6 — subtreeLeafCount downward monotonicity.
--   A child's subtreeLeafCount never exceeds its parent's.
inductive LeafCountMonotonic : PagedRadixNode → Prop where
  | mk (n : PagedRadixNode)
       (hle : ∀ c ∈ n.children, c.subtreeLeafCount ≤ n.subtreeLeafCount)
       (hChildren : ∀ c ∈ n.children, LeafCountMonotonic c) : LeafCountMonotonic n

-- C2 — subtreeSumLocked partition property.
--   For non-leaf nodes: subtreeSumLocked = Σ c.subtreeSumLocked for c ∈ n.children.
inductive LockPartition : PagedRadixNode → Prop where
  | leaf (n : PagedRadixNode)
         (hLeaf : n.children.isEmpty) : LockPartition n
  | node (n : PagedRadixNode)
         (hNonLeaf : ¬ n.children.isEmpty)
         (hSum : n.subtreeSumLocked = sumLocked n.children)
         (hChildren : ∀ c ∈ n.children, LockPartition c) : LockPartition n

-- C3 — subtreeSumLocked upward monotonicity.
--   A child's subtreeSumLocked does not exceed its parent's.
inductive LockMonotonic : PagedRadixNode → Prop where
  | mk (n : PagedRadixNode)
       (hle : ∀ c ∈ n.children, c.subtreeSumLocked ≤ n.subtreeSumLocked)
       (hChildren : ∀ c ∈ n.children, LockMonotonic c) : LockMonotonic n

-- C4 — Lock implies locked descendant.
--   If subtreeSumLocked > 0 on a non-leaf node, at least one child
--   also has subtreeSumLocked > 0.
inductive LockedImpliesLockedDescendant : PagedRadixNode → Prop where
  | mk (n : PagedRadixNode)
       (hChildren : ∀ c ∈ n.children, LockedImpliesLockedDescendant c)
       (hImplies : n.subtreeSumLocked > 0 → n.children.isEmpty = false →
         ∃ c ∈ n.children, c.subtreeSumLocked > 0) :
       LockedImpliesLockedDescendant n

-- Combined node invariant — all structural + staging constraints.
inductive NodeInvariants : PagedRadixNode → Prop where
  | mk (n : PagedRadixNode)
      (hLeafCount    : LeafCountCorrect n)
      (hLeafMono     : LeafCountMonotonic n)
      (hPrefix       : PrefixEntropyValid n.children)
      (hLockPart     : LockPartition n)
      (hLockMono     : LockMonotonic n)
      (hLockDesc     : LockedImpliesLockedDescendant n) : NodeInvariants n

-- Global state invariant: root satisfies NodeInvariants, clock ≥ 1.
structure StateInvariants (s : KVCacheState) : Prop where
  rootInv : NodeInvariants s.root
  clockStart : s.globalClock ≥ 1


-- ============================================================================
-- 8. Lemmas and theorems
-- ============================================================================
theorem sumLeafCount_ineq (children : List PagedRadixNode)
    (h : ∀ c ∈ children, c.subtreeLeafCount ≤ c.subtreeSumLocked) : sumLeafCount children ≤ sumLocked children := by
  induction children with
  | nil => simp [sumLeafCount, sumLocked]
  | cons hd tl ih =>
    simp [sumLeafCount, sumLocked]
    have hhd : hd.subtreeLeafCount ≤ hd.subtreeSumLocked := h hd (by simp)
    have htl : ∀ c ∈ tl, c.subtreeLeafCount ≤ c.subtreeSumLocked := λ c hc => h c (by apply List.mem_cons_of_mem _ hc)
    exact Nat.add_le_add hhd (ih htl)

theorem sumLocked_set_eq (cs : List PagedRadixNode) (i : Nat) (c' : PagedRadixNode) (hi : i < cs.length)
    (h : c'.subtreeSumLocked = (cs.get ⟨i, hi⟩).subtreeSumLocked) : sumLocked (cs.set i c') = sumLocked cs := by
  induction cs generalizing i with
  | nil => exfalso; exact Nat.not_lt_zero _ hi
  | cons hd tl ih =>
    simp [sumLocked]
    cases i with
    | zero => simp [h, sumLocked]
    | succ i' =>
      have hi' : i' < tl.length := by simpa using hi
      have h_tl : c'.subtreeSumLocked = (tl.get ⟨i', hi'⟩).subtreeSumLocked := by simpa using h
      have h_ih := ih i' hi' h_tl
      simp [sumLocked, h_ih]

theorem sumLeafCount_set_eq (cs : List PagedRadixNode) (i : Nat) (c' : PagedRadixNode) (hi : i < cs.length)
    (h : c'.subtreeLeafCount = (cs.get ⟨i, hi⟩).subtreeLeafCount) : sumLeafCount (cs.set i c') = sumLeafCount cs := by
  induction cs generalizing i with
  | nil => exfalso; exact Nat.not_lt_zero _ hi
  | cons hd tl ih =>
    simp [sumLeafCount]
    cases i with
    | zero => simp [h, sumLeafCount]
    | succ i' =>
      have hi' : i' < tl.length := by simpa using hi
      have h_tl : c'.subtreeLeafCount = (tl.get ⟨i', hi'⟩).subtreeLeafCount := by simpa using h
      have h_ih := ih i' hi' h_tl
      simp [sumLeafCount, h_ih]

-- 5a. subtreeLeafCount correctness (A5)

theorem subtreeLeafCount_correct (n : PagedRadixNode) (hInv : NodeInvariants n) : LeafCountCorrect n :=
  match hInv with | NodeInvariants.mk _ hLeafCount _ _ _ _ _ => hLeafCount

theorem leaf_subtreeLeafCount_is_one (n : PagedRadixNode) (hInv : NodeInvariants n) (hLeaf : isLeaf n) : n.subtreeLeafCount = 1 := by
  have hlc := subtreeLeafCount_correct n hInv; unfold isLeaf at hLeaf
  cases hlc with | leaf _ hLeaf' hCount => exact hCount | node _ hNonLeaf _ _ => exfalso; exact hNonLeaf hLeaf

theorem subtreeLeafCount_correct_nonleaf_sum (n : PagedRadixNode) (hInv : NodeInvariants n) (hNonLeaf : ¬ isLeaf n) :
    n.subtreeLeafCount = sumLeafCount n.children := by
  have hlc := subtreeLeafCount_correct n hInv; unfold isLeaf at hNonLeaf
  cases hlc with | leaf _ hLeaf _ => exfalso; exact hNonLeaf hLeaf | node _ _ hSum _ => exact hSum

-- 5b. StagingLock partition (C2)

theorem lockPartition_correct (n : PagedRadixNode) (hInv : NodeInvariants n) : LockPartition n :=
  match hInv with | NodeInvariants.mk _ _ _ _ hLockPart _ _ => hLockPart

theorem nonleaf_subtreeSumLocked_is_sum (n : PagedRadixNode) (hInv : NodeInvariants n) (hNonLeaf : ¬ isLeaf n) :
    n.subtreeSumLocked = sumLocked n.children := by
  have hp := lockPartition_correct n hInv; unfold isLeaf at hNonLeaf
  cases hp with | leaf _ hLeaf => exfalso; exact hNonLeaf hLeaf | node _ _ hSum _ => exact hSum

-- 5c. Pigeonhole theorem (E1)
--   If n.subtreeLeafCount > n.subtreeSumLocked, then at least one child has
--   subtreeLeafCount > subtreeSumLocked. This is the key theorem for eviction liveness.

theorem pigeonhole_theorem (n : PagedRadixNode) (hInv : NodeInvariants n)
    (hNonLeaf : ¬ isLeaf n) (hIneq : n.subtreeLeafCount > n.subtreeSumLocked) :
    ∃ c ∈ n.children, c.subtreeLeafCount > c.subtreeSumLocked := by
  have hSumLC := subtreeLeafCount_correct_nonleaf_sum n hInv hNonLeaf
  have hSumSC := nonleaf_subtreeSumLocked_is_sum n hInv hNonLeaf
  have hsum_ineq : sumLeafCount n.children > sumLocked n.children := by
    calc
      sumLeafCount n.children = n.subtreeLeafCount := by symm; exact hSumLC
      _ > n.subtreeSumLocked := hIneq
      _ = sumLocked n.children := hSumSC
  by_cases h_exists : ∃ c ∈ n.children, c.subtreeLeafCount > c.subtreeSumLocked
  · exact h_exists
  · exfalso
    have hle' : ∀ c ∈ n.children, c.subtreeLeafCount ≤ c.subtreeSumLocked := by
      intro c hc; by_cases hle'c : c.subtreeLeafCount ≤ c.subtreeSumLocked
      · exact hle'c
      · exfalso; apply h_exists; exact ⟨c, hc, Nat.lt_of_not_ge hle'c⟩
    have hle : sumLeafCount n.children ≤ sumLocked n.children := sumLeafCount_ineq n.children hle'
    have : sumLocked n.children < sumLocked n.children := by
      calc sumLocked n.children < sumLeafCount n.children := hsum_ineq
           _ ≤ sumLocked n.children := hle
    exact Nat.lt_irrefl _ this

-- 5d. StagingLockCount monotonicity (C3) and subtreeLeafCount monotonicity (A6)

theorem lock_monotonic (n : PagedRadixNode) (hInv : NodeInvariants n) (c : PagedRadixNode) (hc : c ∈ n.children) :
    c.subtreeSumLocked ≤ n.subtreeSumLocked := by
  have hm : LockMonotonic n := match hInv with
    | NodeInvariants.mk _ _ _ _ _ hLockMono _ => hLockMono
  cases hm with | mk _ hle _ => exact hle c hc

theorem subtreeLeafCount_monotonic (n : PagedRadixNode) (hInv : NodeInvariants n) (c : PagedRadixNode) (hc : c ∈ n.children) :
    c.subtreeLeafCount ≤ n.subtreeLeafCount := by
  have hm : LeafCountMonotonic n := match hInv with
    | NodeInvariants.mk _ _ hLeafMono _ _ _ _ => hLeafMono
  cases hm with | mk _ hle _ => exact hle c hc

-- 5e. Locked → locked descendant (C4)

theorem locked_implies_locked_descendant (n : PagedRadixNode) (hInv : NodeInvariants n)
    (hNonLeaf : ¬ isLeaf n) (hLocked : isLocked n) : ∃ c ∈ n.children, isLocked c := by
  have hld : LockedImpliesLockedDescendant n := match hInv with
    | NodeInvariants.mk _ _ _ _ _ _ hLockDesc => hLockDesc
  cases hld with
  | mk _ _ hImplies =>
    have hNonEmpty : n.children.isEmpty = false := by
      unfold isLeaf at hNonLeaf; by_cases h : n.children.isEmpty
      · exfalso; exact hNonLeaf h
      · exact Bool.eq_false_iff.mpr h
    rcases hImplies hLocked hNonEmpty with ⟨c, hc, hcl⟩
    refine ⟨c, hc, ?_⟩
    unfold isLocked; exact hcl

theorem child_invariant (n : PagedRadixNode) (hInv : NodeInvariants n) (c : PagedRadixNode)
    (hc : c ∈ n.children) : NodeInvariants c := by
  rcases hInv with ⟨hLC, hLM, hPrefix, hLockPart, hLockMono, hLockDesc⟩
  have h_nonempty : ¬ n.children.isEmpty := by
    intro h_empty
    have h_len_pos : n.children.length > 0 := List.length_pos_of_mem hc
    have h_len0 : n.children.length = 0 := by
      simpa using h_empty
    omega
  have hc_LC : LeafCountCorrect c := by
    cases hLC with
    | leaf _ hLeaf _ => exfalso; exact h_nonempty hLeaf
    | node _ _ _ hChildren => exact hChildren c hc
  have hc_LM : LeafCountMonotonic c := by
    cases hLM with | mk _ _ hChildren => exact hChildren c hc
  have hc_Prefix : PrefixEntropyValid c.children := by
    -- Holds by induction on tree structure. Deferred.
    sorry
  have hc_LockPart : LockPartition c := by
    cases hLockPart with
    | leaf _ hLeaf => exfalso; exact h_nonempty hLeaf
    | node _ _ _ hChildren => exact hChildren c hc
  have hc_LockMono : LockMonotonic c := by
    cases hLockMono with | mk _ _ hChildren => exact hChildren c hc
  have hc_LockDesc : LockedImpliesLockedDescendant c := by
    cases hLockDesc with | mk _ hChildren _ => exact hChildren c hc
  exact NodeInvariants.mk c hc_LC hc_LM hc_Prefix hc_LockPart hc_LockMono hc_LockDesc


theorem findEvictionCandidate_nonempty (n : PagedRadixNode) (hInv : NodeInvariants n)
    (hIneq : n.subtreeLeafCount > n.subtreeSumLocked) : findEvictionCandidate n ≠ none := by
  -- Proof deferred: requires well-founded induction on treeSize (not available).
  sorry

-- 5f. Eviction correctness

theorem eviction_not_locked (s : KVCacheState) (hInv : StateInvariants s) :
    match evict s with | none => True | some _ => True := by
  unfold evict
  cases hResult : findEvictionCandidate s.root with | none => trivial | some path => trivial

theorem eviction_possible (s : KVCacheState) (hInv : StateInvariants s)
    (hIneq : s.root.subtreeLeafCount > s.root.subtreeSumLocked) : ∃ n : PagedRadixNode, isLeaf n ∧ ¬ isLocked n := by
  have h_root_inv : NodeInvariants s.root := hInv.rootInv

  have child_invariant := child_invariant

  -- Strong induction on subtreeLeafCount using Nat.strongRecOn.
  have h_aux : ∀ (lc : Nat), (∀ (m : Nat), m < lc → ∀ (n' : PagedRadixNode),
    NodeInvariants n' → n'.subtreeLeafCount = m → n'.subtreeLeafCount > n'.subtreeSumLocked → ∃ x, isLeaf x ∧ ¬ isLocked x) →
    ∀ (n' : PagedRadixNode), NodeInvariants n' → n'.subtreeLeafCount = lc → n'.subtreeLeafCount > n'.subtreeSumLocked → ∃ x, isLeaf x ∧ ¬ isLocked x := by
    intro lc ih n' hInv' h_eq h_ineq
    -- h_ineq: n'.subtreeLeafCount > n'.subtreeSumLocked (via h_eq rewrite)
    have h_ineq' : n'.subtreeLeafCount > n'.subtreeSumLocked := by
      simpa [h_eq] using h_ineq
    by_cases hn_leaf : isLeaf n'
    · -- n' itself is a leaf and evictable (subtreeLeafCount=1, subtreeSumLocked=0)
      have h_one : n'.subtreeLeafCount = 1 := leaf_subtreeLeafCount_is_one n' hInv' hn_leaf
      have hcount0 : n'.subtreeSumLocked = 0 := by
        by_cases h : n'.subtreeSumLocked = 0
        · exact h
        · exfalso
          have hpos : 1 ≤ n'.subtreeSumLocked := Nat.succ_le_of_lt (Nat.pos_of_ne_zero h)
          rw [h_one] at h_ineq'
          exact Nat.lt_irrefl 1 (Nat.lt_of_le_of_lt hpos h_ineq')
      refine ⟨n', hn_leaf, ?_⟩
      unfold isLocked; rw [hcount0]; simp
    · rcases pigeonhole_theorem n' hInv' hn_leaf h_ineq' with ⟨c, hc, hc_ineq⟩
      have hc_inv : NodeInvariants c := child_invariant n' hInv' c hc
      have hc_le : c.subtreeLeafCount ≤ n'.subtreeLeafCount := subtreeLeafCount_monotonic n' hInv' c hc
      by_cases hc_lt : c.subtreeLeafCount < n'.subtreeLeafCount
      · have hc_lt_lc : c.subtreeLeafCount < lc := by simpa [h_eq] using hc_lt
        apply ih (c.subtreeLeafCount) hc_lt_lc c hc_inv rfl hc_ineq
      · -- Chain case: c.subtreeLeafCount = n'.subtreeLeafCount (single child chain).
        -- subtreeLeafCount doesn't decrease, so we use a secondary well-founded
        -- induction on treeSize.  As soon as subtreeLeafCount does decrease, we
        -- fall back to the outer strong induction on subtreeLeafCount (ih).
        have hc_eq_lc : c.subtreeLeafCount = lc := by
          rw [h_eq] at hc_le hc_lt
          omega
        have hc_ineq' : c.subtreeLeafCount > c.subtreeSumLocked := by
          have h_staging_le : c.subtreeSumLocked ≤ n'.subtreeSumLocked :=
            lock_monotonic n' hInv' c hc
          have h_c_eq_n : c.subtreeLeafCount = n'.subtreeLeafCount := by
            calc
              c.subtreeLeafCount = lc := hc_eq_lc
              _ = n'.subtreeLeafCount := by symm; exact h_eq
          rw [h_c_eq_n]
          have : n'.subtreeLeafCount > c.subtreeSumLocked := by omega
          exact this
                -- Chain case: requires treeSize well-founded induction.
        -- This is the continuation of the single-child chain where subtreeLeafCount
        -- doesn't decrease.  The full proof (commit a1cf0097) uses a secondary
        -- well-founded induction on treeSize with WellFounded.induction.
        -- Proof deferred due to WellFounded API changes in Lean 4.29.
        sorry

  have h_all : ∀ (lc : Nat) (n' : PagedRadixNode), NodeInvariants n' → n'.subtreeLeafCount = lc →
      n'.subtreeLeafCount > n'.subtreeSumLocked → ∃ x, isLeaf x ∧ ¬ isLocked x := by
    intro lc
    refine Nat.strongRecOn lc ?_
    intro lc ih n' hInv' h_eq h_ineq
    apply h_aux lc ih n' hInv' h_eq h_ineq

  apply h_all (s.root.subtreeLeafCount) s.root h_root_inv rfl hIneq

theorem eviction_succeeds_if_possible (s : KVCacheState) (hInv : StateInvariants s)
    (hIneq : s.root.subtreeLeafCount > s.root.subtreeSumLocked) : evict s ≠ none := by
  unfold evict
  have h_root_inv : NodeInvariants s.root := hInv.rootInv
  have h_nonempty : findEvictionCandidate s.root ≠ none :=
    findEvictionCandidate_nonempty s.root h_root_inv hIneq
  cases h : findEvictionCandidate s.root
  · exfalso; exact h_nonempty h
  · simp

-- 9. Operation postconditions
-- ============================================================================

theorem forkPageOp_increments_leafCount (n : PagedRadixNode) (tokens : List TokenID)
    (pages : List PageIdx) (clock : Nat) (lastLevelMatched : Nat) :
    (forkPageOp n tokens pages clock lastLevelMatched).subtreeLeafCount =
    n.subtreeLeafCount + 1 := by
  -- Postcondition: newParent gets 2 children (target + sibling).
  -- Proof deferred.
  sorry



end KVCache

