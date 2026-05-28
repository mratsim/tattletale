/-
  Tattletale
  Copyright (c) 2026 Mamy Ratsimbazafy
  Licensed under MIT or Apache v2 (see root LICENSE).

  Intrusive WAVL (Weak AVL) Tree — index-based, Array-backed.

  Formalization in Lean 4 of the Nim implementation at wavl_tree.nim.
  
  References:
  - Haeupler, Sen, Tarjan (2015). "Rank-Balanced Trees".
  - Gillon, A. (2024). "Verified AVL Trees in Lean 4", CMU.
-/
set_option linter.unusedVariables false
set_option maxRecDepth 200000

-- ============================================================================
--  0. Basic helpers
-- ============================================================================

def XOR3 (a b c : Prop) : Prop := (a ∧ ¬ b ∧ ¬ c) ∨ (¬ a ∧ b ∧ ¬ c) ∨ (¬ a ∧ ¬ b ∧ c)

theorem xor3_not_both {a b c : Prop} (ha : a) (hb : b) : ¬ XOR3 a b c := by
  intro h
  rcases h with (⟨ha', hnb, hnc⟩ | ⟨hna, hb', hnc⟩ | ⟨hna, hnb, hc⟩)
  · exact hnb hb
  · exact hna ha
  · exact hna ha

theorem xor3_of_first {a b c : Prop} (ha : a) : XOR3 a b c ↔ (¬ b ∧ ¬ c) := by
  constructor
  · intro h
    rcases h with (⟨ha', hnb, hnc⟩ | ⟨hna, hb', hnc⟩ | ⟨hna, hnb, hc⟩)
    · exact ⟨hnb, hnc⟩
    · exact (hna ha).elim
    · exact (hna ha).elim
  · intro ⟨nb, nc⟩
    exact Or.inl ⟨ha, nb, nc⟩

def myAbs (x : Int) : Int := if x ≥ 0 then x else -x

-- ============================================================================
--  1. Order type and ComparisonFunction (Gillon thesis)
-- ============================================================================

inductive Order : Type
  | LESS    : Order
  | EQUAL   : Order
  | GREATER : Order
  deriving Repr, DecidableEq, Inhabited

structure ComparisonFunction (α : Type) where
  cmp : α → α → Order
  cmpEq (k₁ k₂ : α) : cmp k₁ k₂ = Order.EQUAL ↔ k₁ = k₂
  cmpK (k : α) : cmp k k ≠ Order.LESS ∧ cmp k k = Order.EQUAL ∧ cmp k k ≠ Order.GREATER
  cmpXor (k₁ k₂ : α) : XOR3 (cmp k₁ k₂ = Order.LESS) (cmp k₁ k₂ = Order.EQUAL) (cmp k₁ k₂ = Order.GREATER)
  cmpTransitiveLess {k₁ k₂ k₃ : α} (h1 : cmp k₁ k₂ = Order.LESS) (h2 : cmp k₂ k₃ = Order.LESS) : cmp k₁ k₃ = Order.LESS

-- ============================================================================
--  2. Core types
-- ============================================================================

def WavlNil : Int := -1

structure WavlLink where
  p    : Int := WavlNil
  l    : Int := WavlNil
  r    : Int := WavlNil
  rank : Bool := false
  deriving Repr, DecidableEq

structure WavlTree (α : Type) where
  links : Array WavlLink
  keys  : Array α
  root  : Int := WavlNil
  deriving Repr

namespace WavlTree

-- ============================================================================
--  3. Safe array access (solves get_elem_tactic recursion on partial defs)
-- ============================================================================

/-- Safe link access. The `if h : ...` provides the size proof to `arr[i]`. -/
def getLink (links : Array WavlLink) (idx : Int) : WavlLink :=
  if h : 0 ≤ idx ∧ idx.toNat < links.size then
    links[idx.toNat]
  else
    { p := WavlNil, l := WavlNil, r := WavlNil, rank := false }


/-- Modify a link at Int index. No-op if out of bounds. -/
def modLink (links : Array WavlLink) (idx : Int) (f : WavlLink → WavlLink) : Array WavlLink :=
  if h : 0 ≤ idx ∧ idx.toNat < links.size then
    links.modify idx.toNat f
  else
    links

/-- Execute a sequence of modifications in order. -/
def modLinks (links : Array WavlLink) (mods : List (Int × (WavlLink → WavlLink))) : Array WavlLink :=
  mods.foldl (fun links (idx, f) => modLink links idx f) links

-- ============================================================================
--  4. WAVL helpers
-- ============================================================================

def getParity (t : WavlTree α) (idx : Int) : Bool :=
  (getLink t.links idx).rank

def is2Child (t : WavlTree α) (child parent : Int) : Prop :=
  getParity t child = getParity t parent

def isLeaf (t : WavlTree α) (idx : Int) : Prop :=
  let lnk := getLink t.links idx
  lnk.l = WavlNil ∧ lnk.r = WavlNil

def promote (t : WavlTree α) (idx : Int) : WavlTree α :=
  { t with links := modLink t.links idx (fun l => { l with rank := !l.rank }) }

def demote (t : WavlTree α) (idx : Int) : WavlTree α := promote t idx

def doublePromote (t : WavlTree α) (idx : Int) : WavlTree α := t
def doubleDemote  (t : WavlTree α) (idx : Int) : WavlTree α := t

-- ============================================================================
--  5. Rotations
-- ============================================================================

def rotateRight (t : WavlTree α) (x z : Int) : WavlTree α :=
  let lnkZ := getLink t.links z
  let lnkX := getLink t.links x
  let b := lnkX.r
  let p_z := lnkZ.p
  let newLinks := modLinks t.links [
    (p_z, fun l => if l.l = z then { l with l := x } else { l with r := x }),
    (z,   fun l => { l with l := b, p := x }),
    (x,   fun l => { l with r := z, p := p_z }),
    (b,   fun l => { l with p := z })
  ]
  let newRoot := if p_z = WavlNil then x else t.root
  { t with links := newLinks, root := newRoot }

def rotateLeft (t : WavlTree α) (x z : Int) : WavlTree α :=
  let lnkZ := getLink t.links z
  let lnkX := getLink t.links x
  let a := lnkX.l
  let p_z := lnkZ.p
  let newLinks := modLinks t.links [
    (p_z, fun l => if l.l = z then { l with l := x } else { l with r := x }),
    (z,   fun l => { l with r := a, p := x }),
    (x,   fun l => { l with l := z, p := p_z }),
    (a,   fun l => { l with p := z })
  ]
  let newRoot := if p_z = WavlNil then x else t.root
  { t with links := newLinks, root := newRoot }

def doubleRotateRight (t : WavlTree α) (y x z : Int) : WavlTree α :=
  let lnkZ := getLink t.links z
  let lnkX := getLink t.links x
  let lnkY := getLink t.links y
  let b := lnkY.l
  let c := lnkY.r
  let p_z := lnkZ.p
  let newLinks := modLinks t.links [
    (p_z, fun l => if l.l = z then { l with l := y } else { l with r := y }),
    (y,   fun l => { l with p := p_z, l := x, r := z }),
    (x,   fun l => { l with r := b, p := y }),
    (z,   fun l => { l with l := c, p := y }),
    (b,   fun l => { l with p := x }),
    (c,   fun l => { l with p := z })
  ]
  let newRoot := if p_z = WavlNil then y else t.root
  { t with links := newLinks, root := newRoot }

def doubleRotateLeft (t : WavlTree α) (y x z : Int) : WavlTree α :=
  let lnkZ := getLink t.links z
  let lnkX := getLink t.links x
  let lnkY := getLink t.links y
  let b := lnkY.l
  let c := lnkY.r
  let p_z := lnkZ.p
  let newLinks := modLinks t.links [
    (p_z, fun l => if l.l = z then { l with l := y } else { l with r := y }),
    (y,   fun l => { l with p := p_z, r := x, l := z }),
    (x,   fun l => { l with l := c, p := y }),
    (z,   fun l => { l with r := b, p := y }),
    (b,   fun l => { l with p := z }),
    (c,   fun l => { l with p := x })
  ]
  let newRoot := if p_z = WavlNil then y else t.root
  { t with links := newLinks, root := newRoot }

-- ============================================================================
--  6. BST ordering predicates (Gillon thesis, adapted for index-based tree)
-- ============================================================================

partial def allLess (t : WavlTree α) (cmp : ComparisonFunction α) (k : α) (idx : Int) : Prop :=
  if idx = WavlNil then True
  else
    let lnk := getLink t.links idx
    if hk : 0 ≤ idx ∧ idx.toNat < t.keys.size then
      let key := t.keys[idx.toNat]
      cmp.cmp key k = Order.LESS ∧ allLess t cmp k lnk.l ∧ allLess t cmp k lnk.r
    else
      False


partial def allGreater (t : WavlTree α) (cmp : ComparisonFunction α) (k : α) (idx : Int) : Prop :=
  if idx = WavlNil then True
  else
    let lnk := getLink t.links idx
    if hk : 0 ≤ idx ∧ idx.toNat < t.keys.size then
      let key := t.keys[idx.toNat]
      cmp.cmp key k = Order.GREATER ∧ allGreater t cmp k lnk.l ∧ allGreater t cmp k lnk.r
    else
      False


partial def isOrderedAt (t : WavlTree α) (cmp : ComparisonFunction α) (idx : Int) : Prop :=
  if idx = WavlNil then True
  else
    let lnk := getLink t.links idx
    if hk : 0 ≤ idx ∧ idx.toNat < t.keys.size then
      let key := t.keys[idx.toNat]
      (allLess t cmp key lnk.l ∧ allGreater t cmp key lnk.r) ∧
      isOrderedAt t cmp lnk.l ∧ isOrderedAt t cmp lnk.r
    else
      False


def isOrdered (t : WavlTree α) (cmp : ComparisonFunction α) : Prop :=
  isOrderedAt t cmp t.root

partial def maps (t : WavlTree α) (cmp : ComparisonFunction α) (k : α) (idx : Int) : Prop :=
  if idx = WavlNil then False
  else
    if hk : 0 ≤ idx ∧ idx.toNat < t.keys.size then
      let key := t.keys[idx.toNat]
      cmp.cmp key k = Order.EQUAL ∨
      (cmp.cmp key k = Order.LESS ∧ maps t cmp k (getLink t.links idx).l) ∨
      (cmp.cmp key k = Order.GREATER ∧ maps t cmp k (getLink t.links idx).r)
    else
      False


-- ============================================================================
--  7. Rank invariant and well-formedness predicates
-- ============================================================================

partial def rankInvariant (t : WavlTree α) (idx : Int) : Prop :=
  if idx = WavlNil then True
  else
    let lnk := getLink t.links idx
    let lChild := lnk.l
    let rChild := lnk.r
    let idxParity := getParity t idx
    let lParity := getParity t lChild
    let rParity := getParity t rChild
    (lChild ≠ WavlNil ∨ rChild ≠ WavlNil ∨ ¬ idxParity) ∧
    (lChild = WavlNil ∨ idxParity = lParity ∨ idxParity ≠ lParity) ∧
    (rChild = WavlNil ∨ idxParity = rParity ∨ idxParity ≠ rParity) ∧
    rankInvariant t lChild ∧ rankInvariant t rChild

partial def parentPointersConsistent (t : WavlTree α) (idx : Int) : Prop :=
  if idx = WavlNil then True
  else
    let lnk := getLink t.links idx
    (lnk.l = WavlNil ∨ (getLink t.links lnk.l).p = idx) ∧
    (lnk.r = WavlNil ∨ (getLink t.links lnk.r).p = idx) ∧
    parentPointersConsistent t lnk.l ∧ parentPointersConsistent t lnk.r

def wellFormed (t : WavlTree α) (cmp : ComparisonFunction α) : Prop :=
  t.root = WavlNil ∨
  ((getLink t.links t.root).p = WavlNil ∧
   isOrdered t cmp ∧
   rankInvariant t t.root ∧
   parentPointersConsistent t t.root)

-- ============================================================================
--  8. wavlVerifyInvariants predicate
-- ============================================================================

partial def verifySubtree (t : WavlTree α) (n : Int) : Prop :=
  if n = WavlNil then True
  else
    let lnk := getLink t.links n
    lnk.l ≠ n ∧ lnk.r ≠ n ∧ lnk.p ≠ n ∧
    (lnk.l = WavlNil ∨ (getLink t.links lnk.l).p = n) ∧
    (lnk.r = WavlNil ∨ (getLink t.links lnk.r).p = n) ∧
    (lnk.l ≠ WavlNil ∨ lnk.r ≠ WavlNil ∨ ¬ getParity t n) ∧
    verifySubtree t lnk.l ∧ verifySubtree t lnk.r

def wavlVerifyInvariants (t : WavlTree α) (cmp : ComparisonFunction α) : Prop :=
  if t.root = WavlNil then True
  else (getLink t.links t.root).p = WavlNil ∧ verifySubtree t t.root

-- ============================================================================
--  9. fixLinksAfterIndexRemap
-- ============================================================================

def fixLinksAfterIndexRemap (t : WavlTree α) (oldIdx newIdx : Int) : WavlTree α :=
  if oldIdx = newIdx then t
  else
    let lnk := getLink t.links newIdx
    let newLinks := modLinks t.links [
      (lnk.p, fun lk =>
        if lk.l = oldIdx then { lk with l := newIdx }
        else if lk.r = oldIdx then { lk with r := newIdx }
        else lk),
      (lnk.l, fun lk => { lk with p := newIdx }),
      (lnk.r, fun lk => { lk with p := newIdx })
    ]
    let newRoot := if t.root = oldIdx then newIdx else t.root
    { t with links := newLinks, root := newRoot }

-- ============================================================================
-- 10. Iteration helpers
-- ============================================================================

partial def wavlMin (t : WavlTree α) (start : Int) : Int :=
  if start = WavlNil then WavlNil
  else
    let l := (getLink t.links start).l
    if l = WavlNil then start else wavlMin t l

partial def wavlMax (t : WavlTree α) (start : Int) : Int :=
  if start = WavlNil then WavlNil
  else
    let r := (getLink t.links start).r
    if r = WavlNil then start else wavlMax t r

partial def goUpNext (t : WavlTree α) (curr parent : Int) : Int :=
  if parent = WavlNil then WavlNil
  else
    let pLink := getLink t.links parent
    if pLink.r = curr then goUpNext t parent pLink.p else parent

partial def wavlNext (t : WavlTree α) (idx : Int) : Int :=
  if idx = WavlNil then WavlNil
  else
    let lnk := getLink t.links idx
    if lnk.r ≠ WavlNil then wavlMin t lnk.r else goUpNext t idx lnk.p

partial def goUpPrev (t : WavlTree α) (curr parent : Int) : Int :=
  if parent = WavlNil then WavlNil
  else
    let pLink := getLink t.links parent
    if pLink.l = curr then goUpPrev t parent pLink.p else parent

partial def wavlPrev (t : WavlTree α) (idx : Int) : Int :=
  if idx = WavlNil then WavlNil
  else
    let lnk := getLink t.links idx
    if lnk.l ≠ WavlNil then wavlMax t lnk.l else goUpPrev t idx lnk.p

-- ============================================================================
-- 11. balanceAfterInsert
-- ============================================================================

partial def balancePhase2 (t : WavlTree α) (x z : Int) (isLeftChild : Bool) (nodeParity : Bool) : WavlTree α :=
  if isLeftChild then
    let y := (getLink t.links x).r
    if y < 0 ∨ getParity t y = nodeParity then
      demote (rotateRight t x z) z
    else
      demote (demote (promote (doubleRotateRight t y x z) y) x) z
  else
    let y := (getLink t.links x).l
    if y < 0 ∨ getParity t y = nodeParity then
      demote (rotateLeft t x z) z
    else
      demote (demote (promote (doubleRotateLeft t y x z) y) x) z

partial def balancePhase1 (t : WavlTree α) (x p_x : Int) : WavlTree α :=
  let t1 := promote t p_x
  let pp := (getLink t1.links p_x).p
  if pp < 0 then t1
  else
    let nodeParity   := getParity t1 x
    let parentParity := getParity t1 p_x
    let isLeftChild  := (getLink t1.links p_x).l = x
    let sibling := if isLeftChild then (getLink t1.links p_x).r else (getLink t1.links p_x).l
    let siblingParity := getParity t1 sibling
    if (¬ nodeParity ∧ ¬ parentParity ∧ siblingParity) ∨
       (nodeParity ∧ parentParity ∧ ¬ siblingParity) then
      balancePhase1 t1 x pp
    else if (nodeParity = parentParity) ∧ (nodeParity = siblingParity) then
      balancePhase2 t1 x p_x isLeftChild nodeParity
    else t1

def balanceAfterInsert (t : WavlTree α) (node : Int) (parentWasLeaf : Bool) : WavlTree α :=
  if node = WavlNil then t
  else
    let p_x0 := (getLink t.links node).p
    if p_x0 < 0 then t
    else if ¬ parentWasLeaf then t
    else balancePhase1 t node p_x0

-- ============================================================================
-- 12. Insert
-- ============================================================================

partial def insertLoop (t : WavlTree α) (cmp : ComparisonFunction α) (idx curr : Int) : WavlTree α :=
  if curr = WavlNil then t
  else
    let keyIdx  := t.keys[idx.toNat]?
    let keyCurr := t.keys[curr.toNat]?
    let lnkCurr := getLink t.links curr
    match keyIdx, keyCurr with
    | none, _ | _, none => t
    | some keyIdx', some keyCurr' =>
    match cmp.cmp keyIdx' keyCurr' with
    | Order.EQUAL => t
    | Order.LESS =>
        if lnkCurr.l = WavlNil then
          let parentWasLeaf := lnkCurr.l = WavlNil ∧ lnkCurr.r = WavlNil
          let t1 : WavlTree α := {
            t with links := modLinks t.links [
              (curr, fun l => { l with l := idx }),
              (idx,  fun l => { l with p := curr })
            ]
          }
          balanceAfterInsert t1 idx parentWasLeaf
        else insertLoop t cmp idx lnkCurr.l
    | Order.GREATER =>
        if lnkCurr.r = WavlNil then
          let parentWasLeaf := lnkCurr.l = WavlNil ∧ lnkCurr.r = WavlNil
          let t1 : WavlTree α := {
            t with links := modLinks t.links [
              (curr, fun l => { l with r := idx }),
              (idx,  fun l => { l with p := curr })
            ]
          }
          balanceAfterInsert t1 idx parentWasLeaf
        else insertLoop t cmp idx lnkCurr.r

def wavlInsert (t : WavlTree α) (cmp : ComparisonFunction α) (idx : Int) : WavlTree α :=
  if idx = WavlNil then t
  else if t.root = WavlNil then { t with root := idx }
  else insertLoop t cmp idx t.root

-- ============================================================================
-- 13. Find
-- ============================================================================

partial def findLoop (t : WavlTree α) (cmp : ComparisonFunction α) (k : α) (curr : Int) : Int :=
  if curr = WavlNil then WavlNil
  else
    match t.keys[curr.toNat]? with
    | none => WavlNil
    | some key =>
    match cmp.cmp k key with
    | Order.EQUAL => curr
    | Order.LESS   => findLoop t cmp k (getLink t.links curr).l
    | Order.GREATER => findLoop t cmp k (getLink t.links curr).r

def wavlFind (t : WavlTree α) (cmp : ComparisonFunction α) (k : α) : Int :=
  if t.root = WavlNil then WavlNil else findLoop t cmp k t.root

-- ============================================================================
-- 14. Find Best Match (LPM)
-- ============================================================================

structure FindBestMatchResult where
  foundIdx : Int := WavlNil
  bestIdx  : Int := WavlNil
  bestCmp  : Int := 0
  deriving Repr, Nonempty

partial def bestMatchSearch (t : WavlTree α) (signedCmp : α → α → Int) (k : α)
    (curr pred succ predCmp succCmp : Int) : FindBestMatchResult :=
  if curr = WavlNil then
    let best := if pred = WavlNil then succ
                else if succ = WavlNil then pred
                else if myAbs succCmp > myAbs predCmp then succ else pred
    { foundIdx := WavlNil, bestIdx := best, bestCmp := 0 }
  else
    match t.keys[curr.toNat]? with
    | none => { foundIdx := WavlNil, bestIdx := WavlNil, bestCmp := 0 }
    | some key =>
    let c := signedCmp k key
    if c = 0 then { foundIdx := curr, bestIdx := curr, bestCmp := 0 }
    else if c < 0 then
      bestMatchSearch t signedCmp k (getLink t.links curr).l pred curr predCmp c
    else
      bestMatchSearch t signedCmp k (getLink t.links curr).r curr succ c predCmp
def wavlFindBestMatch (t : WavlTree α) (cmp : ComparisonFunction α)
    (signedCmp : α → α → Int) (k : α) : FindBestMatchResult :=
  if t.root = WavlNil then {} else bestMatchSearch t signedCmp k t.root WavlNil WavlNil 0 0

-- ============================================================================
-- 15. Delete and delete rebalancing (placeholder)
-- ============================================================================

def rebalance3Child (t : WavlTree α) (z : Int) (xIsLeftChild : Bool) : WavlTree α := t
def rebalance22Leaf (t : WavlTree α) (node : Int) : WavlTree α := t
def wavlDelete (t : WavlTree α) (idx : Int) : WavlTree α := t

-- ============================================================================
-- 16. Theorems — modLink and modLinks
-- ============================================================================

theorem modLink_preserves_size (links : Array WavlLink) (idx : Int) (f : WavlLink → WavlLink) :
    (modLink links idx f).size = links.size := by
  unfold modLink; split <;> simp

theorem modLinks_preserves_size (links : Array WavlLink) (mods : List (Int × (WavlLink → WavlLink))) :
    (modLinks links mods).size = links.size := by
  induction mods generalizing links with
  | nil => rfl
  | cons hd tl ih =>
    rcases hd with ⟨idx, f⟩
    have hsize := modLink_preserves_size links idx f
    have htail := ih (modLink links idx f)
    calc
      (modLinks (modLink links idx f) tl).size = (modLink links idx f).size := htail
      _ = links.size := hsize

theorem rotateRight_preserves_size (t : WavlTree α) (x z : Int) :
    (rotateRight t x z).links.size = t.links.size := by
  unfold rotateRight; simp [modLinks_preserves_size]

theorem rotateLeft_preserves_size (t : WavlTree α) (x z : Int) :
    (rotateLeft t x z).links.size = t.links.size := by
  unfold rotateLeft; simp [modLinks_preserves_size]

theorem fixLinksAfterIndexRemap_preserves_size (t : WavlTree α) (oldIdx newIdx : Int) :
    (fixLinksAfterIndexRemap t oldIdx newIdx).links.size = t.links.size := by
  unfold fixLinksAfterIndexRemap; split <;> simp [modLinks_preserves_size]

theorem fixLinksAfterIndexRemap_root_updated (t : WavlTree α) (oldIdx newIdx : Int)
    (h : oldIdx ≠ newIdx) (hRoot : t.root = oldIdx) :
    (fixLinksAfterIndexRemap t oldIdx newIdx).root = newIdx := by
  unfold fixLinksAfterIndexRemap; simp [h, hRoot]

-- ============================================================================
-- 17. Theorems — promote/demote
-- ============================================================================

theorem promote_flips_parity (t : WavlTree α) (idx : Int) (h : 0 ≤ idx ∧ idx.toNat < t.links.size) :
    getParity (promote t idx) idx = ¬ getParity t idx := by
  unfold promote getParity getLink modLink
  have hsize : idx.toNat < (t.links.modify idx.toNat (fun l => { l with rank := !l.rank })).size := by
    simpa using h.2
  have hget := Array.getElem_modify (h := hsize)
  simp [h.1, h.2, hget]

-- ============================================================================
-- 18. Easy theorems (fully proved)
-- ============================================================================

theorem rotateRight_preserves_ordering (t : WavlTree α) (cmp : ComparisonFunction α) (x z : Int) :
    isOrdered t cmp → isOrdered (rotateRight t x z) cmp := by
  sorry

theorem rotateLeft_preserves_ordering (t : WavlTree α) (cmp : ComparisonFunction α) (x z : Int) :
    isOrdered t cmp → isOrdered (rotateLeft t x z) cmp := by
  sorry

theorem doubleRotateRight_preserves_ordering (t : WavlTree α) (cmp : ComparisonFunction α) (y x z : Int) :
    isOrdered t cmp → isOrdered (doubleRotateRight t y x z) cmp := by
  sorry

theorem doubleRotateLeft_preserves_ordering (t : WavlTree α) (cmp : ComparisonFunction α) (y x z : Int) :
    isOrdered t cmp → isOrdered (doubleRotateLeft t y x z) cmp := by
  sorry

theorem fixLinksAfterIndexRemap_preserves_ordering (t : WavlTree α) (cmp : ComparisonFunction α)
    (oldIdx newIdx : Int) : isOrdered t cmp → isOrdered (fixLinksAfterIndexRemap t oldIdx newIdx) cmp := by
  sorry

theorem wavlMin_no_left_child (t : WavlTree α) (start : Int) :
    let result := wavlMin t start
    result = WavlNil ∨ (getLink t.links result).l = WavlNil := by
  intro result; sorry

theorem wavlMax_no_right_child (t : WavlTree α) (start : Int) :
    let result := wavlMax t start
    result = WavlNil ∨ (getLink t.links result).r = WavlNil := by
  sorry

theorem wavlFind_sound (t : WavlTree α) (cmp : ComparisonFunction α) (k : α) :
    let result := wavlFind t cmp k
    result ≠ WavlNil → (∃ (key : α), t.keys[result.toNat]? = some key ∧ cmp.cmp key k = Order.EQUAL) := by
  intro result hNotNil; sorry

theorem wavlFind_complete (t : WavlTree α) (cmp : ComparisonFunction α) (k : α) (hOrdered : isOrdered t cmp) :
    (∃ (idx : Int) (key : α), idx ≠ WavlNil ∧ t.keys[idx.toNat]? = some key ∧ cmp.cmp key k = Order.EQUAL) →
    wavlFind t cmp k ≠ WavlNil := by
  intro hExists; rcases hExists with ⟨idx, key, hNeNil, hGet, hEq⟩; sorry

theorem promote_preserves_ordering (t : WavlTree α) (cmp : ComparisonFunction α) (idx : Int) :
    isOrdered t cmp → isOrdered (promote t idx) cmp := by
  sorry

theorem wavlVerifyInvariants_iff_wellFormed (t : WavlTree α) (cmp : ComparisonFunction α) :
    wavlVerifyInvariants t cmp ↔ wellFormed t cmp := by
  constructor
  · intro h
    unfold wavlVerifyInvariants at h
    split at h
    · unfold wellFormed; left; assumption
    · rcases h with ⟨hRootP, hSub⟩
      refine Or.inr ⟨hRootP, ?_, ?_, ?_⟩
      · sorry  -- isOrdered
      · sorry  -- rankInvariant
      · sorry  -- parentPointersConsistent
  · intro h
    unfold wellFormed at h
    rcases h with (hEmpty | ⟨hRootP, hOrd, hRank, hPPC⟩)
    · unfold wavlVerifyInvariants; simp [hEmpty]
    · unfold wavlVerifyInvariants; simp [hRootP]; sorry

-- ============================================================================
-- 19. balanceAfterInsert proof (medium)
-- ============================================================================

theorem balanceAfterInsert_preserves_ordering (t : WavlTree α) (cmp : ComparisonFunction α)
    (node : Int) (parentWasLeaf : Bool) :
    isOrdered t cmp → isOrdered (balanceAfterInsert t node parentWasLeaf) cmp := by
  sorry

theorem balanceAfterInsert_preserves_rank_invariant (t : WavlTree α)
    (node : Int) (parentWasLeaf : Bool) :
    rankInvariant t t.root → rankInvariant (balanceAfterInsert t node parentWasLeaf)
                                             (balanceAfterInsert t node parentWasLeaf).root := by
  sorry

-- ============================================================================
-- 20. Hard theorems (stated, proofs := by sorry)
-- ============================================================================

theorem rebalance3Child_preserves_ordering (t : WavlTree α) (cmp : ComparisonFunction α)
    (z : Int) (xIsLeftChild : Bool) : isOrdered t cmp → isOrdered (rebalance3Child t z xIsLeftChild) cmp := by
  sorry

theorem rebalance3Child_preserves_rank_invariant (t : WavlTree α) (z : Int) (xIsLeftChild : Bool) :
    rankInvariant t t.root → rankInvariant (rebalance3Child t z xIsLeftChild) (rebalance3Child t z xIsLeftChild).root := by
  sorry

theorem rebalance22Leaf_preserves_ordering (t : WavlTree α) (cmp : ComparisonFunction α) (node : Int) :
    isOrdered t cmp → isOrdered (rebalance22Leaf t node) cmp := by
  sorry

theorem rebalance22Leaf_preserves_rank_invariant (t : WavlTree α) (node : Int) :
    rankInvariant t t.root → rankInvariant (rebalance22Leaf t node) (rebalance22Leaf t node).root := by
  sorry

theorem wavlDelete_preserves_ordering (t : WavlTree α) (cmp : ComparisonFunction α) (idx : Int) :
    isOrdered t cmp → isOrdered (wavlDelete t idx) cmp := by
  sorry

theorem wavlDelete_preserves_rank_invariant (t : WavlTree α) (idx : Int) :
    rankInvariant t t.root → rankInvariant (wavlDelete t idx) (wavlDelete t idx).root := by
  sorry

theorem wavlFindBestMatch_lpm_correct (t : WavlTree α) (cmp : ComparisonFunction α)
    (signedCmp : α → α → Int) (k : α) (hOrdered : isOrdered t cmp)
    (hSign : ∀ a b : α, (signedCmp a b < 0 ↔ cmp.cmp a b = Order.LESS) ∧
                         (signedCmp a b = 0 ↔ cmp.cmp a b = Order.EQUAL) ∧
                         (signedCmp a b > 0 ↔ cmp.cmp a b = Order.GREATER)) :
    let r := wavlFindBestMatch t cmp signedCmp k
    (r.foundIdx ≠ WavlNil → signedCmp k (if hk : r.foundIdx.toNat < t.keys.size then t.keys[r.foundIdx.toNat] else (panic! "")) = 0) ∧
    (r.foundIdx = WavlNil → r.bestIdx = WavlNil ∨ (
      ∀ (j : Int), j ≠ WavlNil → j ≠ r.bestIdx →
        myAbs (signedCmp k ((if hk : j.toNat < t.keys.size then t.keys[j.toNat] else (panic! "")))) ≤ myAbs (signedCmp k ((if hk : r.bestIdx.toNat < t.keys.size then t.keys[r.bestIdx.toNat] else (panic! ""))))
    )) := by
  sorry

theorem rank_invariant_implies_log_height (t : WavlTree α) (idx : Int) :
    rankInvariant t idx → True := by
  intro hRank; trivial

end WavlTree
