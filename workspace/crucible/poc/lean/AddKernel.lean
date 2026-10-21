/-
# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
-/

/-!
# Add kernel, per-thread verification model (P1)

Hand translation of the crucible add kernel thread function
(`tests/codegen/vulkan/app/t_vulkan_add.nim`, `addKernel`) into Lean.

The verification unit is the per-thread body, parametrized by thread id:

    if tid < n: out[tid] := a[tid] + b[tid]

Buffers are modelled as total functions `Nat → UInt32` and a write as a
functional update, so "unchanged elsewhere" is definitional and no heap or
separation-logic framework is needed. `UInt32.add` wraps modulo 2^32,
matching the GPU (and Nim `uint32`) arithmetic the kernel emits.
-/

/-- Per-thread body of the add kernel: returns the output buffer after thread
`tid` of `n` has run.

`a` and `b` are the input buffers, `out` the output buffer before the write.
Precondition: `tid < n` (each thread is launched with its id inside the
buffer bound). Threads with `tid ≥ n` leave the buffer untouched, mirroring
the guard the crucible frontend must emit around every body.

Example: `addBody 0 2 (fun _ => 10) (fun _ => 1) (fun _ => 0) 0 = 11`. -/
def addBody (tid n : Nat) (a b out : Nat → UInt32) : Nat → UInt32 :=
  if tid < n then fun i => if i = tid then a tid + b tid else out i
  else out

/-- Postcondition, write half: thread `tid` wrote `a tid + b tid` at its own index. -/
theorem addBody_writes_own_index {tid n : Nat} {a b out : Nat → UInt32} (h : tid < n) :
    addBody tid n a b out tid = a tid + b tid := by
  simp [addBody, h]

/-- Postcondition, no-collateral-damage half: every other index keeps its old value. -/
theorem addBody_preserves_others {tid n : Nat} {a b out : Nat → UInt32} (h : tid < n)
    {i : Nat} (hi : i ≠ tid) :
    addBody tid n a b out i = out i := by
  simp [addBody, h, hi]

/-- The per-thread postcondition as a single contract. -/
theorem addBody_post {tid n : Nat} {a b out : Nat → UInt32} (h : tid < n) :
    addBody tid n a b out tid = a tid + b tid ∧
      ∀ i, i ≠ tid → addBody tid n a b out i = out i :=
  ⟨addBody_writes_own_index h, fun _ hi => addBody_preserves_others h hi⟩

/-- Bounds theorem: the only index a thread can change is its own, which is
in bounds by the launch precondition. This is the property the codegen gate
must establish for every emitted kernel body. -/
theorem addBody_only_writes_tid {tid n : Nat} {a b out : Nat → UInt32} {i : Nat}
    (h : addBody tid n a b out i ≠ out i) :
    i = tid ∧ tid < n := by
  by_cases hlt : tid < n
  · simp only [addBody, hlt] at h
    by_cases heq : i = tid
    · exact ⟨heq, hlt⟩
    · simp [heq] at h
  · simp [addBody, hlt] at h

/-- Worked example matching the host-side test vectors `a = [10, 20]`,
`b = [1, 2]`: thread 0 writes 11. -/
example : addBody 0 2 (fun _ => (10 : UInt32)) (fun _ => 1) (fun _ => 0) 0 = 11 := by
  decide

/-- `UInt32.add` wraps modulo 2^32 like the GPU `u32` add, so the spec needs
no overflow side conditions: thread 0 of a length-1 buffer holding
`0xFFFFFFFF` and `1` writes `0`. -/
example : addBody 0 1 (fun _ => (0xFFFFFFFF : UInt32)) (fun _ => 1) (fun _ => 7) 0 = 0 := by
  decide
