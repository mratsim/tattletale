/-
# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
-/

/-!
# Lean toolchain gate (P0)

Smoke module for the Lean toolchain gate: proves the lake build pipeline
elaborates, typechecks, and compiles real proofs on the pinned toolchain.
-/

theorem add_comm_smoke (a b : Nat) : a + b = b + a := Nat.add_comm a b

example : (1 : Nat) + 1 = 2 := by rfl

#eval s!"hello from lean, 2 + 2 = {2 + 2}"
