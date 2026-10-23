/-
# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
-/

import AddKernel

/-!
# Negative control

Claims the guarded thread body also writes the *next* index with the thread's
sum. False by computation: with the host test vectors (`a = 10`, `b = 1`,
`out = 0`), index 1 stays at 0. Kept out of the lake glob; the gate is
`lake env lean Negative.lean` must exit nonzero. The expected rejection is
`tactic 'rfl' failed`: the two sides are not definitionally equal.
-/

theorem writes_next_index :
    addBody 0 2 (fun _ => (10 : UInt32)) (fun _ => 1) (fun _ => 0) 1 = 11 := by
  rfl
