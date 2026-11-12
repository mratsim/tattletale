# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

## Compile-time VM budget guard: toTensor only, no eval, no kernel enqueue.
##
## The AnyEngine interface vtable eagerly instantiates the eval lambda at vtable creation.
## That instantiation lowers every registered op handler and runs the artifact pipeline at compile time.
##
## On a healthy build this file:
## - compiles within the default VM step budget, no compiler flag needed
## - compiles in the time of an empty program plus imports
##
## Run (from the worktree root):
##   nim c --hints:off --warnings:off --outdir:build/tests/t_vm_budget \
##     --nimcache:nimcache/t_vm_budget workspace/tensorcery/tests/graph/t_vm_budget.nim

import workspace/crucible
import workspace/ceramic
import workspace/tensorcery

let x = @[1.0'f32, 2].toTensor([2], runMetal, float32)
doAssert x.size() == 2
