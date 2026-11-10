# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible
import workspace/ceramic
import workspace/ceramic/src/tile_algebra
import workspace/crucible/src/runtime/engines/engines_api
import workspace/tensorcery/src/tensors_public_types
import workspace/tensorcery/src/tensors_internal_types
import workspace/tensorcery/src/tensors_internals
import workspace/tensorcery/src/tensors_globals
import workspace/tensorcery/src/compute_graph/dispatch
import workspace/tensorcery/src/compute_graph/ops_2i1o/ops_add
import workspace/tensorcery/src/compute_graph/ops_2i1o/ops_matmul
import workspace/tensorcery/src/compute_graph/ops_1i1o/ops_rmsnorm
import workspace/tensorcery/src/compute_graph/ops_Ni1o/ops_rope
import workspace/tensorcery/src/compute_graph/ops_1iNo/ops_split
import workspace/tensorcery/src/compute_graph/ops_Ni1o/ops_embed_gather
import workspace/tensorcery/src/compute_graph/eval as graph_eval
import workspace/tensorcery/src/tensors_public_api

# op files register at this import, the walk expands at the engine accessor's scope
import workspace/tensorcery/src/tensors_public_api

export tensors_public_api
# Client-surface re-exports (D2-6): iface vtable machinery and op dispatch
# resolve at the importing module's scope. An umbrella-only client needs
# these symbols visible at its import site:
# Graph and vtable machinery:
# - tensors_internals: the generated initVTable and the erased-layer types
# - graph_eval.eval: the erased eval implementation behind the iface method
# - tensors_globals and engines_api: engine constructors plus HwEngine
#   concept members that constrain the erased generics
#
# Kernel surface the dispatch instantiates at the eval call site:
# - crucible and ceramic: enqueue macros and view types
# - tile_algebra: the tile kernel templates that enqueue walks resolve:
#   tile load/store and the matmul thread->value maps
export tensors_internals
export graph_eval.eval
export tensors_globals
export engines_api
export crucible
export ceramic
export tile_algebra
