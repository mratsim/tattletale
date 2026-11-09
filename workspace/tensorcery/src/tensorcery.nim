# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/tensorcery/src/tensors_public_types
import workspace/tensorcery/src/tensors_internal_types
import workspace/tensorcery/src/tensors_internals
import workspace/tensorcery/src/tensors_globals
import workspace/tensorcery/src/compute_graph/dispatch
import workspace/tensorcery/src/compute_graph/ops_2i1o/ops_add
import workspace/tensorcery/src/compute_graph/ops_1iNo/ops_split
import workspace/tensorcery/src/compute_graph/eval
import workspace/tensorcery/src/tensors_public_api

# op files register at this import, the walk expands at the user's
# first eval call
import workspace/tensorcery/src/compute_graph/eval

export tensors_public_types
export tensors_globals
# erased plumbing: Tensor's field types live here and toAnyEngine binds
# a vtable at the call site, the impls resolve in the caller scope
export tensors_internals
export tensors_public_api
# eval resolves at that same toAnyEngine site
export eval.eval
