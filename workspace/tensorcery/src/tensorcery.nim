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

# op files register at this import, the walk expands at the engine accessor's scope
import workspace/tensorcery/src/tensors_public_api

export tensors_public_api
