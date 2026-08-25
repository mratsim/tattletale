# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Unified builtin vocabulary for every backend: the canonical MSL names
# plus the per-backend idiom aliases that expand to them (see builtins_catalog.nim).
import ./builtins_pragmas
import ./builtins_catalog
import ./builtins_functions
import ./builtins_gpu_types

export builtins_pragmas
export builtins_catalog
export builtins_functions
export builtins_gpu_types
