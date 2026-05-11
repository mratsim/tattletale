# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# This file exposes the underlying torch API and is only intended
# for testing the raw API itself, debugging or rewrapping in a nim-ified API.

import workspace/libtorch/src/raw/[
  abi/torch_tensors,
  torch_tensors_sugar,
  torch_tensors_overloads,
  abi/c10,
  abi/neural_nets,
  abi/std_cpp
]
export torch_tensors, torch_tensors_sugar, torch_tensors_overloads, c10, neural_nets

# TODO: for now we expose C++ tuples `get` and CppVector
export std_cpp