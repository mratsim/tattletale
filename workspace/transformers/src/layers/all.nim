# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Design
#   For now we don't create a Layer concept or inheritance root
#   We just conventionally use `forward` as the common function to run a tensor through any layer.
#   This might change if there is a need to store multiple heterogenous layers
#   But unlike training where people might want a common interface for
#   - @[Conv2D, MaxPool2D, Conv2D ..., MLP]
#   - @[Embedding, Attention, MLP, ..., MLP]
#   for inference, all the layers are defined at load-time, fixed and can be hidden within a "Model" type

import
  ./rope,
  ./norm,
  ./transformer

export
  rope, norm, transformer
