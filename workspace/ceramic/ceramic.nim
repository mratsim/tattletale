# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CuTe-compatible Layout algebra: shapes, strides, Int[N], coalesce.
##
## Ceramic provides the fundamental tile types (`Layout[Shape, Stride]`,
## `Int[N]`) and layout transformations (`coalesce`, `complement`, `compose`,
## `logical_divide`, filter, sort).
##
## Reference:
##   - CuTe C++: layout.hpp, coalesce.cpp, complement.cpp, composition.cpp, logical_divide.cpp
##   - Python: tensor-layouts

import ./src/int_tuples
import ./src/layouts
import ./src/layout_algebra

export int_tuples, layouts, layout_algebra
