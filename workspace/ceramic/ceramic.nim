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
import ./src/tensors
import ./src/kernel_copy_cpu
import ./src/kernel_copy_gpu
import ./src/kernel_fillwith_cpu
import ./src/kernel_fillwith_gpu
import ./src/kernel_indexing_cpu
import ./src/kernel_indexing_gpu

export int_tuples, layouts, layout_algebra, tensors
export kernel_copy_cpu, kernel_copy_gpu, kernel_fillwith_cpu,
       kernel_fillwith_gpu, kernel_indexing_cpu, kernel_indexing_gpu
