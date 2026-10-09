## Layout algebra: shapes, strides, Int[N], coalesce.
##
## Ceramic provides the fundamental tile types (`Layout[Shape, Stride]`,
## `Int[N]`) and layout transformations (`coalesce`, `complement`, `compose`,
## `logical_divide`, filter, sort).
##
## Reference:
##   - CuTe C++: layout.hpp, coalesce.cpp, complement.cpp, composition.cpp, logical_divide.cpp
##   - Python: tensor-layouts

import ./src/int_tuples
import ./src/layout_algebra
import ./src/tensors
import ./src/kernels/k_layout_gemm
import ./src/kernels/k_layout_copy_cpu
import ./src/kernels/k_layout_copy_gpu
import ./src/kernels/k_layout_fillwith_cpu
import ./src/kernels/k_layout_fillwith_gpu

export int_tuples, layout_algebra, tensors
export k_layout_gemm, k_layout_copy_cpu, k_layout_copy_gpu,
       k_layout_fillwith_cpu, k_layout_fillwith_gpu
