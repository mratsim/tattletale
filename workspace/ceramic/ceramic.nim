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
import ./src/layouts
import ./src/layout_algebra
import ./src/tensors
import ./src/tile_algebra
import ./src/kernel_gemm_epilogues
import ./src/tile_algebra/tile_epilogues
import ./src/kernels/k_tile_gemm
import ./src/kernels/k_tile_rmsnorm
import ./src/kernels/k_tile_attn
import ./src/kernel_copy_cpu
import ./src/kernel_copy_gpu
import ./src/kernel_fillwith_cpu
import ./src/kernel_fillwith_gpu
import ./src/layout_indexing_cpu
import ./src/layout_indexing_gpu
import ./src/layout_indexing

export int_tuples, layouts, layout_algebra, tensors, tile_algebra,
       tile_epilogues, kernel_gemm_epilogues,
       gemm, matmul, rms_single_row, attn_fwd,
       gemm_relu, linear, linear_relu
export kernel_copy_cpu, kernel_copy_gpu, kernel_fillwith_cpu,
       kernel_fillwith_gpu, layout_indexing_cpu, layout_indexing_gpu,
       layout_indexing
