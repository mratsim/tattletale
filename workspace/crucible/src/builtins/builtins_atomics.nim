# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./builtins_pragmas

# ═══════════════════════════════════════════════════════════
#   Atomics
# ═══════════════════════════════════════════════════════════

proc atomicFetchAdd*[T](p: ptr T, v: T): T {.builtin.} = discard
  ## Atomically adds `v` to the value at `p` and returns the old value.
  ## `p` points to device memory. WGSL: none, no float atomics.
  ##
  ## - MSL: `atomic_fetch_add_explicit` with order relaxed over
  ##   a `(volatile device atomic_float*)` cast, fp32 needs Metal 3+
  ## - CUDA: `atomicAdd`
  ## - GLSL: `atomicAdd` under GL_EXT_shader_atomic_float for fp
  ## - OpenCL: `atomic_fetch_add` under cl_ext_float_atomics for fp
