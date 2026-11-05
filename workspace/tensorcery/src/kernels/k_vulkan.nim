# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan GLSL kernel sandbox: one GLSL source per kernel, one kernel per source.
##
## Source contract, shared by every kernel here:
##
## - `#version 450`, `local_size_x = 256`, entry point `void main()`
## - SSBOs at set 0, bindings in the record blob's slot order: the output first,
##   then the inputs, then the scalar image as the last binding
## - each scalar image is one std430 buffer block whose members match the Nim
##   Scalars object field for field (int32 and float32 pack sequentially at 4 bytes)
## - glslangValidator compiles with `-e <kernelName> --source-entrypoint main`,
##   so the SPIR-V entry point is named <kernelName>

const kAddGlsl* = """
#version 450
layout(local_size_x = 256) in;
layout(std430, set = 0, binding = 0) buffer OutBuf { float o[]; } outb;
layout(std430, set = 0, binding = 1) buffer ABuf { float a[]; } abuf;
layout(std430, set = 0, binding = 2) buffer BBuf { float b[]; } bbuf;
layout(std430, set = 0, binding = 3) buffer AddSBuf {
  int aOff;
  int bOff;
  int shape[4];
  int sa[4];
  int sb[4];
  int n;
} p;
void main() {
  int i = int(gl_GlobalInvocationID.x);
  if (i < p.n) {
    int c3 = i % p.shape[3];
    int t2 = i / p.shape[3];
    int c2 = t2 % p.shape[2];
    int t1 = t2 / p.shape[2];
    int c1 = t1 % p.shape[1];
    int c0 = t1 / p.shape[1];
    int ia = p.aOff + (c0 * p.sa[0] + c1 * p.sa[1]) + c2 * p.sa[2] + c3 * p.sa[3];
    int ib = p.bOff + (c0 * p.sb[0] + c1 * p.sb[1]) + c2 * p.sb[2] + c3 * p.sb[3];
    outb.o[i] = abuf.a[ia] + bbuf.b[ib];
  }
}
"""

const kMulGlsl* = """
#version 450
layout(local_size_x = 256) in;
layout(std430, set = 0, binding = 0) buffer OutBuf { float o[]; } outb;
layout(std430, set = 0, binding = 1) buffer ABuf { float a[]; } abuf;
layout(std430, set = 0, binding = 2) buffer MulSBuf {
  int aOff;
  int shape[4];
  int sa[4];
  float scalar;
  int n;
} p;
void main() {
  int i = int(gl_GlobalInvocationID.x);
  if (i < p.n) {
    int c3 = i % p.shape[3];
    int t2 = i / p.shape[3];
    int c2 = t2 % p.shape[2];
    int t1 = t2 / p.shape[2];
    int c1 = t1 % p.shape[1];
    int c0 = t1 / p.shape[1];
    int ia = p.aOff + (c0 * p.sa[0] + c1 * p.sa[1]) + c2 * p.sa[2] + c3 * p.sa[3];
    outb.o[i] = abuf.a[ia] * p.scalar;
  }
}
"""

const kMul2Glsl* = """
#version 450
layout(local_size_x = 256) in;
layout(std430, set = 0, binding = 0) buffer OutBuf { float o[]; } outb;
layout(std430, set = 0, binding = 1) buffer ABuf { float a[]; } abuf;
layout(std430, set = 0, binding = 2) buffer BBuf { float b[]; } bbuf;
layout(std430, set = 0, binding = 3) buffer Mul2SBuf {
  int aOff;
  int bOff;
  int shape[4];
  int sa[4];
  int sb[4];
  int n;
} p;
void main() {
  int i = int(gl_GlobalInvocationID.x);
  if (i < p.n) {
    int c3 = i % p.shape[3];
    int t2 = i / p.shape[3];
    int c2 = t2 % p.shape[2];
    int t1 = t2 / p.shape[2];
    int c1 = t1 % p.shape[1];
    int c0 = t1 / p.shape[1];
    int ia = p.aOff + (c0 * p.sa[0] + c1 * p.sa[1]) + c2 * p.sa[2] + c3 * p.sa[3];
    int ib = p.bOff + (c0 * p.sb[0] + c1 * p.sb[1]) + c2 * p.sb[2] + c3 * p.sb[3];
    outb.o[i] = abuf.a[ia] * bbuf.b[ib];
  }
}
"""

const kFillGlsl* = """
#version 450
layout(local_size_x = 256) in;
layout(std430, set = 0, binding = 0) buffer OutBuf { float o[]; } outb;
layout(std430, set = 0, binding = 1) buffer FillSBuf { int n; } p;
void main() {
  int i = int(gl_GlobalInvocationID.x);
  if (i < p.n) {
    outb.o[i] = 1.0;
  }
}
"""

const kWhereGlsl* = """
#version 450
layout(local_size_x = 256) in;
layout(std430, set = 0, binding = 0) buffer OutBuf { float o[]; } outb;
layout(std430, set = 0, binding = 1) buffer CBuf { float c[]; } cbuf;
layout(std430, set = 0, binding = 2) buffer ABuf { float a[]; } abuf;
layout(std430, set = 0, binding = 3) buffer BBuf { float b[]; } bbuf;
layout(std430, set = 0, binding = 4) buffer WhereSBuf {
  int condOff;
  int aOff;
  int bOff;
  int n;
} p;
void main() {
  int i = int(gl_GlobalInvocationID.x);
  if (i < p.n) {
    if (cbuf.c[p.condOff + i] != 0.0) {
      outb.o[i] = abuf.a[p.aOff + i];
    } else {
      outb.o[i] = bbuf.b[p.bOff + i];
    }
  }
}
"""
