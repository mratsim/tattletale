# Positron pure Nim cuda kernel

Unfortunately it seems impossible at the moment to do a 2-stage compilation of pure Nim cuda kernels.

The goal is to have NVCC compile the cuda kernels and GCC/Clang compile regular Nim code, however
the static library and the calling library will conflicts on function names
- multiple definition of `dollar___systemZdollars_u29(int)'
- multiple definition of `dollar___systemZdollars_u14(long)'
- multiple definition of `raiseEIO__stdZsyncio_u92(NimStringV2)'
- multiple definition of `align__system_u1648(long, long)'
- ....

And single stage compilation is blocked by Nim not being able to specify a per-file compiler.