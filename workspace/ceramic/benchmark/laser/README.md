# Laser Strided GEMM

This folder is a fully standalone reimplementation of Laser's strided GEneral Matrix Multiply. https://github.com/mratsim/laser/tree/master/laser/primitives/matrix_multiplication

The only difference is that instead of relying on https://github.com/pytorch/cpuinfo
for runtime CPU features detection, it uses Constantine's CPU feature detection (https://github.com/mratsim/constantine)