# Positron tile kernels: shared contracts

The tile kernel modules under `src/kernels/` share three contracts.

## Inline tile composition

- device tile procs compose inline into the calling kernel's body
- a megakernel stage calls the tile core at its own coordinates,
  no exit to the host between stages

## No-copy binding

- buffers bind as page-aligned host memory with page-multiple byte lengths
- the device reads and writes the host's pages in place
- anything else copy-ins, in-place writes are lost

## One monomorphization per call-site line

- each generic device proc needs one distinct
  call-site line per static binding set
- the engine's monomorphization key erases the generic static bindings,
  calls sharing a call-site line collapse into a single compiled body
