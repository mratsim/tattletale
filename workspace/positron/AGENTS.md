# Positron Agents Guidelines

## Benchmarks

- Score of record: arithmetic flops actually done divided by end-to-end wall-clock, all loads, stores and copies paid in the denominator.
- An mma-only rate is never an achievement metric, no percent-of-peak claims, no parity claims.
- Optimizing a kernel is mostly optimizing data movement, keeping the cores busy is the goal, not a given.
- Roofline terms (bytes moved, achieved bandwidth, compute ceiling) are diagnostics that attribute time, never scores.
