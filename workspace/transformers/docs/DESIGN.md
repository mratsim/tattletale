# Tattletale Transformers: Design Principles

## Core Principle: Stateless Forward Pass

**All layer `forward()` functions must be pure computations with no side effects.**

This means:
- No mutation of layer state during forward
- No hidden counters or position tracking
- No implicit cache updates
- All state must be passed explicitly as parameters

Consequence: idempotency, same inputs mean same output

Stateless forward passes enable:

1. **Tensor Parallelism**: Split computation across GPUs without synchronizing hidden state
2. **Pipeline Parallelism**: Pass activations between stages without coupling to layer-internal counters
3. **Prefill-Decode Disaggregation**: Run prefill and decode on different workers with explicit state handoff
4. **Continuous Batching**: Process requests at different sequence positions simultaneously
5. **Prefix Caching**: Share KV cache blocks across requests with explicit position mapping
6. **Speculative Decoding**: Run draft and target models with independent state tracking
7. **CUDA Graphs**: Capture static computation graphs without mutable state invalidation

This follows the commonly recurring designs:
- functional core with isolated side-effects as championed by Haskell
- ports and adapters pattern or hexagonal architecture.