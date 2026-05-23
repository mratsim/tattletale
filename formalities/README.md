# Formal specifications and formal verification

Tattletale strives to provide high-quality and reliable primitives. Hence it makes thorough tests, extensive uses of fixtures to avoid regression, diferential fuzzing where possible.

Furthermore it strive to isolate stateful and state management from idempotent functions (i.e. same inputs -> same output often called pure functions).

The next step is having formal specifications and formal verification of stateful datastructure to ensure gaps are closed.
This folder is a collection of symlink to formal specifications of Tattletale inner workings.

We hope to formalize:
- KV Cache data structure
- Layout Algebra
- Threadpool

Formal verification will likely be in Lean and/or TLA+