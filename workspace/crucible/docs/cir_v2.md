# CIRv2: the Crucible IR v2 tree

CIRv2 is the Crucible IR v2 tree, where IR means intermediate representation.
Crucible is the GPU compile stack of this workspace: a family of Nim macros, one per target.
The targets are `metal:`, `cuda:`, `opencl:`, `glsl:`, `spirv:`, `webgpu:`, `hlsl:`.
Each macro compiles one kernel body to one target source.

A Nim macro is a function that runs during compilation and generates the program's code.
`{. ... .}` braces mark Nim pragmas, annotations the compiler reads.
A kernel is the GPU function the host launches, and a pass is one in-place rewrite of the tree.
The frontend builds CIRv2, each pass rewrites it, and each printer consumes it.

CIRv2 is a plain Nim object-variant tree, and a Nim object-variant is a tagged union.
The representation has no compile-time evaluation requirement, and no text parses back into CIRv2.
The production frontend builds the tree during macro expansion, and the tests hand-build it at plain run time.
The only NimNode-typed field is `CirType.origin`, a debug reference to the source node, and hand-built fixtures leave it nil.

Macro time names the phase of Nim compilation where macros run.
CIRv2 nodes exist only inside one macro expansion.

**Why "v2".**
CIRv2 is the second revision of the Crucible IR.
Its predecessor shaped the tree as a GPU abstract syntax tree, one node per source construct.

Its predecessor keeps the same plain Nim representation, a `GpuNode = ref object` object-variant
tree with one node kind per construct in the Nim grammar.
The v1 kinds come from the Nim grammar, and the v2 kinds encode the normal form.

The normal form is the tree shape that the verifier enforces at every pass boundary.
The Normal form section below states it as four rules.

CIRv2 builds on the multi-join control-flow model of Shoggoth IR (nimony @ a987a15d, 2026-10-29).
Shoggoth IR is the last tree that Nimony, the rewrite of the Nim compiler, builds before code generation.

The IR keeps its tree small by joining related constructs into fewer node kinds.
CIRv2 adopts that model: a closed node set in the Leng vocabulary.

The set is larger than the v1 AST's, not smaller: 63 statement and expression kinds against v1's 44.
The growth buys typed operands and the merge-point contracts.
It is stated here as a cost, not hidden under "small".

**Why Leng.**
Leng is Nimony's midlevel IR dialect (`doc/internals/leng-spec.md` in the Nimony repository).
CIRv2 borrows its vocabulary wholesale.

Node kinds have Leng names (`eAdd`, `sIte`, `lab`/`jmp`).
Typed operations state their type in the `(add Type a b)` discipline of the Leng grammar.
`ptr`/`aptr` keep the Leng distinction between a pointer to one element and a pointer to an array of elements.
Its debug renderer speaks the same dialect, so a CIRv2 tree reads like a Leng fragment.

## CIRv2 from the high level to the nitty gritty details

Let's dive into CIRv2 from the high-level to the nitty gritty details.

But first we need to introduce the basic blocks.
A module (`CirModule`) contains nominal types, procs, and program-scope globals.
Nominal means two struct types are the same type only when they share the same name.
A proc (`CirProc`, Nim's function) has parameters, a return type, and a body of statements.

A statement (`CirStmt`) is a declaration, an assignment, a control-flow construct, or a terminator.
An expression (`CirExpr`) is a literal, a symbol, an operator, or a call.
A type (`CirType`) records the type of every typed node.
As a user of the macros you never see these nodes: the frontend builds them and the printers consume them.

The stack has a front-end and a backend.
Its front-end is the lowering in `frontend/nim_to_crucible.nim`.
It translates the typed Nim tree of a macro body into CIRv2, normal form by construction.

Its backend is the pass pipeline plus one printer per target.
Passes rewrite the tree toward what the target can express, and the printer prints it in the target source language.

```
typed NimNode ──lower──▶ CIRv2 normal form ──verify──▶ CirModule ──passes──▶ printer ──▶ target source
```

Implementation points, all under `workspace/crucible/src/`:

| Module | Role |
|---|---|
| `codegen/ir/cirv2_lang.nim` | datatypes for address spaces, the type kinds (the closed, fixed set the IR models), expressions, statements, procs, modules, and the builtin categories |
| `codegen/ir/cirv2_build.nim` | builders that produce well-shaped nodes by construction |
| `codegen/ir/cirv2_pretty.nim` | the debug renderer, s-expression shaped (one parenthesized node per line, as in Lisp), in the Leng vocabulary |
| `codegen/ir/cirv2_verify.nim` | the normal-form verifier |
| `codegen/frontend/nim_to_crucible.nim` | the typed-NimNode → CIRv2 lowering |
| `codegen/passes/pass_datatypes.nim` | the pass machinery (`pass_registry.nim` keeps the registry) |
| `codegen/gpu_compiler.nim` | the macro entry points and the per-target pipeline |
| `codegen/backends/*_lang.nim` | one printer per target |
| `builtins/builtins_catalog.nim` | the public builtin catalog |

---

## Terminology

| Term | Meaning |
|---|---|
| Leng | Nimony's midlevel IR dialect. CIRv2 borrows its node names, its `(add Type a b)` typing discipline, and its `ptr`/`aptr` distinction. |
| Macro time | The phase of Nim compilation where macros run. CIRv2 nodes live only there. |
| Kernel | A compute entry point that the GPU host launches. In source it is a `{.global.}` proc, and the printers emit the backend's kernel qualifier and argument ABI for it. |
| Argument ABI | The application binary interface: the calling convention the emitted code and the host side agree on. |
| Device function | An ordinary proc compiled for the GPU, called from a kernel or from another device function. |
| `launchDeviceFn` shape | One of the two shapes the frontend accepts: a block of kernel proc definitions, or a single `{.global.}` wrapper proc whose body calls the kernel. |
| Workgroup | A group of threads that launch together, share memory, and synchronize with a barrier. Metal calls it a threadgroup, CUDA calls it a block, OpenCL calls it a work-group. |
| Threadgroup | Metal's name for a workgroup. CIRv2 uses the Metal names because the canonical builtin names are Metal Shading Language (MSL) names. |
| Address space | The memory class a pointer addresses: per-thread registers, workgroup-shared storage, constant storage, or device-global storage. |
| Normal form | The tree shape that every producer guarantees and that every consumer assumes. The next section states it as four axioms. |
| `lab`/`jmp` | A label statement and the forward jump to it. Both have a name, and both respect the scope rules of the statements section. |
| Pass | One in-place rewrite of a `CirModule`. A pass run returns `""` when clean, else the first refusal. |
| Refusal | The CIRv2 error channel: a pure string naming what went wrong, delivered as a compile-time error. No exceptions cross it. |

---

## Normal form

Normal form is an invariant, a means and never an end.
It buys exactly two things:

- a pass that needs a temporary hoists it into its own statement
- a printer walks a shape that it can assume

Shoggoth IR names the same instrument in two core rules:
statement-based control flow, and a conditionless `loop`.
Its hoisting rule states the payoff:

> evaluating any sub-expression in a new statement placed immediately in front of that statement is semantics-preserving

CIRv2 adopts that two-rule core and adds two rules,
the call-destination rule and the typed-op rule.
A macro-time compiler with seven printers needs the rules stated once in the tree, where every consumer reads them.
Four rules carry all the weight:

1. **Control flow is statement-based.** Branching is `sIte`, selection is `sCase`, iteration is
   `sLoop`, and `sScope` marks a fresh scope. No control flow hides inside an expression.
   The boolean connectives `eAnd`/`eOr` are pure bool operators over side-effect-free operands,
   and the verifier refuses a call under them.

2. **A `loop` has no condition slot.** `sLoop` has a body and an unroll hint, nothing else.
   Unrolling replicates the loop body inline at compile time.
   A `while` lowers to a conditionless loop whose body opens with the condition as a leading guard:

   ```
   while cond: body      ⇒      loop:
                                   <statements computing cond>
                                   ite (not cond): (then (jmp exit))
                                   body
                                lab exit
   ```

3. **Calls bind to their destination.** A value-producing call exists only as the whole right-hand
   side of `asgn` (`eCall`), and a void call exists only as a statement (`sCall`).
   No call appears in any other operand position.

4. **Every typed operation states its type.** Binary operators have `opType`, comparisons
   have `cmpType`, casts have `castType`, and constructors have their result type.
   All state it in the `(add Type a b)` discipline of the Leng grammar.
   A consumer never infers an operand type from context.

Together the axioms give one property, and every pass depends on it:

> **Insertion rule.** Evaluating a sub-expression in a new statement placed immediately in front
> of the statement that consumes it is semantics-preserving.

The verifier refuses any expression that hides a short-circuit region (axiom 1).
It refuses any condition slot outside the loop body (axiom 2), and any call in an operand position (axiom 3).
A pass that needs a temporary emits a statement in front of the consumer instead of fabricating a node inside the expression.

**Why calls bind to destinations.**
Lowering a call in an operand position hides statements inside an expression.
The result goes into a temporary that no reader can see.
A pass that needs a temporary then has no statement to carry it.

The frontend binds each call to a destination because the destination gives a later pass a statement,
and the front of that statement is a legal insertion point for every later pass.

Destination binding works by construction.
Each call has exactly one `asgn` statement as its destination.
While lowering, the frontend hoists nested call operands in front of the enclosing statement, so no consumer ever meets a nested call.

---

## Types

CIRv2 types use a closed set of type kinds, `CirTypeKind`:

```
ctVoid  ctBool  ctInt  ctUint  ctFloat  ctVector  ctPtr  ctAptr  ctStruct  ctArray  ctProc
```

Each type derives once at the frontend boundary, and no pass re-derives it.
`CirType` is a `ref object` with a case over the kind, so a type value is a shared reference.
A `ref object` is a shared, heap-allocated reference, like a C++ `shared_ptr`.

`withSpace(t, space)` returns a re-spaced copy of a pointer type with the pointee shared.
A caller never re-spaces a type in place.

| Kind | Fields | Rendering | `sameType` rule |
|---|---|---|---|
| `ctVoid` | (none) | `(void)` | identity |
| `ctBool` | (none) | `(bool)` | identity |
| `ctInt` | `bits` | `(i +32)` | same `bits` |
| `ctUint` | `bits` | `(u +32)` | same `bits` |
| `ctFloat` | `bits` | `(f +32)` | same `bits` |
| `ctVector` | `vN`, `vElem` | `(vector 4 (f +32))` | same length, same element type |
| `ctPtr` | `space`, `pElem` | `(ptr device (f +32))` | same space, same pointee |
| `ctAptr` | `space`, `pElem` | `(aptr device (f +32))` | same space, same pointee |
| `ctStruct` | `name`, `fields` | `:Name` | name only (nominal) |
| `ctArray` | `aN`, `aElem` | `(array (f +32) 4)` | same length, same element type |
| `ctProc` | `params`, `ret` | `(proctype (params ...) ret)` | same params, same return |

Notes:

- `bits` is 8, 16, 32 or 64, or `-1` for the machine word, the same convention as Leng's `IntBits`.
- `ptr` points to a single element and `aptr` points to an array of elements, the Leng distinction.
  The frontend stores the memory class of a pointer kernel parameter inside the `ptr`/`aptr` type,
  not beside the parameter.
- Structs are nominal: `sameType` compares two structs by name and every other pair of kinds
  field by field, recursively.
- Every `CirType` has `origin: NimNode`, a reference to the original Nim node kept so diagnostics
  can point back at the source, macro-time only, and nil in hand-built fixtures.
  `sameType` never reads it.

### Address spaces

`CirAddrSpace` is the memory class a pointer or an `aptr` addresses.
Its value order matches the compiler's own address-space enum, so the two convert by position:

| Space | Memory class | Carried by |
|---|---|---|
| `caThread` | per-thread registers | the unannotated default, plain locals |
| `caThreadgroup` | workgroup-shared storage | `{.smem.}` locals |
| `caConstant` | constant storage | `{.const_mem.}` locals |
| `caDevice` | device-global storage | buffer parameters and device pointers |

### Half-precision floats

No fp16 or bf16 kind exists in the set.
`float16` and `bfloat16` are `distinct uint16` types that keep raw bits, and conversion goes through `eCast` nodes.
Nim's `distinct` reuses the representation and inherits no operations.

Each printer prints the backend conversion in its own vocabulary.
On Metal, the printer renders the conversion as `as_type<ushort>(half(x))`.
The Metal compiler refuses an fp8 conversion at compile time.

---

## Expressions

Expressions are the evidence vocabulary, in 36 kinds, all with Leng names.

```
eNil eTrue eFalse eSym eIntLit eFloatLit eDeref eAt ePat eDot eAdd eSub eMul eDiv eMod
eShl eShr eBitAnd eBitOr eBitXor eBitNot eEq eNeq eLt eLe eNot eNeg eAnd eOr eCast eConv
eAddr eSizeof eOconstr eAconstr eCall
```

| Group | Kinds | Operand fields | Typing |
|---|---|---|---|
| Leaves | `eNil`, `eTrue`, `eFalse` | `nilType` | each literal states its type |
| Symbols | `eSym` | `name`, `typ`, `builtin`, plus the four typed builtin kinds | the symbol's declared type |
| Int literals | `eIntLit` | `value: int64`, `litType` | `litType` |
| Float literals | `eFloatLit` | `fValue: float64`, `fLitType` | `fLitType` |
| Binary arithmetic and bitwise | `eAdd` `eSub` `eMul` `eDiv` `eMod` `eShl` `eShr` `eBitAnd` `eBitOr` `eBitXor` | `opType`, `left`, `right` | `opType` |
| Unary | `eBitNot`, `eNeg` | `unType`, `unOperand` | `unType` |
| Logical negation | `eNot` | `notOperand` | `bool` |
| Comparisons | `eEq` `eNeq` `eLt` `eLe` | `cmpType`, `a`, `b` | state `cmpType`, yield `bool` |
| Boolean connectives | `eAnd`, `eOr` | `andLeft`, `andRight` | `bool`, side-effect-free operands only |
| Casts | `eCast`, `eConv` | `castType`, `castOperand` | `castType` |
| Address of | `eAddr` | `addrType`, `addrOperand` | `addrType`, operand must be an lvalue |
| Size of | `eSizeof` | `ofType` | `ctUint` with `bits = -1` (the machine word) |
| Deref | `eDeref` | `derefOperand` | the pointee type |
| Array indexing | `eAt` | `atBase`, `atIndex` | the array element type |
| Pointer indexing | `ePat` | `patBase`, `patIndex` | the pointee type |
| Field access | `eDot` | `dotBase`, `fieldName`, `inherit` | the field's type |
| Object constructor | `eOconstr` | `ocType`, `kvFields: seq[CirKv]` (`seq` is Nim's growable array) | `ocType` |
| Array constructor | `eAconstr` | `acType`, `elems` | `acType` |
| Call | `eCall` | `callee`, `args` | the callee's return type |

`CirKv` is one field binding, `name: string` plus `value: CirExpr`.

Notes:

- `(add Type a b)` discipline: the frontend stores the operand type in the binary node first.
  The renderer prints `(add (f +32) a b)`, and a consumer never infers types from context.
- Comparisons state their operand type and yield `bool`.
- `eAnd`/`eOr` are pure bool operators over side-effect-free operands.
  The frontend produces them only over such operands, and the verifier refuses a call
  under them: source `and`/`or` with side effects lowers to control flow instead.

- `eCall` exists only as the whole RHS of `asgn`. The verifier refuses it in any other position.
- `eSizeof` yields the machine word. GLSL has no `sizeof`, so the GLSL printer computes
  the std430 byte size (Vulkan's default buffer layout rule) instead, and the OpenCL printer
  prints C `sizeof`.
- `eDot` has `inherit`, Leng's inheritance number: 0 for a direct field, 1 for a field
  of the parent struct, and so on. CIRv2 keeps the field but locks it to 0 (verifier contract).

- Lvalues are exactly `eSym`, `eDeref`, `eAt`, `ePat` and `eDot` (`isLvalue`).
  An `asgn` left side and an `eAddr` operand must both be lvalues.
- `getExprType(e)` returns the type that the node's shape determines, and nil when the shape
  does not determine one. Nil arises for a deref of a non-pointer, an `at` of a non-array,
  and a `dot` that references no field.
  `eTrue`/`eFalse`, the comparisons, and `eSizeof` always produce a type.
- `getExprChildren(e)` returns one entry per operand position, so a pass walks children
  without a per-kind case.

A worked example traces the whole path.
The frontend lowers this kernel body:

```nim
proc saxpy(n: uint32, a: float32, x, y: ptr UncheckedArray[float32]) {.global.} =
  var i: uint32 = 0
  while i < n:
    y[i] = y[i] + a * x[i]
    i = i + 1
```

The lowering produces the CIRv2 normal form, and `cirv2_pretty` renders the proc:

```
(proc saxpy (param n (u +32)) (param a (f +32)) (param x (aptr device (f +32)))
            (param y (aptr device (f +32))) (void)
 (var i (u +32) 0)
 (loop
  (ite (not (lt (u +32) i n))
   (then
    (jmp exit)
    )
   )
  (asgn (pat y i) (add (f +32) (pat y i) (mul (f +32) a (pat x i))))
  (asgn i (add (u +32) i 1))
  )
 (lab exit)
 )
```

The trace, source line by source line:
`var i: uint32 = 0` renders as the `(var i (u +32) 0)` node.
`while i < n` lowers to the conditionless `loop` whose body opens with the leading-guard `ite`.
The guard tests the negated condition, and its then branch jumps to the `lab exit`.

The frontend lowers `y[i]` on a pointer parameter to the `pat` node.
The renderer prints the type on every operator, `(add (u +32) i 1)`.

---

## Statements

Statements carry the control flow in 15 kinds, all with Leng names.

| Kind | Fields | Semantics |
|---|---|---|
| `sVar` | `name`, `declType`, `init`, `space` | declares a local. `space` states the memory class of the declaration (`caThread` is the register default, `caThreadgroup` a `{.smem.}` local, `caConstant` a `{.const_mem.}` local). `const` ignores `space`. |
| `sConst` | `name`, `declType`, `init` | declares a named constant |
| `sAsgn` | `lhs`, `rhs` | assignment. `lhs` must be an lvalue (the lvalue set is in the expressions section), and a call may be the whole `rhs` |
| `sIte` | `cond`, `thenBody`, `elseBody` | if-then-else. `cond` is bool |
| `sLoop` | `loopBody`, `unroll` | conditionless loop. `unroll` marks a `{.unroll.}`-hinted loop: six of the seven printers emit a full-unroll directive in front of it (Metal `#pragma clang loop unroll(full)`, OpenCL `#pragma unroll`), and WGSL silently drops the hint |
| `sCase` | `selector`, `branches`, `caseElse` | switch over an int/uint selector. Branches carry disjoint inclusive literal ranges (`lo`/`hi` as `int64`) |
| `sLab` | `labName` | a label, proc-unique and targeted |
| `sJmp` | `jmpTarget` | a forward jump, scoped |
| `sScope` | `scopeBody` | a fresh scope, the destructor discipline (scopes mark where the target runs its cleanup code, such as destructors and releases) |
| `sAssume` | `assumeCond` | a statement the lowering vouches for |
| `sRet` | `retVal` | return, must match the proc return type |
| `sDiscard` | `discardVal` | discard |
| `sCall` | `callCallee`, `callArgs` | a void call in statement position |
| `sAsm` | `asmParts`, `asmOperands` | inline assembly: verbatim template text alternating with typed symbol operands |
| `sEmit` | `emitParts`, `emitOperands` | a `{.emit.}` statement: verbatim target-source text alternating with typed symbol operands |

`CirParam` is one proc parameter, `name: string` plus `typ: CirType`.

Notes:

- `sAsm` and `sEmit` interleave text and operands: `parts` has one element more than the operands.
  Piece `parts[i]` precedes operand `i`, the printer splices the operands' names,
  and the emitted text supplies its own terminators.
- The printers except WGSL emit a full-unroll directive in front of a `sLoop` that has
  `unroll = true`, and WGSL silently drops the hint.
- `sAssume` asserts nothing at run time.
  It records a condition that the lowering vouches for, and the verifier accepts comparisons
  or and/or trees over comparisons only.

### Control flow

**`ite`.** One condition, one then body, an optional else body.
Source `if`/`elif` chains lower flat over one shared merge label, and the structuring pass
rebuilds them into nested `ite` before printing.

**`loop`.** No condition slot (axiom 2 of the normal form).
Source `while` and `for` lower to a conditionless loop with a leading guard, and the guard's
then branch jumps to the loop's exit label.

**`case`.** A selector of int or uint type, and branches over inclusive literal ranges.
On `case`, the verifier refuses empty, overlapping, or negative ranges over an unsigned selector.
It refuses a selector of 63 or more unsigned bits, because the range literals are `int64`.

**`lab`/`jmp`.** Each branch body (an `ite` variant, a loop body, a `case` variant, a `scope`) is
a fresh scope.
The verifier tracks the scope chain as a sequence of scope ids (`ScopeChain = seq[int]`).

A `lab` at chain `L` is reachable from a `jmp` with chain `J` if and only if `L` is a prefix
of `J`.
That means the same scope or an enclosing scope.
Every `lab` is proc-unique, and every `jmp` targets a `lab` that appears textually later.

After structuring, the only remaining `lab`/`jmp` are a loop's continue and exit labels.
Each printer maps the exit-label `jmp` to `break`, the continue-label `jmp` to `continue`, and refuses any other label.

This label machinery is a cost of CIRv2, not a saving.
The v1 AST was jump-free: `gpuIf`/`gpuFor`/`gpuWhile` carried control flow directly, and a v1 printer never saw a label.

CIRv2 adds `lab`/`jmp`, the scope chain, the structuring pass, and the two-target condition lowering.
The reward is that passes reason over one loop shape with named joins.
A backend author pays for that with the mapping rules in this section.
v1 needed none of them.

CIRv2 has no `break` or `continue` node.
A loop exit is a `jmp` to the exit label, and a continue is a `jmp` to the continue label.
The frontend refuses a source-level `break`/`continue` outside the shapes that the two-target condition lowering produces.
The frontend boundary section explains the two-target condition compiler.

**`scope`.** `sScope` wraps a statement list in a fresh scope, the destructor discipline of the target lowering.
Scopes mark where the target runs its cleanup code, such as destructors and releases.
Scopes matter to the verifier because labels and jumps are scoped.

---

## Procs, kernels, modules

A `CirProc` has `name`, `params: seq[CirParam]`, `ret`, `body: seq[CirStmt]`, plus four flags:

| Flag | Source | Meaning |
|---|---|---|
| `isKernel` | `{.global.}` | a compute entry point. The printers emit the backend's kernel qualifier and its argument ABI for it |
| `forceInline` | `{.forceinline.}` | an inlining hint at emission. The inline-splice pass splices the call site before printing |
| `workgroup` | `{.workgroup(x, y, z)}` | the (x, y, z) workgroup count the host launches, baked into the shader for targets that compile it (Vulkan, WebGPU) |

The kernel entry-point name comes from the proc's name, and an `exportc` pragma on the proc overrides it.

A `CirModule` has `name`, `types: seq[CirType]`, `procs: seq[CirProc]`, and `globals: seq[CirStmt]`.
`globals` contains the program-scope declarations that a target lowering pass hoisted from the kernel body.
The pass populates it in kernel-body order, and the printer emits it before the proc set.
Every module a lowering pass has not touched keeps an empty `globals`.

---

## Builtin vocabulary

Four canonical builtin categories, each a closed enum with a compile-time name table:

| Category | Kind values | Canonical names |
|---|---|---|
| Coordinates (`CoordBuiltinKind`) | `cbkThreadPositionInGrid`, `cbkThreadgroupPositionInGrid`, `cbkThreadPositionInThreadgroup`, `cbkThreadsPerThreadgroup`, `cbkThreadgroupsPerGrid`, `cbkThreadIndexInThreadgroup` | `thread_position_in_grid`, `threadgroup_position_in_grid`, `thread_position_in_threadgroup`, `threads_per_threadgroup`, `threadgroups_per_grid`, `thread_index_in_threadgroup` |
| Synchronization (`SynchroBuiltinKind`) | `sbkThreadgroupBarrier`, `sbkThreadgroupBarrierDevice` | `threadgroup_barrier`, `threadgroup_barrier_device` |
| Subgroup reductions (`ReductionBuiltinKind`) | `rbkReduceAdd`, `rbkReduceMax` | `reduceAdd`, `reduceMax` |
| Atomics (`AtomicsBuiltinKind`) | `abkAtomicFetchAdd` | `atomicFetchAdd` |

Canonical names are MSL names.
The canonical names come from the public catalog `src/builtins/builtins_catalog.nim`.
I keep the MSL names because the canonical builtin catalog is MSL first, and every other backend aliases from it.
Per-backend alias templates expand during Nim semantic analysis, so the IR never stores an alias name.

Each enum has a `*None` value (`cbkNone`, `sbkNone`, `rbkNone`, `abkNone`).
Each category has one lookup proc, and each returns the `none` value for any other name:
(`coordBuiltinKind(name)`, `synchroBuiltinKind(name)`, `reductionBuiltinKind(name)`, `atomicsBuiltinKind(name)`).

A builtin outside the four categories (say `rsqrt`) keeps `none` in all four, and the printers
resolve its native name in their own tables.

Alias canonicalization fills the four typed builtin kinds of `eSym` from the name tables:
`coordBuiltin`, `synchroBuiltin`, `reductionBuiltin`, `atomicsBuiltin`.
Downstream code dispatches on the kinds, never on names, and no pass matches on names.

---

## Verifier

`cirv2_verify.nim` enforces the part of the node grammar that the tree shape cannot express.
Every pass wrapper runs `verify` (or `verifyAfter`) over its output.
A violation is a compile-time rejection through `macros.error`.
`verifyMessage` runs the same walk purely and returns the message.

Entry points:

- `verify(fn: CirProc)`, `verify(m: CirModule)`
- `verifyMessage(fn, passName = "")`, `verifyMessage(m: CirModule)`
- `verifyAfter(passName, fn)`

| Invariant | Contract |
|---|---|
| Types | the frontend stores the operand or result type of every operator, literal, cast, and constructor |
| Types | the two `asgn` sides carry types of the same kind |
| Types | the `asgn` lhs is an lvalue |
| Types | a `!=` against the zero u32 literal compares a bool (the WGSL bool launch ABI, explained under the contracts below) |
| Control flow | `jmp` is forward-only and scoped (target `lab` textually later, same or enclosing scope) |
| Control flow | every `lab` is proc-unique and targeted |
| Control flow | `ite` conditions are bool |
| Control flow | `case` ranges are disjoint literal ranges over an int/uint selector |
| Calls | `eCall` appears only as the whole RHS of `asgn` |
| Calls | `sCall` appears only as a void statement |
| Calls | a call never appears under `and`/`or` or in any other operand position |
| Returns | `ret` matches the proc return type |
| Assumptions | `assume` conditions are comparisons or and/or trees over comparisons |
| Structs | struct `dot` inheritance is 0 |

### Rejection messages

A rejection message names the construct, its node path, and the detail:

```
cirv2 verify: <construct> at <proc>/<path>: <detail>
[pass <name>] cirv2 verify: <construct> at <proc>/<path>: <detail>
```

Construct names lead with the Leng vocabulary (the renderer renders `eAdd` as `add`).
`emitTypeDbg` renders types for messages.
One worked example:

```
cirv2 verify: ite at saxpy/body[0]: condition is not bool
```

### Contracts the shape cannot carry

- **Address-space re-spacing at calls.** A pointer argument may hold another address space than
  the formal parameter states.
  The compiler re-spaces the argument at the call site (`callArgReSpace`, `retReSpace`).
  The pointee type must match exactly.
- **Literal width.** An `int64` literal must fit the bit width of the type it names
  (`litFitsBits`).
- **Case widths.** A selector of 63 or more unsigned bits refuses, because range literals are
  `int64`.
  Ranges must be non-empty, non-overlapping, and non-negative over an unsigned selector.
- **The WGSL bool launch ABI.** The launch ABI stores a bool kernel parameter as u32.
  The verifier accepts an `eNeq` over a bool lhs and the literal u32 zero.
  It refuses the same shape under the general comparison rules.

---

## Frontend boundary

`frontend/nim_to_crucible.nim` lowers typed Nim to CIRv2, normal form by construction.
Its entry is the typed node the `metal:`-family macros receive: a block of kernel proc definitions,
or the single `{.global.}` wrapper proc whose body calls the kernel (the `launchDeviceFn` shape).

```
typed NimNode ──lower──▶ CIRv2 normal form ──verify──▶ CirModule
```

Five guarantees hold by construction:

1. A value-producing call is an `asgn` RHS.
   Nested call operands hoist in front of the enclosing statement.
2. `while` and `for` lower to a conditionless `loop` with a leading guard.
3. `and`/`or` compile through the two-target condition compiler (Appel's `Cx` scheme),
   against a target label and a polarity: transfer to the target exactly when the condition is
   true or false, otherwise fall through (Andrew W. Appel, *Compiling with Continuations*, 1992).
   A condition in value position materializes into a bool temp through `ite`.

4. `if`/`elif` chains lay flat over one shared merge label.
5. Types translate exactly once and the translation is cached, so every occurrence of the same
   Nim type yields the same `CirType`.
   Struct names take a readable base name plus a 7-character base58 hash of the canonical
   type text (a compact alphanumeric encoding), and Nim `int`/`uint` lower to 32 bits.

The frontend refuses constructs outside the supported subset through the pure refusal channel (`lowerMessage`).
The macro wrapper turns a refusal into a compile-time `error`.

---

## Hand-building CIRv2

Tests, fixtures, and new passes build CIRv2 nodes directly.
Two modules support the work:

**Builders** (`cirv2_build.nim`) produce well-shaped nodes by construction:

- typed ops get their type from the caller,
- comparisons derive their operand type from the first operand,
- calls exist only as statements (`callS`) or destination-bound expressions (`callE`).

Anything the builders cannot express correctly is a verifier violation, and `cirv2_verify` catches it.

**The renderer** (`cirv2_pretty.nim`) prints one node per line, s-expression shaped, in the Leng
vocabulary (`(add Type a b)`, `(i +32)`, `aptr`, `ite`, `lab`/`jmp`).

`renderProc(fn)` renders one proc and `renderModule(m)` renders a whole module: nominal types,
then the program-scope globals when any were hoisted, then procs.
Tests diff rendered trees, and a hand-built fixture leaves `origin` nil.

---

*CIRv2 is the IR.
`DESIGN.md` beside this document is the compiler design: goals, pipeline, pass stages, backend contracts, and the refusal discipline.*
