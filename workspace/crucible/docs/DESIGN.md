# Crucible compiler design

> There are two ways of constructing a piece of software: one way is to make it so simple
> that there are obviously no deficiencies, and the other way is to make it so complex
> that there are obviously no deficiencies.
> -- C.A.R. Hoare, 1980 Turing Award lecture

Crucible is the GPU compile stack of this workspace: a family of Nim macros, one per target.
The targets are `metal:`, `cuda:`, `opencl:`, `glsl:`, `spirv:`, `webgpu:`, `hlsl:`.
Each macro compiles one kernel body written in Nim to one target source.

This document records the design: the goals, the pipeline, the pass stages, the backend contracts,
and the refusal discipline.
For the IR (intermediate representation) itself, see `cir_v2.md` beside this document.
This document stands alone.

---

## Goals

The goals are the ends, ranked as the design study ranked them.
The machinery that serves them is a means, listed after the table.

| # | Goal | What it buys |
|---|---|---|
| 1 | One body, seven targets | A Nim kernel compiles to Metal, CUDA C++, OpenCL C, GLSL 450, WGSL, HLSL and SPIR-V |
| 2 | SPIR-V printer small, passes easy to write | The SPIR-V printer was about 1,500 lines on CIRv1 and it must come down. Writing an optimization pass over the module must stay easy |
| 3 | Ingestion decouples from Nim | The Tile-IR discussion (MLIR-inspired): a frontend other than Nim can feed the IR, the Nim macro frontend is one producer among possible |
| 4 | Nim regenerates | The IR can re-emit Nim source for the CPU backend |

Two goals concern consumers from CIRv1 that CIRv2 does not carry yet.
One is the Lean verification backend.
One is the autodiff pass.
Rebuilding both is a goal, with maintenance as the measure.

Both read the same three answers off the tree:

- which assertions hold, through the `assume` channel
- which version of a value each statement reads
- where control flow rejoins after a merge

CIRv2 states each answer in the tree itself, so a rebuild consumes the tree
instead of re-deriving it. A new design-by-contract requirement takes the form
of tree data, and the grammar stays fixed.

### Means

| Means | Role |
|---|---|
| Normal form by construction | The frontend emits the shape invariant that every stage assumes, the normal form of `cir_v2.md` |
| Compile-time verification | The verifier rejects a malformed IR tree at the pass that produced it, before any source is printed |
| Refusals name the construct | A rejection message names the construct, the node path and the detail, in the Leng vocabulary |
| One rejection channel | A pass run returns `""` when clean, else the first refusal string, no exceptions cross the channel |
| Data-table printers | Type syntax, qualifiers, builtin names and literal suffixes are rows over the Leng vocabulary, one shared walk, the mechanism behind goal 4 |

## Non-goals

- **No run-time type information.** The type kinds are a closed set at macro time,
  and the target source sees only what that set states.
- **No exception handling.** Device targets have no unwinding. Failure is a compile-time refusal,
  never a run-time error path.
- **No generic dispatch below the frontend.** Nim semantics instantiate generics.
  The frontend receives fully typed trees.
  A shell, the residue that a generic proc instantiation leaves in the typed tree,
  is folded into the normalization stage or refused.

- **No dynamic dispatch.** The frontend stores the callee in the call node directly, and no node
  kind models an indirect call.
- **No backend-specific IR.** All seven targets read the same tree.
  Target-specific reshaping happens in lowering passes, never in a per-target IR dialect.
- **No silent fallback.** The compiler refuses an unsupported construct loudly at compile time,
  and wrong code never prints.

---

## Terminology

| Term | Meaning |
|---|---|
| Kernel | A compute entry point that the GPU host launches. In source it is a `{.global.}` proc (a proc is Nim's function, and `{. ... .}` braces mark Nim pragmas, annotations the compiler reads). |
| Device function | An ordinary proc compiled for the GPU, called from a kernel or from another device function. |
| `launchDeviceFn` shape | One of the two shapes a macro body takes: a block of kernel proc definitions, or a single `{.global.}` wrapper proc whose body calls the kernel. |
| Leng | Nimony's midlevel IR dialect. Nimony is the rewrite of the Nim compiler, and CIRv2 borrows Leng's node names, its `(add Type a b)` typing discipline, and its `ptr`/`aptr` distinction. |
| Workgroup | A group of threads that launch together, share memory, and synchronize with a barrier. Metal calls it a threadgroup, CUDA a block, OpenCL a work-group. |
| Address space | The memory class a pointer addresses: per-thread registers, workgroup-shared storage, constant storage, or device-global storage. |
| Push-block target | A target that packs all kernel parameters into one block passed as a unit, in host `store` packing order: GLSL, SPIR-V, and the HLSL Vulkan flavor. |
| Flat resource target | A target that binds each kernel parameter separately: Metal, WGSL, OpenCL, CUDA, and the HLSL D3D12 flavor. |
| Host `store` packing order | The argument order of the host-side `store` proc that serializes kernel arguments at launch, one per engine under `runtime/engines/`. It defines the parameter block layout. |
| Builtin categories | Four closed enums over the builtins the IR models natively: coordinates, synchronization, subgroup reductions, atomics. Canonical names are MSL names, and each builtin symbol carries one enum tag per category, the typed kinds the printers dispatch on. |
| Normal form | The tree shape that every producer guarantees and that every consumer assumes: statement-based control flow, a conditionless `loop`, calls bound to destinations, and fully typed operations. |
| Macro time | The phase of Nim compilation where macros run. CIRv2 nodes live only there. |
| Pass | One in-place rewrite of a `CirModule`, the IR module record with one proc per kernel or device function. A pass run returns `""` when clean, else the first refusal. |
| Refusal | The CIRv2 error channel: a pure string naming what went wrong, delivered as a compile-time error. |

---

## Pipeline

Its public API is `import workspace/crucible` plus one macro per target.
Each `codegen<Target>` macro runs:

```
lowerCirv2Unverified(body) ──▶ PassRegistry (stage order) ──▶ runPasses ──▶ GpuContext ──▶ printer
```

Registration runs in seven stages, in order (`codegen/gpu_compiler.nim`):

```
1. legalization     registerLegalizationPasses            builtin kind assignment, inline splice
2. normalization    registerNormalizationPasses           shell folding, structuring
3. optimization     registerOptimizationPasses            identity folding, dead locals
4. shared lowering  registerSharedLoweringPasses          target-independent struct-parameter policy
5. reserved rename  registerRenameReservedIdentifiers     per-target reserved-word table
6. target lowering  per-target registration               struct explosion, pointer dissolution, ...
7. validation       registerValidationPostPasses          the single verify pass
```

`backends_api.nim` then dispatches: `codegen` prints the module for a static `Backend`
(CUDA, WebGPU, OpenCL, GLSL, Metal, HLSL) by calling the matching `*_lang.codegen(ctx)`,
`codegenSpirvShader` returns the binary SPIR-V words.

### Pass machinery

`codegen/passes/pass_datatypes.nim` and `pass_registry.nim` define the machinery:

- `CirPass` is a `ref object` with `name`, `kind`, `phase`, `description`, `dependsOn: seq[string]`,
  and `run: proc(m: CirModule): string`.
  A pass run takes the module in place and returns its refusal: `""` when clean, else the first refusal.
- `PassKind` classifies the pass: `pkValidation` (check-only), `pkTransform` (changes the IR),
  `pkAnalysis` (computes metadata, changes nothing).
- `PassPhase` orders the pass inside a stage: `phaseEarly` (right after IR construction),
  `phasePreprocessing`, `phaseMain` (before printing).
- `PassRegistry` keeps `passes` and `donePasses`, and `runPasses` executes the registered passes
  in registration order, which is the stage order by construction.

`CirModule` is the IR module record, one proc per kernel or device function.
The datatypes live in `cir_v2.md`.

Two registration orders matter, and the contract fixes both of them:

- Builtin kind assignment runs first, because the inline splice's kernel-only-reads check keys
  on the typed kinds.
- Folding runs before dead-local elimination, because the eliminations that folding enables
  cascade into the dead-local pass.

Per-target registration notes:

- Metal, OpenCL, GLSL and WGSL register the reserved-word rename with their own reserved table.
  GLSL registers it before the GLSL lowering passes, so the relocated program-scope declarations
  reference the renamed names.
- WGSL renames with the prefix `wgsl_`, because WGSL reserves the `__` identifier head.
- HLSL registers the rename inside `registerHlslLoweringPasses` (`hlslReservedIdentifiers`).

- CUDA registers the shared passes plus validation only.
  The rename registration inside `registerCudaLoweringPasses` is unwired, so the `cuda:` pipeline performs no reserved-word rename.
  `CudaReserved` names the four execution-space qualifiers the CUDA-C++ emitter writes:
  `__global__`, `__device__`, `__shared__`, `__constant__`.
- SPIR-V registers no rename.

---

## Stages

One section per stage, one bullet per pass.
Every pass pairs its job with the refusal it raises.

### Legalization

`codegen/passes/passes_legalizations.nim`: conforms the IR to the model that every later stage assumes. It is target-independent.

- **Builtin kind assignment** (`passes_norm_aliases.nim`) maps the canonical MSL name
  of a builtin symbol to its typed kind, so the printers dispatch on the typed kinds alone and no
  name recognition survives downstream.
- **Inline splice** (`passes_legal_inline.nim`) splices `{.inline.}`/`{.forceinline.}` call sites
  into their caller. A device function cannot take the thread builtins as attributed parameters,
  because thread coordinates are meaningful only inside a kernel, so an `{.inline.}` proc
  that reads them splices into its caller.

Per site, the arguments bind to fresh temps, the callee locals and labels rename fresh,
`result` maps to the call's destination, and a trailing `ret` moves its value into the destination.
The splice refuses a mid-body `ret`.

### Normalization

`codegen/passes/passes_normalizations.nim`: canonical reshaping.
It establishes the forms that every later stage assumes.

- **Shell folding** (`passes_norm_shell_fold.nim`). A shell is what Nim's type checking leaves
  of a generic proc that Nim instantiates with only static arguments (compile-time values, like
  C++ `template<int N>`), and the workspace's tile-layout library crosses the kernel boundary
  through such shells, and it instantiates generic procs with compile-time shape constants.

The pass substitutes the resolved constant at the shell call site and deletes the dead shell definitions.
Detection is syntactic: single-assignment locals, no control flow, no calls,
and resolved returns of literals, wrappers, or constant constructors.

- **Structuring** (`passes_norm_structure.nim`) compiles the frontend's flat `lab`/`jmp`
  condition code back into structured statements:

  ```
  flat lab/jmp list ──structure──▶ nested ite/loop + break/continue jmps
  ```

  Source `and`/`or` compile through the two-target condition compiler (Appel's `Cx` scheme).
  A condition compiles against a target label and a polarity: transfer to the target exactly
  when the condition is true, otherwise fall through. The layout encodes the short-circuit directly.

  Structuring reads that flat form back.
  A test `ite(c, [jmp A], [jmp B])` is a `Cx` test.
  An `elif` chain is a row of tests over the shared merge label. Block-exit `jmp`s and the break
  form complete the shapes.

  Structuring refuses anything else loudly.
  Output contract: the only remaining `lab`/`jmp` are a loop's continue and exit labels.

### Optimization

`codegen/passes/passes_optimizations.nim`:

- **`foldArithmeticIdentities`** collapses mul/div by literal 1 and add/sub with literal 0
  to the surviving operand, in one bottom-up walk.
- **`removeDeadLocals`** deletes a local that nothing reads afterwards, so its type never reaches
  the printer.
- Folding runs first: the eliminations that it enables cascade into the dead-local pass.

### Shared lowering

`codegen/passes/passes_lowering.nim`. The shared stage registers no pass.
Shared contract: struct kernel parameters stay whole, and the per-target stage decides the layout.

Push-block targets (GLSL, HLSL, SPIR-V) pack them into the block, and flat resource targets
register the explosion per target.
The block is a push constant, a small parameter block the host uploads with each kernel launch.

### Reserved-word rename

`codegen/passes/passes_low_reserved.nim`. The rename pass gives an identifier that collides with a target reserved word
the `_nim` suffix at every emitted site, so the printers never rename.

It covers proc names, call sites, parameters, locals, struct names and fields, and object-constructor field names.
A local and a proc that share a name are one symbol after the frontend's flat lowering, and the rename pass renames them together.
WGSL additionally prefixes every `__`-prefixed name with `wgsl_`.

### Per-target lowering

`codegen/passes/passes_low_kernel_params.nim` and `codegen/passes/passes_lowering_<target>.nim`:

- **`explodeKernelStructs`** (WGSL, OpenCL, Metal) expands a struct kernel parameter with a (nested)
  pointer field into flat ptr/scalar leaf parameters, one buffer slot per leaf in field order,
  and the kernel body opens with a local rebuilt from the leaves.
  The pass keeps pure-value structs whole. It refuses a leaf-name collision and body locals named like a leaf.
- **GLSL.** `glslDissolvePointerLocals` dissolves pointer locals into their pointee,
  because GLSL has no function-storage addressing and a local cannot be addressed.
  `glslHoistProgramScopeLocals` relocates `{.smem.}`/`{.const_mem.}` locals
  into `CirModule.globals` in encounter order.
- **HLSL.** `hlslMaterializeNumWorkgroups` rewrites `threadgroups_per_grid` reads to a trailing
  kernel parameter named `crucible_numworkgroups` (HLSL has no shader-model semantic for the grid size).
  `hlslDissolvePointerLocals` dissolves pointer locals and pointer-holding struct locals,
  and reads and writes a `var T` parameter through its `inout` binding.
- **MSL.** `registerMslLoweringPasses` registers `explodeKernelStructs` as well.
  `mslHoistProgramScopeConstants` and `mslRewriteSmemLocals` run after it: a kernel's top-level
  `{.smem.}` local stays in-body as kernel-scope `threadgroup` storage.

- **OpenCL.** `openclHoistProgramScopeConstants` hoists program-scope constants.
  `lowerByrefParams` passes large structs to device functions as hidden thread pointers.
  The pass leaves `{.smem.}` locals untouched, and the printer emits `__local` directly.
- **SPIR-V.** `spirvPatchBoolToI32`: the 4-byte push-constant convention has no bool member.
  The pass retypes module-level bool storage and kernel-boundary bool symbols to i32
  before printing. It wraps reads in `conv(bool)` and writes in `conv(i32)`.
  Device functions stay untouched.
- **WGSL.** `explodeKernelStructs` → `wgslDissolvePointerLocals` → `wgslRewriteBoolKernelParamReads` (the pass rewrites the kernel's bool parameter reads to `name != 0u`) → `wgslHoistProgramScopeLocals`.
  Registration contract: the passes sit after the reserved-word rename.

Per-target chains, as the module docs record them:

```
GLSL:   kernel body → glslDissolvePointerLocals → glslHoistProgramScopeLocals → printed GLSL source
MSL:    kernel body → explodeKernelStructs → mslHoistProgramScopeConstants → mslRewriteSmemLocals → printed MSL source
OpenCL: kernel body → openclHoistProgramScopeConstants → printed OpenCL C source
WGSL:   kernel body → explodeKernelStructs → wgslDissolvePointerLocals
                    → wgslRewriteBoolKernelParamReads → wgslHoistProgramScopeLocals → printed WGSL source
```

### A traced kernel

The stages act on one concrete kernel.
The frontend lowers this `metal:` kernel body to the CIRv2 normal form:

```nim
proc axpy(x, y: ptr UncheckedArray[float32], alpha: float32) {.global.} =
  var device: float32 = 0.0
  var sum {.smem.}: float32 = 0.0
  device = alpha
  sum = sum + device
  y[0] = x[0] + sum
```

```
(proc axpy (param x (aptr device (f +32))) (param y (aptr device (f +32)))
           (param alpha (f +32)) (void)
 (var device (f +32) 0.0)
 (var threadgroup sum (f +32) 0.0)
 (asgn device alpha)
 (asgn sum (add (f +32) sum device))
 (asgn (pat y 0) (add (f +32) (pat x 0) sum))
 )
```

- Reserved rename (stage 5): the local `device` collides with the MSL `device` address-space qualifier,
  so the rename pass rewrites it to `device_nim` at the declaration and at every use.
- Target lowering (stage 6, Metal): `explodeKernelStructs` finds no struct parameter,
  `mslHoistProgramScopeConstants` finds no `{.const_mem.}` local, and `mslRewriteSmemLocals` keeps
  the top-level `{.smem.}` local in the kernel body, discarding its initializer.
- Validation (stage 7): the verify pass checks the normal form and accepts it.
- The MSL printer emits the kernel.
  Scalars print as `constant T&`, and pointers take one buffer slot per parameter in order.

The printed Metal source, abbreviated:

```c
kernel void axpy(device float* x [[buffer(0)]], device float* y [[buffer(1)]],
                 constant float& alpha) {
  float device_nim = 0.0f;
  threadgroup float sum;
  device_nim = alpha;
  sum = sum + device_nim;
  y[0] = x[0] + sum;
}
```

### Validation

`codegen/passes/passes_validations.nim`.
The verifier registers as the single validation pass (`pkValidation`), and post-validation runs
after the transforms, right before printing.

### Pass inventory

| Stage | Pass | Job | File |
|---|---|---|---|
| Legalization | builtin kind assignment | canonical MSL name → typed kind | `passes_norm_aliases.nim` |
| Legalization | inline splice | splice inline call sites | `passes_legal_inline.nim` |
| Normalization | shell folding | substitute resolved shells, delete dead shells | `passes_norm_shell_fold.nim` |
| Normalization | structuring | flat lab/jmp → structured statements | `passes_norm_structure.nim` |
| Optimization | foldArithmeticIdentities | identity folding | `passes_optimizations.nim` |
| Optimization | removeDeadLocals | delete unread locals | `passes_optimizations.nim` |
| Shared lowering | (none) | struct-parameter policy | `passes_lowering.nim` |
| Reserved rename | renameReservedIdentifiers | `_nim` suffix on collisions | `passes_low_reserved.nim` |
| Target lowering | explodeKernelStructs | struct kernel params → flat leaves | `passes_low_kernel_params.nim` |
| Target lowering | per-target passes | pointer dissolution, hoisting, bool retyping | `passes_lowering_<target>.nim` |
| Validation | verify | the normal-form contract | `passes_validations.nim` |

All pass files live under `workspace/crucible/src/codegen/passes/`.

---

## Why this shape

### A destination for every call

Lowering a call in an operand position hides statements inside an expression, and the result goes
into a temporary that no reader can see.
Without a destination binding, every pass that needs a temporary must fabricate a statement
inside the expression, and a repair pass must flatten it afterward.

Destination binding removes the round trip by construction.
The frontend binds each call to one `asgn` statement.

The frontend hoists nested call operands while it lowers.
The front of the destination statement is a legal insertion point for every later pass.

### Statement-based control flow before optimization

Structuring runs in the normalization stage because it establishes the forms that every later stage assumes.
After structuring, every pass reasons over `ite`/`loop`/`case`, and the only jumps left are the loop's continue and exit labels.

Optimization over flat `lab`/`jmp` lists would put reachability reasoning in every pass.
Optimization over structured statements keeps it in the verifier, which already checks that every
label is proc-unique, targeted, and reachable only through the scope-chain prefix rule.

The benefit is real, and so is the cost.
The structuring pass, the lab/jmp refusal rules, and the two-target condition lowering are machinery v1 did not have.

A v1 pass reasoned over `gpuIf`/`gpuFor`/`gpuWhile` with no labels in the tree.
CIRv2 trades a bigger node set and label discipline for the single loop shape.
This document counts both sides of that trade.

### Why kind assignment runs first

Inline splicing decides per call site whether an `{.inline.}` callee reads kernel-only builtins.
That check keys on the typed builtin kinds, not on names.
Kind assignment therefore runs before the splice, and the splice sees typed kinds.

### Why the verifier runs per pass and last

Every pass wrapper runs `verify` (or `verifyAfter`) over its output.
A violation is a compile-time rejection at the pass that produced it, not at the pass that trips over it.
Both checks run the same walk: the per-pass wrapper checks each pass's output, and the single
registered validation pass runs the walk once more over the final tree, right before printing.

### Why the rename precedes the per-target lowering

Renaming runs before the per-target lowering so every declaration a lowering pass relocates
already has its final name: GLSL registers the rename before its lowering passes so the relocated
program-scope declarations reference the renamed names, and the WGSL lowering passes sit after
the rename by contract.

### Why the type kinds are a closed set

Seven backends dispatch over the type kinds.

The set stays closed because seven backends dispatch over it: an open set
would push type decisions into string handling in seven places, and a closed one turns a missing
case into a visible gap in one place.

Today that dispatch is case functions in each printer.
Only the SPIR-V encoder is a literal table.
The emit-engine work turns the per-printer dispatch into real tables.
Until then, "table-driven" describes the intent, not the code.

One further ruling keeps fp16 and bfloat16 out of the type kinds.
The backends disagree only on the conversion syntax, and the bit payload is the same.
The IR therefore stores the raw bits in a `distinct uint16`, and each printer emits its own conversion.

---

## Backends

The kernel entry-point name comes from the proc's name, and an `exportc` pragma on the proc overrides it.

Common contract, every printer:

- Table-driven over the closed set of type kinds, and builtins dispatch on the typed kinds.
- A `jmp` to the loop exit label prints as `break`, and a `jmp` to the continue label
  prints as `continue`, and any other `lab`/`jmp` refuses.
- Six of the seven printers emit a full-unroll directive in front of an `unroll` loop:
  Metal `#pragma clang loop unroll(full)`, OpenCL `#pragma unroll`.
  WGSL silently drops the hint.

| Target | Kernel signature | Parameter model | Pointer support |
|---|---|---|---|
| Metal (MSL) | one buffer slot per kernel parameter in order | pointers `device T* name [[buffer(n)]]` (the resource slot index the host assigns), scalars unattributed `constant T& name`, coordinate builtins attributed `uint3`/`uint` params with no slot | device pointers, `&x`/`*p` native |
| CUDA C++ | `extern "C" __global__ void name(args)`, host looks up by unmangled name | plain C++ parameters, one generic address space | generic pointers, `__shared__`/`__constant__` for placed locals |
| OpenCL C | `__kernel void name(...)`, every kernel in encounter order, one program per module | device pointers `__global T* restrict`, large by-ref structs as hidden thread pointers | `(&x)`/`(*p)` address and deref locals |
| GLSL 450 (Vulkan compute) | `void <name>()`, one kernel per module | one push-constant block member per parameter in host `store` packing order | buffer-reference handles (`GL_EXT_buffer_reference`), element access `<handle>.x[index]` |
| WGSL | one kernel per module, signature takes only `@builtin(...)` coordinate parameters in first-use order | one `@group(0) @binding(i)` per parameter (the resource slot pair the host assigns), output first, scalars and value structs packed into the 256-byte-aligned scratch buffer | device buffers `var<storage, read_write> name: array<T>` |
| HLSL (DXC) | one kernel per module, emitted as `main` | fvD3D12: `StructuredBuffer`/`RWStructuredBuffer` on one global register ordinal (`t` vs `u`, see the write-scan note below), scalars and value structs in one `ConstantBuffer<Push> push : register(bN, space0)`, fvVulkan: one `[[vk::push_constant]]` struct, buffers as `uint64_t` device addresses with `[[vk::offset]]`-locked offsets | fvD3D12 structured buffers, fvVulkan constructs `vk::BufferPointer<E>` per use |
| SPIR-V | one kernel per module, Vulkan 1.4 environment on the encoder's SPIR-V 1.3 baseline | one push-constant block binding kernel parameters in host `store` packing order | buffer parameters contribute one u64 device address each, converted to physical storage buffer pointers in the kernel prologue |

### Printer size

All seven printers grew between the v1 and v2 eras.
This table measures the growth in source lines:

| Printer | v1 lines | v2 lines |
|---|---|---|
| Metal (`msl_lang.nim`) | 560 | 665 |
| WGSL (`wgsl_lang.nim`) | 546 | 901 |
| OpenCL (`opencl_lang.nim`) | 558 | 820 |
| CUDA (`cuda_lang.nim`) | 497 | 683 |
| GLSL (`glsl_lang.nim`) | 725 | 1024 |
| HLSL (`hlsl_lang.nim`) | 873 | 1429 |
| SPIR-V (`spirv_shader_lang.nim` + `spirv_encoder.nim`) | 2151 | 2264 |
| Total | 5,910 | 7,786 |

New capability drives the growth.
HLSL gained the fvVulkan route, and GLSL gained buffer-reference handles plus std430 sizing.

Some semantics remain printer-resident:

- the HLSL printer keeps `classifyRwBuffers` and `scanCallStmts`, about 250 lines
- the GLSL printer keeps `std430Size` and `std430Align`, about 80 lines

I treat the extraction of that resident semantics into passes as the tracked path to shady-style small printers.

### Metal (`backends/msl_lang.nim`)

- The MSL printer dispatches coordinate builtins to its vocabulary through the typed kinds.
- The printer keeps `{.smem.}` locals in-body as kernel-scope `threadgroup` storage and discards their initializer.
- The printer prints `{.const_mem.}` locals as program-scope `constant` through `CirModule.globals`.
- The IR stores fp16/bf16 as `distinct uint16` raw bits, and the printer renders the conversion
  as `as_type<ushort>(half(x))`, while the Metal compiler refuses fp8 conversions at compile time.

### CUDA (`backends/cuda_lang.nim`)

- The CUDA printer emits `extern "C" __global__` for kernels, so the host looks them up by unmangled name,
  and `__device__` for device functions keeping their C++ mangled names.
- One generic address space: the printer prints `{.smem.}` as `__shared__` and program-scope
  globals as `__constant__`.
- The CUDA printer maps coordinate builtins to `blockIdx`/`blockDim`/`gridDim`/`threadIdx`.
  The canonical Nim type is `int` while the CUDA builtins are unsigned, so the printer casts
  reads to `int`.

- The CUDA printer emits a butterfly template at module scope on first use for warp reductions.
- The CUDA printer prints `#pragma unroll` for an `{.unroll.}` loop.

### OpenCL C (`backends/opencl_lang.nim`)

- The OpenCL printer prints every kernel in encounter order, one program per module.
- The OpenCL printer prints C `sizeof` for `eSizeof`, and addresses and derefs locals with `(&x)`/`(*p)`.
- The OpenCL index builtins are `size_t`, so the printer casts reads to `int`, and prints
  the x-major flat thread index parenthesized against mis-association.

### GLSL (`backends/glsl_lang.nim`, GLSL 450 Vulkan compute)

- The GLSL printer emits one `{.global.}` kernel per module, signature `void <name>()`.
- The GLSL printer prints kernel parameters as one push-constant block, one member per parameter
  in host `store` packing order.
- The GLSL printer renders buffer pointers as buffer-reference handles
  (`GL_EXT_buffer_reference`), and renders element access as `<handle>.x[index]`.
- The GLSL printer computes the std430 byte size (Vulkan's default buffer layout rule) for `eSizeof`,
  because GLSL has no `sizeof`.

### WGSL (`backends/wgsl_lang.nim`)

- The WGSL printer emits one `@group(0) @binding(i)` per parameter, output first.
- The WGSL printer prints device buffers as `var<storage, read_write> name: array<T>`.
- The WGSL printer packs scalars and value structs into the 256-byte-aligned scratch buffer,
  as `var<storage, read>`.

- The WGSL printer declares bool scalars as u32, and the lowering pass has already rewritten
  the body reads to `name != 0u`.
- The WGSL printer emits a kernel signature that takes only `@builtin(...)` coordinate
  parameters in first-use order.
- The WGSL printer emits struct declarations for referenced value structs only.

### HLSL (`backends/hlsl_lang.nim`, DXC)

- The HLSL printer emits one kernel per module, as `main`.
- Two resource-binding flavors share one language body.
- The printer's `fvD3D12` flavor (default) binds structured buffers on one global register
  ordinal and scalars in one `ConstantBuffer<Push>`.

- The write scan is conservative: it counts a buffer as read-write when the kernel or any device
  function stores through it, directly or transitively over the call graph.
- The printer's `fvVulkan` flavor packs everything into one `[[vk::push_constant]]` struct,
  with buffer parameters as `uint64_t` device addresses and `[[vk::offset]]`-locked offsets,
  and constructs `vk::BufferPointer<E>` per buffer use.
- The printer drops program-scope constants (`hlslHoistProgramScopeConstants`), which dxc folds
  at compile time (dxc is the DirectX shader compiler), so no register appears in the constant-buffer
  (`$Globals`/cbuffer) layout.
- The printer refuses a `threadgroups_per_grid` read.
  The lowering pass materializes it as a trailing kernel parameter.

### SPIR-V (`backends/spirv_shader_lang.nim` + `spirv_encoder.nim`)

- One kernel per module, Vulkan 1.4 environment on the encoder's SPIR-V 1.3 baseline.
- The SPIR-V printer binds kernel parameters in order (host `store` packing order) through one
  push-constant block, and the printer gives each buffer parameter one u64 device address,
  converted to physical storage buffer pointers in the kernel prologue.
- Opcode, capability, and execution-mode numbers are SPIR-V core grammar values.
  The runtime module scanner shares them: one source of truth.

---

## Refusal discipline

Four channels own the rejections, and a refusal always names what went wrong:

| Channel | Owner | Trigger | Delivery |
|---|---|---|---|
| Frontend refusal | `frontend/nim_to_crucible.nim` | a construct outside the supported subset | `lowerMessage`, and the macro wrapper turns it into a compile-time `error` |
| Pass refusal | each pass | the first violation a pass run hits | `""` when clean, else the first refusal, pure strings, no exceptions |
| Verifier rejection | `ir/cirv2_verify.nim` | a normal-form violation | `cirv2 verify: <construct> at <proc>/<path>: <detail>` through `macros.error` |
| Printer refusal | each `backends/*_lang.nim` | a construct the target cannot express | a refusal naming the construct |

Every forbidden construct pairs with the refusal that enforces it:

| Forbidden construct | Enforcing refusal |
|---|---|
| a call in an operand position or under `and`/`or` | verifier: a call never appears in any operand position |
| a backward or cross-scope `jmp` | verifier: `jmp` is forward-only and scoped |
| a source-level `break`/`continue` outside the lowering shapes | the frontend refuses loudly |
| a mid-body `ret` in an inlined callee | the inline splice refuses |
| an unstructured `lab`/`jmp` after structuring | structuring refuses loudly, and printers refuse any label beyond the loop's continue and exit labels |
| a case selector of 63 or more unsigned bits | verifier: range literals are `int64` |
| an `int64` literal that does not fit its type | verifier: `litFitsBits` |
| a struct `dot` with inheritance | verifier: struct `dot` inheritance is 0 |
| pointer locals where the target cannot address them | the target's dissolve pass dissolves them or refuses what it cannot dissolve |
| a kernel struct leaf name collision | `explodeKernelStructs` refuses |
| a builtin name outside the four categories | keeps `none` in all four kinds, and the printer's own table resolves the native name |

---

## Status

Labels are explicit: **implemented** (the printer emits correct source for the recorded runs),
**gated green** (a recorded run passes), **refused loudly** (a documented compile-time refusal).
A gate is a recorded test or benchmark run, the gated green label means the run passed, and each name
names its test binary under `tests/`.

Recorded runs (2026-10-30, CIRv2 review-fix branch):

| Gate | State |
|---|---|
| Metal GEMM (general matrix multiplication, `t_metal_gemm`) | gated green |
| Metal axpy (`t_metal_axpy_view`) | gated green |
| SDPA (scaled dot-product attention) bench f32 | gated green, 1.85 peak TFLOP/s |
| SDPA bench bf16 | gated green, 7.35 peak TFLOP/s (2048, causal=false) |
| rmsnorm (root-mean-square normalization) Metal full suite | gated green, 1145 case-runs, worst deviation 0 |
| OpenCL axpy (`t_opencl_add`, `t_opencl_axpy_view`) | gated green |
| SPIR-V axpy (`t_spirv_axpy`, `spirv-val` + MoltenVK run, `t_vulkan_axpy_view`) | gated green |
| WebGPU axpy | refused loudly, documented, the printer lacks the fp16 conversion chain |
| GEMM bench, 4096³ | 14.28 TFLOP/s (MLX 14.63, NAX 14.19), no regression |

MLX is Apple's machine-learning framework, and NAX is Apple's cooperative-tensor matmul kernel,
both reference points on the same machine.

Per-target status:

| Target | Status |
|---|---|
| Metal | implemented, gated green |
| CUDA | implemented (NVRTC-compilable CUDA C++ contract), runs recorded on the CUDA workstation |
| OpenCL | implemented, gated green |
| GLSL | implemented |
| HLSL | implemented, both flavors (the call-graph read/write scan is dxc-verified) |
| SPIR-V | implemented, gated green (`spirv-val` + MoltenVK run) |
| WGSL | printer implemented, the `webgpu:` macro refuses loudly on kernels needing the fp16 conversion chain, pending the fp16 port |

*For the IR details, the normal form, the node taxonomy, the verifier contract, and the builtin vocabulary,
see `cir_v2.md`.*
