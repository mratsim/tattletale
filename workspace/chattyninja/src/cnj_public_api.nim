# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Chattyninja public API. This module IS the package surface, holding every
## name a consumer may write and nothing else.
##
## - the render tier, `parseJinjaTemplate`, `startJinjaRender`, `pullInto`,
##   `renderToString`, over the compiled artifact and the caller's `JinjaRenderContext`
## - the data tier, the value constructors and `JinjaError`, the whole-mapping
##   dict form `dictVal` over key/value pairs, the `tojson` form `toJson`
## - the types the public signatures name and the depth, nesting, step, element caps
##
## Engine-internal by design, never exported:
## - the mutable-mapping dict form, templates reaching `namespace()` through
##   the engine's global machinery
## - `dictGet`/`dictSet`, `eqVal`, `ValueKind` and the error cause, callers
##   building whole values inline through the constructors
## - mappings are never changed after construction, the whole mapping built up front
##
## - each public proc delegates to the engine module that owns the implementation
## - the caller contracts live on these delegates
## - engine internals stay reachable only through `import x {.all.}`

import ./cnj_types {.all.}
import ./cnj_parse {.all.}
import ./cnj_engine {.all.}
import ./jinja_data_model {.all.}
import ./jinja_serialize {.all.}

# Types, knobs and error exports the public signatures name.
export cnj_types.CompiledTemplate, cnj_types.CompiledSymbols,
    cnj_types.JinjaRenderContext,
    cnj_types.TTT_CNJ_MacroDepthCap, cnj_types.TTT_CNJ_ExprDepthCap,
    cnj_types.TTT_CNJ_ParseNestingCap, cnj_types.TTT_CNJ_StepBudget,
    jinja_data_model.JinjaVal, jinja_data_model.JinjaError,
    jinja_data_model.TTT_CNJ_RangeElemCap, jinja_data_model.TTT_CNJ_ValueDepthCap

# Render tier, parse once and render over the shared artifact.

proc parseJinjaTemplate*(src: string): (CompiledTemplate, CompiledSymbols) =
  ## Returns the compiled template artifact plus its `CompiledSymbols`:
  ## - interned names build in parse order, read-only at render
  ## - the interned-name table is a heap object the parse allocates once, returned by ref,
  ##   every render over the artifact sharing it
  ## - the template borrows `src`, so it must not outlive the caller's text
  cnj_parse.parseJinjaTemplate(src)

proc startJinjaRender*(tmpl: CompiledTemplate, sym: CompiledSymbols, root: JinjaVal, clock = 0.0): JinjaRenderContext =
  ## Returns a render context over the shared artifact, ready to render `root`, the render
  ## context dict with `messages`, `tools`, `add_generation_prompt` and template kwargs.
  ##
  ## Contract:
  ## - `clock` is the epoch `strftime_now` reads, never artifact state, so one artifact
  ##   renders reproducibly under different clocks
  ## - `sym` is the parse-built interned-name table by ref, every render over the artifact
  ##   holding the same heap object, no borrow contract
  cnj_engine.startJinjaRender(tmpl, sym, root, clock)

proc pullInto*(c: JinjaRenderContext, buf: var openArray[char]): int =
  ## Returns the render's next bytes, written into `buf[0 ..< result]`.
  ##
  ## Ownership sits with the caller, whose buffer capacity is the delivery window.
  ## Resumption state is `c.state`, so consumers over one artifact each hold a context
  ## from `startJinjaRender` and own their delivery position.
  ##
  ## Delivery contract:
  ## - `c.state.pend.pos` and `c.state.cur` advance before the call returns, so a consumer
  ##   that stops mid-drain and resumes never re-receives a byte
  ## - a piece longer than the window drains across calls, a lazy piece resuming
  ##   through the serializer in `c.state.lazy`
  ##
  ## Termination and budget:
  ## - 0 means the render is complete, nothing pending and `c.state.curNode == NoLink`
  ## - one call dispatches at most `TTT_CNJ_StepBudget` steps, a breach raising located at the reached node
  ##
  ## A raise discards the bytes already written into `buf` in the failing call, the caller
  ## never receiving them and the render state having advanced past their render, so a repull
  ## resumes after them:
  ## - a consumer that must hold every byte across a raise keeps the window at one byte,
  ##   which makes each delivered byte a returned byte
  ## - span pieces copy out of `CompiledTemplate.jinja`, string pieces and cut pieces copy
  ##   out of render-state storage, lazy pieces out of the serializer state in `c.state.lazy`
  ## - a zero-capacity buffer returns 0 without stepping the render
  cnj_engine.pullInto(c, buf)

proc renderToString*(src: string, root: JinjaVal, clock = 0.0): string =
  ## Returns the whole render of `src` over the value `root`, compiling and rendering in one call.
  ##
  ## - compiling happens at the scope that owns `src`, the artifact borrowing the template
  ##   text and never outliving it
  ## - the one-shot entry owns its drain, the render pulled through a stack window
  ##   that is drained until it reports 0
  cnj_engine.renderToString(src, root, clock)

# Data tier, values, their construction and the `tojson` form.

func undefinedVal*(): JinjaVal =
  ## Returns the absent-binding value, rendering empty, failing truthiness, equaling only itself.
  jinja_data_model.undefinedVal()

func noneVal*(): JinjaVal =
  ## Returns Python's `None`, rendering `None`, failing truthiness, distinct from undefined.
  jinja_data_model.noneVal()

func boolVal*(b: bool): JinjaVal = jinja_data_model.boolVal(b)
func intVal*(i: int64): JinjaVal = jinja_data_model.intVal(i)
func intVal*(i: int): JinjaVal = jinja_data_model.intVal(i)
func floatVal*(f: float64): JinjaVal = jinja_data_model.floatVal(f)
func strVal*(s: string): JinjaVal = jinja_data_model.strVal(s)
func seqVal*(xs: seq[JinjaVal]): JinjaVal = jinja_data_model.seqVal(xs)
func rangeVal*(start, stop, step: int64): JinjaVal =
  ## Returns the lazy range value over `start`, `stop` and `step`.
  jinja_data_model.rangeVal(start, stop, step)

func cutVal*(s: sink string, lo, hi: int32): JinjaVal =
  ## Returns the stripped cut of `s`, rendering as the bytes of `s[lo ..< hi]`:
  ## - `s` moves in, so the cut shares the source's buffer
  ## - the cut materializes its string only where a consumer stores or re-computes it
  ##
  ##   cutVal(text, 4'i32, 9'i32)
  jinja_data_model.cutVal(s, lo, hi)

func dictVal*(pairs: openArray[tuple[k: string, v: JinjaVal]]): JinjaVal =
  ## Returns the dict value built from the key/value pairs, the whole mapping
  ## up front, never changed afterwards:
  ## - the pairs keep their order, so a read scans in construction order
  ## - a repeated key keeps the last pair's value, matching the Python dict literal
  ##
  ##   dictVal(@[("role", strVal("system")), ("content", strVal("hi"))])
  ##
  ## Each pair's key and value may name any expressions, so a caller builds a full
  ## context inline without ever naming `DictVal`.
  jinja_data_model.dictVal(pairs)

func toJson*(v: JinjaVal, opts = JsonOpts()): string =
  ## Returns the `tojson` filter rendering, non-ASCII as raw UTF-8 unless the template
  ## passes `ensure_ascii`, Jinja's HTML escaping applied as the filter's post-pass.
  ## Caller-side drain-and-grow over the serializer, no presize pass.
  jinja_serialize.toJson(v, opts)
