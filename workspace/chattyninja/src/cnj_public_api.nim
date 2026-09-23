# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Chattyninja public API. This module IS the package surface, holding every
## name a consumer may write and nothing else.
##
## - the render tier, `parseJinjaTemplate`, `startJinjaRender`, `pullInto`,
##   `renderToString`, over the compiled artifact and the caller's `ChatContext`
## - the data tier, the typed chat records (`ChatContext`, `Message`, `Tool`)
##   with their constructors and `JinjaError`
## - the types the public signatures name and the depth, nesting, step, element caps
##
## - the caller's data unit is a `ChatContext`, the conversion into the engine's
##   value model running inside `chatContextValue`, once per render entry
## - the engine's value model stays engine-internal, harness code reaching it through `import x {.all.}`

import ./cnj_types {.all.}
import ./cnj_parse {.all.}
import ./cnj_engine {.all.}
import ./jinja_data_model {.all.}
import ./chat_completions

# Types, error and chat-context exports the public signatures name.
export cnj_types.CompiledTemplate, cnj_types.CompiledSymbols,
    cnj_types.JinjaRenderContext,
    jinja_data_model.JinjaCause, jinja_data_model.JinjaError,
    chat_completions

# Render tier, parse once and render over the shared artifact.

proc parseJinjaTemplate*(src: string): (CompiledTemplate, CompiledSymbols) =
  ## Returns the compiled template artifact plus its `CompiledSymbols`:
  ## - interned names build in parse order, read-only at render
  ## - the interned-name table is a heap object the parse allocates once, returned by ref,
  ##   every render over the artifact sharing it
  ## - the template borrows `src`, so it must not outlive the caller's text
  cnj_parse.parseJinjaTemplate(src)

proc startJinjaRender*(tmpl: CompiledTemplate, sym: CompiledSymbols, ctx: ChatContext, clock = 0.0): JinjaRenderContext =
  ## Returns a render context over the shared artifact, ready to render `ctx`.
  ##
  ## - `clock` is the epoch `strftime_now` reads, never artifact state, so one artifact
  ##   renders reproducibly under different clocks
  ## - `sym` is the parse-built interned-name table by ref, every render over the artifact
  ##   holding the same heap object, no borrow contract
  cnj_engine.startJinjaRender(tmpl, sym, chatContextValue(ctx), clock)

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

proc renderToString*(tmpl: CompiledTemplate, sym: CompiledSymbols, ctx: ChatContext, clock = 0.0): string =
  ## Returns the whole render of the artifact over `ctx`, callers rendering repeatedly
  ## over one artifact through this overload.
  ##
  ## - the drain pulls through a stack window, drained until it reports 0
  let c = startJinjaRender(tmpl, sym, ctx, clock)
  var buf: array[4096, char]
  while true:
    let n = pullInto(c, buf)
    if n == 0:
      break
    addView(result, buf.toOpenArray(0, n - 1))

proc renderToString*(src: string, ctx: ChatContext, clock = 0.0): string =
  ## Returns the whole render of `src` over `ctx`, compiling and rendering in one call.
  ##
  ## - compiling happens at the scope that owns `src`, the artifact borrowing the template
  ##   text and never outliving it
  let (tmpl, sym) = cnj_parse.parseJinjaTemplate(src)
  result = renderToString(tmpl, sym, ctx, clock)
