# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Chattyninja package umbrella.
## Explicit re-export list, the whole package surface. A module's own `*` surface is Nim's
## cross-module plumbing, not API, and nothing under `src/` re-exports.
##
## - the render tier, `startRender`, `pullInto`, `pullAll` and `renderToString`, and `parseTemplate`
## - the data tier, `JinjaVal` with `ValueKind`, the constructors, `dictSet`/`dictGet`, `JinjaError`
## - the caller objects, `JinjaRenderContext`, `RenderState` and the compiled artifact behind `parseTemplate`
##
## `jinja_interpolation` and `jinja_builtins` stay internal, no re-export.

import ./src/cnj_engine
import ./src/cnj_types, ./src/jinja_data_model, ./src/jinja_serialize, ./src/cnj_parse

# Render driver and parser.
export cnj_engine.startRender, cnj_engine.pullInto, cnj_engine.pullAll,
    cnj_engine.renderToString, cnj_parse.parseTemplate

# Compiled artifact, render state and the caller's objects.
export cnj_types.JinjaRenderContext, cnj_types.RenderState, cnj_types.CompiledTemplate,
    cnj_types.CompiledSymbols, cnj_types.Node, cnj_types.NodeKind,
    cnj_types.MacroForcer, cnj_types.Row, cnj_types.RowKind, cnj_types.Piece,
    cnj_types.PieceKind, cnj_types.Binding, cnj_types.Scope,
    cnj_types.scopeHas, cnj_types.lookupName, cnj_types.internName,
    cnj_types.lo, cnj_types.hi, cnj_types.succ, cnj_types.child, cnj_types.alt,
    cnj_types.loopName, cnj_types.filterLo, cnj_types.filterHi, cnj_types.target,
    cnj_types.field, cnj_types.macroName, cnj_types.targetCount,
    cnj_types.targetAt, cnj_types.paramCount, cnj_types.paramNameAt,
    cnj_types.paramDefLoAt, cnj_types.paramDefHiAt,
    cnj_types.SlotHi, cnj_types.SlotSucc, cnj_types.SlotChild, cnj_types.SlotAlt,
    cnj_types.TTT_CNJ_MacroDepthCap, cnj_types.TTT_CNJ_ExprDepthCap, cnj_types.TTT_CNJ_ParseNestingCap,
    cnj_types.TTT_CNJ_StepBudget

# Values and their construction, the dict reads, the error and the JSON form.
export jinja_data_model.JinjaVal, jinja_data_model.ValueKind, jinja_data_model.SeqVal,
    jinja_data_model.DictVal, jinja_data_model.RangeVal, jinja_data_model.LoopState,
    jinja_data_model.MacroVal, jinja_data_model.PendingCallVal, jinja_data_model.Args,
    jinja_data_model.Arg, jinja_data_model.ArgKeyword, jinja_data_model.JinjaError,
    jinja_data_model.JinjaCause, jinja_data_model.JsonOpts, jinja_data_model.NoOffset, jinja_data_model.NoLink,
    jinja_data_model.ArgsCap, jinja_data_model.TTT_CNJ_RangeElemCap, jinja_data_model.TTT_CNJ_ValueDepthCap,
    jinja_data_model.undefinedVal, jinja_data_model.noneVal, jinja_data_model.boolVal,
    jinja_data_model.intVal, jinja_data_model.floatVal, jinja_data_model.strVal,
    jinja_data_model.seqVal, jinja_data_model.dictVal, jinja_data_model.nsVal,
    jinja_data_model.loopVal, jinja_data_model.macroVal, jinja_data_model.callVal,
    jinja_data_model.rangeVal, jinja_data_model.cutVal, jinja_data_model.dictSet,
    jinja_data_model.dictGet

export jinja_serialize.toJson
