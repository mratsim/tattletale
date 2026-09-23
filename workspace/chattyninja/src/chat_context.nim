# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Chat caller's typed context, the records a consumer builds plus the engine-internal
## conversion into the renderer's value model.
##
## - `renderToString` and `startJinjaRender` take one `ChatContext` argument
## - the conversion runs once inside `chatContextValue`, reachable only through an explicit `import x {.all.}`
## - a reasoning payload travels under all three spellings the readers consult
##   (`reasoning_content`, `reasoning`, `thinking`), each reader an or/elif/get
##   chain consulting exactly one spelling

import std/options
import ./jinja_data_model {.all.}

type
  KwargsKind* = enum
    kwText, kwFlag, kwInt
      ## scalar form of one template kwarg

  KwargsValue* = object
    ## One template kwarg's value
    ##
    ## - the corpus passes the `string` and `bool` forms
    ## - the int form serves numeric knobs a template may compare
    case kind*: KwargsKind
    of kwText: text*: string
    of kwFlag: flag*: bool
    of kwInt: num*: int64

  ParamKind* = enum
    pvText, pvFlag, pvInt, pvReal, pvList, pvRec
      ## scalar or container form of one value inside a tool's parameter schema

  ParamValue* = object
    ## One value inside a tool's parameter schema, the JSON-schema subset the corpus passes, key order kept
    ##
    ## - scalar leaves
    ## - a list of leaves, the schema's `required` array
    ## - nested records, the schema's `properties` objects
    case kind*: ParamKind
    of pvText: text*: string
    of pvFlag: flag*: bool
    of pvInt: num*: int64
    of pvReal: real*: float64
    of pvList: items*: seq[ParamValue]
    of pvRec: fields*: seq[tuple[k: string, v: ParamValue]]

  BlockKind* = enum
    bkvText, bkvBlocks
      ## scalar or nested-block form of one content block field

  BlockValue* = object
    ## One content block field's value, a string or a nested block list
    ## (a tool response's `output` carrying its own text blocks)
    case kind*: BlockKind
    of bkvText: text*: string
    of bkvBlocks: blocks*: seq[ContentBlock]

  ContentBlock* = object
    ## One content block, the ordered key/value pairs the recording passed
    ##
    ## - the `type` pair names the block kind
    ## - the remaining pairs are the template-read fields (`text`, `image_url`, `video_url`, `audio_url`, `name`, `tool_call_id`, `output`)
    ##
    ## A field absent from `fields` stays absent from the rendered message,
    ## what the templates' `is defined` and membership checks observe.
    fields*: seq[tuple[k: string, v: BlockValue]]

  ContentKind* = enum
    ckNone, ckText, ckBlocks
      ## absent, string or block-list form of one message's content

  Content* = object
    ## One message's content, the absent form, the string form or the block-list form
    ## the multimodal and chunked rows pass
    case kind*: ContentKind
    of ckNone: nil
    of ckText: text*: string
    of ckBlocks: blocks*: seq[ContentBlock]

  ToolCall* = object
    ## One assistant tool call, the function name and its arguments in one ordered record
    ##
    ## - `id` and `content_type` carry the call id and the gpt-oss content type
    ## - the corpus never passes the string arguments form
    ## - the templates branch on it defensively, the ordered record being the only form the corpus observes
    name*: string
    arguments*: seq[tuple[k: string, v: ParamValue]]
    id*: Option[string]
    content_type*: Option[string]

  Tool* = object
    ## One function tool, the name, description and JSON-schema parameter record
    ##
    ## - `tool_type` is the wrapper key the OpenAI-style recordings pass
    ## - the gemma4 recording omits it and the builder emits the tool bare
    name*: string
    description*: string
    parameters*: seq[tuple[k: string, v: ParamValue]]
    tool_type*: Option[string]

  Message* = object
    ## One chat message, the keys the recorded templates read
    ##
    ## - an `Option` field left as `none` renders as the absent key
    ## - absent is what the templates' `is string` and membership checks observe
    role*: string
    content*: Content
    tool_calls*: seq[ToolCall]
    reasoning_content*: Option[string]
      ## the reasoning payload, delivered under `reasoning_content`, `reasoning`
      ## and `thinking`, see the module contract
    name*: Option[string]
      ## persona name the tool-declare templates read off the message
    tool_call_id*: Option[string]
      ## id a tool-role message answers

  ChatContext* = object
    ## Typed chat context a caller hands the render entry points
    ##
    ##   let msg = Message(role: "user", content: textContent("hi"))
    ##   let tool = Tool(name: "get_weather", description: "Weather for one city", parameters: @[("type", pvVal("object"))])
    ##   let ctx = ChatContext(messages: @[msg], tools: @[tool], add_generation_prompt: true, kwargs: @[("bos_token", kwVal("<bos>"))])
    messages*: seq[Message]
    tools*: seq[Tool]
      ## empty renders as the none value, the recorded render path's absent shape
    documents*: seq[ContentBlock]
      ## grounding documents the north-bay template reads, empty renders as the none value like `tools`
    add_generation_prompt*: bool
    kwargs*: seq[tuple[name: string, val: KwargsValue]]
      ## template kwargs in order, appended after the standard keys

# Scalar constructors, the typed records' value language, one helper per form.

func kwVal*(s: string): KwargsValue =
  ## Returns the string form of a template kwarg.
  KwargsValue(kind: kwText, text: s)

func kwVal*(b: bool): KwargsValue =
  ## Returns the bool form of a template kwarg.
  KwargsValue(kind: kwFlag, flag: b)

func kwVal*(i: int64): KwargsValue =
  ## Returns the int form of a template kwarg.
  KwargsValue(kind: kwInt, num: i)

func pvVal*(s: string): ParamValue =
  ## Returns the string leaf of a parameter schema.
  ParamValue(kind: pvText, text: s)

func pvVal*(b: bool): ParamValue =
  ## Returns the bool leaf of a parameter schema.
  ParamValue(kind: pvFlag, flag: b)

func pvVal*(i: int64): ParamValue =
  ## Returns the int leaf of a parameter schema.
  ParamValue(kind: pvInt, num: i)

func pvVal*(f: float64): ParamValue =
  ## Returns the float leaf of a parameter schema.
  ParamValue(kind: pvReal, real: f)

func pvSeq*(items: varargs[ParamValue]): ParamValue =
  ## Returns the list form of a parameter-schema value, the schema's `required` array.
  ##
  ##   let required = pvSeq(pvVal("city"), pvVal("unit"))
  ParamValue(kind: pvList, items: @items)

func pvRec*(fields: openArray[tuple[k: string, v: ParamValue]]): ParamValue =
  ## Returns the record form of a parameter-schema value, the schema's
  ## nested objects (`properties`), key order kept.
  ParamValue(kind: pvRec, fields: @fields)

func bvVal*(s: string): BlockValue =
  ## Returns the string form of a content-block field.
  BlockValue(kind: bkvText, text: s)

func bvList*(blocks: sink seq[ContentBlock]): BlockValue =
  ## Returns the nested-block form of a content-block field, a tool response's
  ## `output` list of text blocks.
  BlockValue(kind: bkvBlocks, blocks: blocks)

func blockOf*(fields: openArray[tuple[k: string, v: BlockValue]]): ContentBlock =
  ## Returns one content block from its ordered key/value pairs, the `type` pair
  ## naming the block kind
  ##
  ##   let b = blockOf([("type", bvVal("image")), ("url", bvVal("moon.png"))])
  ContentBlock(fields: @fields)

func textContent*(s: string): Content =
  ## Returns the string form of a message's content.
  Content(kind: ckText, text: s)

func blockContent*(blocks: sink seq[ContentBlock]): Content =
  ## Returns the block-list form of a message's content.
  Content(kind: ckBlocks, blocks: blocks)

func noneContent*(): Content =
  ## Returns the absent form of a message's content, the templates'
  ## `is none` test observing it.
  Content(kind: ckNone)

# Conversion tier, once per render entry.

func paramValue(p: ParamValue): JinjaVal =
  ## One schema value into the engine's value model, containers recursed.
  case p.kind
  of pvText: strVal(p.text)
  of pvFlag: boolVal(p.flag)
  of pvInt: intVal(p.num)
  of pvReal: floatVal(p.real)
  of pvList:
    var xs: seq[JinjaVal]
    for item in p.items:
      xs.add paramValue(item)
    seqVal(xs)
  of pvRec:
    var pairs: seq[tuple[k: string, v: JinjaVal]]
    for (k, v) in p.fields:
      pairs.add (k, paramValue(v))
    dictVal(pairs)

func blockValue(p: BlockValue): JinjaVal
  ## forward declaration, the two block forms recursing into each other

func blockValueOf(c: ContentBlock): JinjaVal =
  ## One content block into an insertion-ordered dict, absent fields staying absent.
  var d = DictVal()
  for (k, v) in c.fields:
    dictSet(d, k, blockValue(v))
  dictVal(d)

func blockValue(p: BlockValue): JinjaVal =
  case p.kind
  of bkvText: strVal(p.text)
  of bkvBlocks:
    var xs: seq[JinjaVal]
    for item in p.blocks:
      xs.add blockValueOf(item)
    seqVal(xs)

func contentValue(c: Content): JinjaVal =
  ## One message's content into the value model, blocks recursed.
  case c.kind
  of ckNone: noneVal()
  of ckText: strVal(c.text)
  of ckBlocks:
    var xs: seq[JinjaVal]
    for b in c.blocks:
      xs.add blockValueOf(b)
    seqVal(xs)

func toolCallValue(tc: ToolCall): JinjaVal =
  ## One tool call into its message dict
  ##
  ## - flat keys (`name`, `arguments`, `content_type`) for flat-access templates
  ## - a `function` record for wrapper-reading templates
  ## - both spellings delivered, the corpus readers branch on either
  var d = DictVal()
  if tc.id.isSome:
    dictSet(d, "id", strVal(tc.id.get))
  dictSet(d, "name", strVal(tc.name))
  var argPairs: seq[tuple[k: string, v: JinjaVal]]
  for (k, v) in tc.arguments:
    argPairs.add (k, paramValue(v))
  let args = dictVal(argPairs)
  dictSet(d, "arguments", args)
  var fn = DictVal()
  dictSet(fn, "name", strVal(tc.name))
  dictSet(fn, "arguments", args)
  dictSet(d, "function", dictVal(fn))
  if tc.content_type.isSome:
    dictSet(d, "content_type", strVal(tc.content_type.get))
  dictVal(d)

func messageValue(m: Message): JinjaVal =
  ## One message into its dict, absent `Option` fields staying absent.
  ##
  ## A reasoning payload lands under all three spellings the readers consult
  ## (`reasoning_content`, `reasoning`, `thinking`), every reader an or/elif/get
  ## chain that consults exactly one spelling.
  var d = DictVal()
  dictSet(d, "role", strVal(m.role))
  dictSet(d, "content", contentValue(m.content))
  if m.reasoning_content.isSome:
    let r = strVal(m.reasoning_content.get)
    dictSet(d, "reasoning", r)
    dictSet(d, "reasoning_content", r)
    dictSet(d, "thinking", r)
  if m.name.isSome:
    dictSet(d, "name", strVal(m.name.get))
  if m.tool_call_id.isSome:
    dictSet(d, "tool_call_id", strVal(m.tool_call_id.get))
  if m.tool_calls.len > 0:
    var xs: seq[JinjaVal]
    for tc in m.tool_calls:
      xs.add toolCallValue(tc)
    dictSet(d, "tool_calls", seqVal(xs))
  dictVal(d)

func toolValue(t: Tool): JinjaVal =
  ## One tool into its recording dict, the wrapper key emitted when set.
  var d = DictVal()
  if t.tool_type.isSome:
    dictSet(d, "type", strVal(t.tool_type.get))
  var fn = DictVal()
  dictSet(fn, "name", strVal(t.name))
  dictSet(fn, "description", strVal(t.description))
  var paramPairs: seq[tuple[k: string, v: JinjaVal]]
  for (k, v) in t.parameters:
    paramPairs.add (k, paramValue(v))
  dictSet(fn, "parameters", dictVal(paramPairs))
  dictSet(d, "function", dictVal(fn))
  dictVal(d)

func chatContextValue*(ctx: ChatContext): JinjaVal =
  ## Returns the engine value of one typed chat context
  ##
  ## - keys in recording order, `messages`, `tools`, `documents`, `add_generation_prompt`,
  ##   then the kwargs
  ## - empty `tools` and `documents` render as the none value, the recorded
  ##   render path's absent shape
  var ms: seq[JinjaVal]
  for m in ctx.messages:
    ms.add messageValue(m)
  var d = DictVal()
  dictSet(d, "messages", seqVal(ms))
  if ctx.tools.len == 0:
    dictSet(d, "tools", noneVal())
  else:
    var ts: seq[JinjaVal]
    for t in ctx.tools:
      ts.add toolValue(t)
    dictSet(d, "tools", seqVal(ts))
  if ctx.documents.len == 0:
    dictSet(d, "documents", noneVal())
  else:
    var ds: seq[JinjaVal]
    for doc in ctx.documents:
      ds.add blockValueOf(doc)
    dictSet(d, "documents", seqVal(ds))
  dictSet(d, "add_generation_prompt", boolVal(ctx.add_generation_prompt))
  for (name, kw) in ctx.kwargs:
    case kw.kind
    of kwText: dictSet(d, name, strVal(kw.text))
    of kwFlag: dictSet(d, name, boolVal(kw.flag))
    of kwInt: dictSet(d, name, intVal(kw.num))
  dictVal(d)
