# Corpus manifest: 18 templates ordered by feature burden

Survey run 2026-09-17 over the extracted `corpus/<model>/` tree. Counts come from string-stripped greps
of the template source, with the expression tier measured inside `{{ }}` and tag expressions only. The
rows column counts fixture frames, split ok / err (`err_*` rows carry `expected_error`, no `rendered`).

| # | model | bytes | rows | if/elif | for (multi) | set | ns assign | macro (chain, cycle) | break | gen | setBlock | dict lit | paren | loop attrs |
|--:|---|--:|--|--|--|--|--|--|--|--|--|--|--|---|
| 1 | deepseekv2lite | 459 | 4/0 | 3/2 | 1 (0) | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | none |
| 2 | moonlight | 527 | 4/0 | 5/0 | 1 (0) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | first |
| 3 | gemma3 | 1532 | 3/2 | 7/2 | 2 (0) | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | index0, first |
| 4 | kimi | 1850 | 4/0 | 7/0 | 3 (0) | 4 | 0 | 1 (0, no) | 0 | 0 | 0 | 0 | 2 | none |
| 5 | glm47flash | 3120 | 5/0 | 14/5 | 7 (1) | 9 | 1 | 1 (0, no) | 0 | 0 | 0 | 0 | 2 | index0, first |
| 6 | lagunaxs21 | 3788 | 5/0 | 12/0 | 4 (1) | 12 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | none |
| 7 | qwen3 | 4168 | 5/0 | 18/0 | 4 (0) | 11 | 2 | 0 | 0 | 0 | 0 | 0 | 2 | index0, first, last |
| 8 | lfm25 | 4762 | 5/0 | 20/0 | 6 (1) | 28 | 11 | 3 (1, no) | 0 | 1 | 0 | 0 | 2 | index0, last |
| 9 | ling30 | 6041 | 5/0 | 22/0 | 5 (1) | 17 | 2 | 0 | 0 | 0 | 0 | 0 | 2 | index0, first, last |
| 10 | mistral7bv01 | 7572 | 5/1 | 26/0 | 9 (0) | 30 | 18 | 0 | 0 | 0 | 0 | 5 | 1 | index, index0, length |
| 11 | qwen35 | 7755 | 4/0 | 30/0 | 6 (1) | 19 | 4 | 1 (0, no) | 0 | 0 | 0 | 0 | 2 | index0, first, last, previtem, nextitem |
| 12 | qwen36 | 7764 | 5/0 | 30/0 | 6 (1) | 19 | 4 | 1 (0, no) | 0 | 0 | 0 | 0 | 2 | index0, first, last, previtem, nextitem |
| 13 | mimo25 | 8259 | 6/0 | 29/0 | 8 (2) | 21 | 1 | 2 (0, no) | 0 | 0 | 0 | 0 | 1 | index0, last, previtem, nextitem |
| 14 | qwen38flashnext | 8952 | 5/9 | 34/0 | 6 (1) | 21 | 4 | 1 (0, no) | 0 | 0 | 0 | 0 | 2 | index0, first, last, previtem, nextitem |
| 15 | glm53flash | 10950 | 12/0 | 45/18 | 22 (2) | 43 | 16 | 12 (2, no) | 7 | 0 | 0 | 0 | 2 | index0, first |
| 16 | northminicode10 | 12397 | 5/0 | 44/0 | 9 (0) | 43 | 8 | 6 (1, no) | 1 | 0 | 1 | 2 | 2 | index0, first, last |
| 17 | gptoss20b | 16714 | 4/4 | 46/17 | 8 (2) | 20 | 5 | 4 (2, yes) | 0 | 0 | 0 | 0 | 1 | index, last |
| 18 | gemma4 | 18569 | 4/0 | 65/22 | 22 (5) | 59 | 21 | 5 (4, yes) | 0 | 0 | 1 | 0 | 3 | index0, last |

Column notes:

- macro (chain, cycle):
  count of `{% macro %}` defs (37 total), the longest static call chain,
  and cycle presence, measured by string-stripped call-graph DFS per template
- ns assign:
  `ns.field = ...` change sites, the `nkSetNamespace` fixture demand. The `namespace(...)` creation appears
  alongside in the same templates
- dict lit:
  `{`-opening literals inside expressions, 7 sites across 2 templates. `mistral7bv01` holds 5 within
  `for` iterables and `set ns.x = ... + [{...}]` appends. `northminicode10` holds 2, in `set text_wrapper = {...}`

- paren:
  max paren nesting depth reached by any single expression. Corpus max is 3 (`gemma4`)
- gen:
  `{% generation %}` tag at `lagunaxs21.jinja:44-76` and `lfm25.jinja:77-107`. Recorded spans in row
  data are codepoint ranges into `rendered`, and 8 rows carry non-empty spans
- setBlock:
  the `{% set x %}...{% endset %}` capture, at `northminicode10.jinja:2-9` and `gemma4.jinja:322-346`.
  No other template exercises it

- `break`:
  `glm53flash` x7 (inside macro `has_dup_tool_result_id`) and `northminicode10` x1, 8 total
- `not in`:
  `gemma4`, `mimo25`, `mistral7bv01` x2, `northminicode10`, `gptoss20b` x3, `qwen38flashnext`, 9 total
- `strftime_now`:
  one site, `gptoss20b.jinja:202`, inside macro `build_system_message`

- Cycles, measured on this corpus:
  `gptoss20b` (`render_typescript_type` self-recurses on `items` /`variant`) and `gemma4`
  (`format_parameters` with `format_argument` recurse mutually)
- Not cyclic in these files:
  the `kimi` `render_content`, `mimo25` `render_extra_keys` and `northminicode10` macros, whose calls
  all sit outside macro bodies
- Depth tracks input data:
  the shipped `gemma4/tools_tool_response.json` drives tools-subtree JSON nesting to 6

- caller():
  appears in zero templates

## Build targets: the three cheapest stages, justified

**Stage 1 (verbatim + emit + if). deepseekv2lite** (459 B, 4 ok rows). Exercises verbatim, emit,
`if` /`elif`, the `defined` test, string concatenation and the `is defined` set-guard.
Every corpus template loops over `messages`, so a byte-exact check here needs a minimal `nkFor` /`nkSet` too.

No cheaper honest option exists. The second-cheapest template, `moonlight` (527 B), has the same shape plus `loop.first`.

**Stage 2 (for + set + setNs + loop). ling30** (6041 B, 5 ok rows). Cheapest template carrying `nkSetNamespace`
(2 change sites, 2 `namespace()` creations) plus `for` (5 sites incl. 1 multi-target), `set` (17),
`loop.index0/first/last`, `length`/`tojson` filters, `defined`/`string` tests, 7 string methods, zero macros and generation tags.

Cheaper ns-bearing templates, `glm47flash` (3120 B) and `lfm25` (4762 B), both carry macros, which
belongs to stage 3. `mistral7bv01` (7572 B) is macro-free too but strictly heavier, with 18 ns sites,
5 dict literals and `not in`.

**Stage 3 (macro + capture + generation). A forced pair**, because no single template covers all three.
setBlock capture exists only in `northminicode10` and `gemma4`, generation only in `lagunaxs21`,
`lfm25`, and their macro-bearing intersection is empty.

The cheapest covering pair is **northminicode10** with macros x6, the capture, `break`, 2 dict literals
and `set` /`setNs`, plus **lfm25** with macros x3, a generation span whose 5 rows all carry recorded spans,
plus `setNs`, `default` /`join`/`trim` filters and `mapping` /`string` tests. Together 17 KB, not 27 KB.

The `gemma4` +`lagunaxs21` alternative costs 27 KB. Recurring-macro coverage cannot use `kimi`,
whose single macro does not self-call in these files.

It runs **gptoss20b** `render_typescript_type` (self-recursive on schema `items`/`variant`) or **gemma4**
`format_argument`/`format_parameters` (mutual recursion, nesting driven to 6 by the shipped fixture).
