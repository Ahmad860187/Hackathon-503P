# Page contract (generator ⇄ template)

`agent.py` asks the LLM for a compact **spec** (JSON) plus a **compute** JS function, validates them,
converts every prose field to safe HTML (escaping + LaTeX→MathML), and injects two things into
`templates/page.html`:

```html
<script id="page-data" type="application/json">/*__PAGE_JSON__*/</script>
<script>/*__COMPUTE_JS__*/</script>      <!-- defines: function compute(s) { ... return out; } -->
```

`</` is escaped as `<\/` in both injections. The template's own JS reads `PAGE = JSON.parse(...)`
and renders the whole page. It must render all static sections even if `compute` is missing or throws.

## PAGE object

```js
{
  meta: { title, subtitle, audience, source_url, model, generated_at },
  paper: { title, authors, year, section, equation },           // plain strings, may be ""
  idea_html, why_html,                                           // HTML strings
  equation: { mathml_html, caption_html },                      // display equation (MathML <math display="block">)
  symbols: [ { symbol_html, meaning_html, where_html } ],        // symbol_html is MathML; where_html may be ""
  steps:   [ { title_html, text_html } ],                        // "how it works" numbered steps (may be [])
  controls: [ Control ],
  readouts: [ { key, label, unit, digits } ],                    // big live numbers from out[key]
  views:    [ { id, type, title, caption_html, data } ],         // data = key into out
  invariants: [ { label, expr } ],                               // JS boolean expr over (out, s); shown live ✓/✗
  tests:      [ { label, state, expect } ],                      // state = partial control values; expect = JS bool expr over (out, s)
  explorations: [ { title, change_html, observe_html, why_html, preset, expect } ],
  caveat: { kind, title, text_html },                            // kind: "limitation" | "assumption" | "misconception"
  grounding: { from_paper_html: [..], simplifications_html: [..], not_claimed_html },
  report: { checks: [ {name, ok, detail} ], revisions, calls, prompt_tokens, completion_tokens, elapsed_s }
}
```

## Controls  (`s[id]` is the value passed to compute)

| type     | fields                                                                                       | value          |
|----------|----------------------------------------------------------------------------------------------|----------------|
| slider   | id, label, min, max, step, default, help?                                                     | number         |
| number   | id, label, min?, max?, step?, default, help?                                                  | number         |
| toggle   | id, label, default, help?                                                                     | boolean        |
| select   | id, label, options:[{value,label}], default, help?                                            | string         |
| vector   | id, label, default:number[], min, max, step, min_len?, max_len?, item_labels?:string[], help? | number[]       |
| matrix   | id, label, default:number[][], min, max, step, row_labels?, col_labels?, resizable?:bool, min_rows?, max_rows?, min_cols?, max_cols?, help? | number[][] |

`label` is plain text (unicode allowed, no LaTeX). `help` is plain text.
Vector with `min_len < max_len` gets add/remove buttons; matrix with `resizable` gets +/- row/col buttons.

## compute(s) → out

Pure JS, no DOM. `out` is a plain object. View `data` keys point into `out`.
If `out.warning` is a non-empty string the template shows it near the visuals.

### View data shapes (`out[view.data]`)

- **bars**: `{labels:string[], values:number[]}` or `{labels, series:[{name, values:number[]}]}`;
  optional `xlabel, ylabel, ymin, ymax, highlight:number[] (indices), digits, reference:{value,label}`.
- **line**: `{x:number[], series:[{name, y:number[]}]}`; optional `xlabel, ylabel, xmin, xmax, ymin, ymax,
  markers:[{x,y,label}], vline:{x,label}, hline:{y,label}`.
- **heatmap**: `{matrix:number[][]}`; optional `row_labels, col_labels, digits, min, max, xlabel, ylabel`.
- **plane** (2-D geometry): optional `vectors:[{x,y,label,from?:[x0,y0]}], points:[{x,y,label}],
  paths:[{points:[[x,y],...], label}], xlabel, ylabel, range:[xmin,xmax,ymin,ymax]`.
- **graph**: `{nodes:[{id,label?,value?,x?,y?}], edges:[{from,to,weight?,label?}]}`; optional `directed, digits`.
  x,y (if given) are in [0,1]; otherwise nodes are laid out on a circle.
- **pipeline**: `{stages:[{label, value?, note?}]}` — value may be number | string | number[] | number[][].
- **table**: `{columns:string[], rows:(string|number)[][]}`; optional `highlight_rows:number[], digits`.
