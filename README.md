# Paper to Playground

An agent that turns a research-paper concept plus a learning brief into a single, offline, interactive
explainer page (`out/index.html`). Every run also writes a full execution trace (`out/trace.jsonl`).

**Team:** Ahmad Karnib, Mohamad Nasrallah, Hassan Nasrallah

## Run

```bash
python -m pip install -r requirements.txt
export OPENROUTER_API_KEY=...        # read from the environment; never logged, never embedded in the page
python agent.py --input case.json --output out --model deepseek/deepseek-v4.1-flash
```

**MODEL_ID:** `deepseek/deepseek-v4.1-flash`. All calls go to `https://openrouter.ai/api/v1/chat/completions`
with Bearer auth and use exactly the given MODEL_ID.

`case.json` holds three strings: `source_url`, `focus` and `audience`. Any extra string fields are passed to
the model as well. Requires Python 3.11 and nothing beyond `pip install -r requirements.txt`. Exit codes: 0 when
the page was generated and its mandatory checks pass; 3 when a usable page was written but a mandatory check
(a crash, a brief-required test, or an invariant that is false at the defaults) is still failing, and the
page shows it as failing; 1 when generation failed (a fallback page and the trace are still written); 2 for
a missing key or unreadable input (trace only).

### Optional: browser input page

```bash
python webui.py            # opens http://127.0.0.1:8800/ in your browser
```

A local page with three fields: paper link, what to explain, and who it is for. Example buttons fill in
the public examples. While the page is generated it shows live progress through the agent's stages, then
the finished page with its checks, calls, tokens and time. It also offers open, download, the trace, and a
list of recent pages. It runs `agent.py` unchanged in a subprocess and stores results in `ui_runs/`. The API
key comes from the environment (or can be typed in) and is never written to disk. It uses only the Python
standard library and is not needed for the graded command above.

## Architecture: the LLM writes the science, Python writes the page

```
case.json
  └─ understand   split `focus` into requirements and its explicit "check that …" statements
  └─ generate     1 LLM call → compact JSON spec + one pure JS `compute(s)` function
  └─ check        deterministic, no LLM: run compute() in embedded V8 about 150×
  └─ revise       deterministic fixes first; then ≤1 targeted LLM patch per draft; a fresh draft only if
                  correctness checks still fail; keep the better candidate; at most one final patch
  └─ review       1 LLM call: prose checked against a table of the page's own computed values
                  (prose-only patch, re-validated, rejected if anything gets worse)
  └─ render       fixed, tested template + LaTeX→MathML + computed numbers filled into the prose
```

1. **Understand.** The brief's explicit checks (e.g. "check that four equally likely outcomes give two bits")
   are extracted. Each one must become an executable test.
2. **Generate (one call).** The model returns a compact spec and one pure JavaScript `compute(s)` function.
   It never writes HTML or CSS, which keeps the token count and latency low. The spec contains:
   - a short plan with a requirement→view coverage map;
   - teaching text with LaTeX;
   - a symbol table and how-it-works steps;
   - typed controls, readouts and views;
   - invariants and brief-derived tests;
   - two guided explorations made of one-click steps, each with an assertion.

   **Numbers in the prose are never typed by the model.** It writes `{{out.H}}` (or `{{2:out.H}}` for
   exploration step 2), and the generator fills in the value that `compute()` actually produces at that
   setting.
3. **Check (deterministic).** `compute` is executed in embedded V8 (`mini-racer`, installed by pip; Node.js
   is a fallback). It runs at the defaults, every exploration step, every test state, the edge values of
   every control (min, max, zeros, one-hot, resize) and 24 seeded random states. The checks are:
   - no exceptions and no NaN/Infinity;
   - every brief-derived test passes;
   - every exploration claim is true at its preset, and from step 2 on a claim can compare with the
     previous step via `prev` (e.g. `prev.H < out.H`);
   - invariants hold on every reachable input;
   - **every control actually changes the output**;
   - readout and view data shapes are valid;
   - every `{{…}}` number evaluates;
   - the LaTeX parses;
   - the page loads no external resources.
4. **Revise.**
   - Low-value failures are fixed without an LLM: conditional "invariants" are dropped, as are display-only
     controls and malformed optional readouts or views.
   - Real failures (code contradicting a test or a claimed observation) go back to the model with the
     failing checks, the actual computed values and only the relevant keys. The model returns a patch, and a
     patch is accepted only if a severity-weighted error score does not rise.
   - If a correctness check still fails, an independent fresh draft is generated and the better candidate
     is kept, because repairs tend to stay anchored to a wrong formula.
   - A check that still cannot pass is not hidden: it stays on the page, shown as failing, and is listed in
     the trace (`keep_failing_checks_visible`). An invariant that is false at the default settings counts as
     a real error and goes to repair; only invariants that hold at the defaults but fail elsewhere (that is,
     mis-specified) are dropped, and each drop is logged.
   - Each exploration step starts from the default settings before applying its preset, exactly as it
     was validated, so earlier edits by the learner cannot change what a step shows.
5. **Review.** One short call shows the model its own prose next to the outputs `compute()` produces at the
   defaults and at every exploration step. Every `{{…}}` placeholder is annotated with the value it will
   render. The model may rewrite only sentences that contradict the numbers, wrong causal explanations,
   overreaching paper attributions, a boilerplate caveat or undefined jargon. Presets, checks and code
   cannot be changed in this pass, and the patch is re-validated.
6. **Render.** Python escapes all prose and converts LaTeX to MathML (`latex2mathml`). In multi-step
   explorations, every filled number is labelled with its step, e.g. "H rises from 0 (step 1) to 2 (step 2)".
   Section and equation numbers that the brief does not state are dropped from the citation line, as a guard
   against invented citations. A section number in the brief does not validate an equation number, and vice
   versa. The tested template
   (`templates/page.html`, vanilla JS/SVG) provides:
   - views: bars, line, heatmap, 2-D plane, graph, computation pipeline, table;
   - accessible controls: sliders, toggles, selects, and editable vectors and matrices with add/remove;
   - live intermediate values and live invariant ✓/✗;
   - in-page, re-runnable built-in checks;
   - "Set up this exploration" step buttons with live checks;
   - a source-grounding panel separating paper claims from our simplifications.

   The output is one self-contained file: no CDN, fonts or remote assets, and no API key.

**Budget and limits.** At most 6 requests (the cap is 10). Completion tokens are capped against 30k and
there is a 540 s internal deadline (the cap is 600 s). Reasoning is disabled, and requests use OpenRouter's
`provider.sort = "throughput"` for the given MODEL_ID. On the public examples, a typical run uses 2–3 calls,
about 10–14k total tokens and 15–25 s. Hard cases that need a fresh draft can use up to about 57k total
tokens and up to about 2 minutes. Completion tokens are hard-capped at 30k minus a 300-token margin; over 130
test runs the worst run used 29.7k, and only 3 runs used more than 20k.

**No network access to the paper is assumed** (assessment allows only OpenRouter). The model works from
the brief and its own knowledge of the paper. It is told to copy section and equation locations from the
brief or name them only when certain, to keep the paper's hedges, and to label everything else as a
simplification.

**Trace** (`out/trace.jsonl`). Each line is one JSON event with `t`, `stage`, `action` and `result`. LLM call
events include `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `elapsed_s`, the OpenRouter
`generation_id` and `provider`. The trace also records the plan, every check result and issue,
deterministic fixes, repair rounds (accepted or rejected), regeneration decisions, filled numbers and a
final summary with totals. Credentials and hidden reasoning are never logged.

## Testing

- `examples/`: the two public examples, plus one generated input/output pair in `examples/attention_out/`
  (`index.html` and `trace.jsonl`; assessed outputs are generated afresh).
- `tests/cases/`: 16 additional briefs written to differ from the public examples: Bitcoin attacker
  probability, Adam bias correction, PageRank, nucleus sampling, GAN optimal discriminator, ResNet
  shortcuts, LoRA, BSC capacity, positional encoding, Diffie–Hellman, the Kalman filter, convolution
  stride/padding, backpropagation, Black–Scholes, BLEU, and dropout with no audience field. When the
  audience field is missing, an engineering-undergraduate audience is assumed and stated.
- `python tools/run_cases.py TAG tests/cases/*.json examples/*.json` runs them in parallel and summarizes
  calls, tokens, time and unresolved issues.

During development, independent LLM judges scored generated pages against the rubric. They re-derived each
paper's formula in Node to verify the numbers. Their findings drove the generic fixes above: computed
numbers in prose, `prev`-step comparisons, coverage mapping, fixed probability colour scales and fresh-draft
regeneration. The prompts contain only generic format examples; nothing in the code or prompts encodes the
answer for a specific paper.

## Files

- `agent.py`: pipeline (OpenRouter client, parsing and JSON repair, normalization, V8 validation, repair
  loop, rendering, trace)
- `webui.py`: optional local browser input page (standard library only; runs `agent.py` unchanged)
- `prompts.py`: generic system prompts
- `templates/page.html`: generic page template and renderer
- `CONTRACT.md`: a summary of the data contract between the generator and the template

## Credits / reuse

- [latex2mathml](https://github.com/roniemartinez/latex2mathml) (MIT): LaTeX → MathML.
- [mini-racer](https://github.com/bpcreech/PyMiniRacer) (ISC): embedded V8 used to execute generated `compute()` during validation.
- [json-repair](https://github.com/mangiucugna/json_repair) (MIT): last-resort repair of malformed model JSON.
- [requests](https://requests.readthedocs.io/) (Apache-2.0) and its pinned dependencies urllib3 (MIT), certifi (MPL-2.0),
  charset-normalizer (MIT) and idna (BSD-3-Clause).
