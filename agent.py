#!/usr/bin/env python3
"""Paper to Playground: turn a paper concept + learning brief into an interactive offline explainer.

Pipeline: understand brief -> generate spec + compute JS (1 LLM call) -> deterministic checks
(schema + executing compute in embedded V8) -> targeted repair calls if needed -> render template.

Usage: python agent.py --input case.json --output out --model MODEL_ID
"""
import argparse
import datetime as _dt
import html
import json
import os
import random
import re
import sys
import time
from pathlib import Path

T0 = time.monotonic()

import requests  # noqa: E402

from prompts import REPAIR_SYSTEM, REVIEW_SYSTEM, SYSTEM  # noqa: E402

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "templates" / "page.html"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Assessment limits are 10 requests / 30k completion tokens / 600 s per case; stay well inside.
MAX_REQUESTS = 6
MAX_COMPLETION_TOKENS = 30000
DEADLINE_S = 540
GEN_MAX_TOKENS = 12000
REPAIR_MAX_TOKENS = 6000
MAX_REPAIRS = 2
# Route to the fastest provider serving the same MODEL_ID (latency is scored); "" disables.
PROVIDER_SORT = os.environ.get("P2P_PROVIDER_SORT", "throughput")

CONTROL_TYPES = {"slider", "number", "toggle", "select", "vector", "matrix"}
VIEW_TYPES = {"bars", "line", "heatmap", "plane", "graph", "pipeline", "table"}
CONTROL_ALIASES = {"range": "slider", "checkbox": "toggle", "bool": "toggle", "boolean": "toggle",
                   "switch": "toggle", "dropdown": "select", "radio": "select", "list": "vector",
                   "array": "vector", "int": "number", "integer": "number", "grid": "matrix"}
VIEW_ALIASES = {"bar": "bars", "barchart": "bars", "histogram": "bars", "lines": "line", "plot": "line",
                "chart": "line", "matrix": "heatmap", "heat": "heatmap", "flow": "pipeline",
                "steps": "pipeline", "network": "graph", "scatter": "plane", "vectors": "plane",
                "2d": "plane", "geometry": "plane"}


def elapsed():
    return time.monotonic() - T0


# ----------------------------------------------------------------------------- trace

class Trace:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def log(self, stage, action, result, **kw):
        ev = {"t": round(elapsed(), 3), "stage": stage, "action": action, "result": result}
        ev.update(kw)
        self.f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ----------------------------------------------------------------------------- LLM client

class BudgetExceeded(Exception):
    pass


class LLM:
    def __init__(self, model, key, trace):
        self.model, self.key, self.trace = model, key, trace
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.session = requests.Session()

    def chat(self, stage, system, user, max_tokens, attempts=2, reasoning=None):
        last_err = None
        for attempt in range(attempts):
            if self.requests >= MAX_REQUESTS:
                raise BudgetExceeded("request budget exhausted")
            remaining_time = DEADLINE_S - elapsed()
            cap = min(max_tokens, MAX_COMPLETION_TOKENS - self.completion_tokens - 300)
            if remaining_time < 25 or cap < 1000:
                raise BudgetExceeded(f"time/token budget exhausted (t={elapsed():.0f}s, tokens left={cap})")
            body = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": cap,
                "temperature": 0.2,
                "reasoning": {"effort": reasoning} if reasoning else {"enabled": False},
            }
            if PROVIDER_SORT:
                body["provider"] = {"sort": PROVIDER_SORT}
            self.requests += 1
            t = time.monotonic()
            try:
                r = self.session.post(
                    API_URL, json=body, timeout=(15, max(20, remaining_time - 15)),
                    headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                             "X-Title": "paper-to-playground"})
                status = r.status_code
                data = r.json() if r.content else {}
            except (requests.RequestException, ValueError) as e:
                last_err = f"{type(e).__name__}: {e}"[:300]
                self.trace.log(stage, "llm_call", "error", attempt=attempt + 1, error=last_err,
                               elapsed_s=round(time.monotonic() - t, 2))
                time.sleep(2.0 * (attempt + 1))
                continue
            usage = data.get("usage") or {}
            pt, ct = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
            rt = int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.reasoning_tokens += rt
            choice = (data.get("choices") or [{}])[0]
            content = ((choice.get("message") or {}).get("content")) or ""
            ev = dict(attempt=attempt + 1, http_status=status, generation_id=data.get("id"),
                      provider=data.get("provider"), prompt_tokens=pt, completion_tokens=ct,
                      reasoning_tokens=rt, total_tokens=pt + ct, finish_reason=choice.get("finish_reason"),
                      elapsed_s=round(time.monotonic() - t, 2), output_chars=len(content))
            if status != 200 or data.get("error") or not content.strip():
                last_err = f"HTTP {status}: {json.dumps(data.get('error') or 'empty content')[:300]}"
                self.trace.log(stage, "llm_call", "error", error=last_err, **ev)
                if status in (400, 401, 402, 403, 404):
                    break  # not retryable
                time.sleep(2.0 * (attempt + 1))
                continue
            self.trace.log(stage, "llm_call", "ok", **ev)
            return content, choice.get("finish_reason")
        raise RuntimeError(f"LLM call failed: {last_err}")


# ----------------------------------------------------------------------------- parsing

_LATEX_ESC_CMDS = ("frac|tfrac|dfrac|theta|vartheta|tau|times|to|top|text|textbf|textit|textrm|texttt|tilde|"
                   "triangle|nabla|nu|neq|ne|neg|not|notin|rho|right|rangle|rightarrow|Rightarrow|rceil|"
                   "rfloor|rm|beta|bar|begin|binom|big|bigl|bigr|Big|Bigl|Bigr|bigg|boldsymbol|bot|bullet|"
                   "bf|bmatrix|forall|flat|underbrace|underline|uparrow|cup|cap|backslash|bmod|biggl|biggr")
_LATEX_FIX = re.compile(r"(?<!\\)\\(?=(?:%s)(?![A-Za-z]))" % _LATEX_ESC_CMDS)
_BAD_ESC = re.compile(r'(?<!\\)\\(?!["\\/bfnrtu])')
_BAD_U = re.compile(r"(?<!\\)\\u(?![0-9a-fA-F]{4})")


def _block(text, tag):
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S | re.I)
    if m:
        return m.group(1)
    m = re.search(rf"<{tag}>\s*(.*?)(?=<(?:spec|compute|patch)>|$)", text, re.S | re.I)  # unterminated block
    return m.group(1) if m else None


def _strip_fences(s):
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*\n", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _fix_control_chars(obj):
    """Undo JSON escapes that swallowed LaTeX commands (\\frac -> formfeed + 'rac', etc.)."""
    if isinstance(obj, str):
        return (obj.replace("\x0c", "\\f").replace("\x08", "\\b").replace("\t", "\\t")
                .replace("\x0b", "\\v"))
    if isinstance(obj, list):
        return [_fix_control_chars(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _fix_control_chars(v) for k, v in obj.items()}
    return obj


def loads_lenient(s):
    s = _strip_fences(s)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object found")
    s = s[i:j + 1]
    s = _LATEX_FIX.sub(r"\\\\", s)
    s = _BAD_ESC.sub(r"\\\\", s)
    s = _BAD_U.sub(r"\\\\u", s)
    s = re.sub(r'","\s*([}\]])', r'"\1', s)  # model glitch: stray `,"` before a closing bracket
    return _fix_control_chars(_loads_repairing(s))


def _balance_brackets(s):
    """Fix mismatched closers (e.g. `"]}` written for `"}]`) and close a truncated document."""
    pairs = {"{": "}", "[": "]"}
    out, stack, in_str, esc = [], [], False, False
    for ch in s:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in pairs:
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            ch = pairs[stack.pop()]
        out.append(ch)
    if in_str:
        out.append('"')
    out += [pairs[c] for c in reversed(stack)]
    return "".join(out)


def _loads_repairing(s):
    """json.loads with position-guided repairs for common LLM slips, then json_repair as a last resort."""
    try:
        return json.loads(s, strict=False)
    except json.JSONDecodeError:
        pass
    s = _balance_brackets(s)
    first_err = None
    for _ in range(25):
        try:
            return json.loads(s, strict=False)
        except json.JSONDecodeError as e:
            first_err = first_err or e
            pos, msg = e.pos, e.msg
            if msg.startswith("Extra data"):
                s = s[:pos]
                continue
            head = s[:pos].rstrip()
            if msg.startswith(("Expecting property name", "Expecting value")) and head.endswith(","):
                s = head[:-1] + s[pos:]  # trailing comma
                continue
            if msg.startswith("Expecting ',' delimiter") and pos > 0:
                # usually an unescaped quote inside a string: escape the quote that ended the string early
                q = s.rfind('"', 0, pos)
                if q > 0 and s[q - 1] != "\\":
                    s = s[:q] + '\\"' + s[q + 1:]
                    continue
            break
    try:
        import json_repair
        obj = json_repair.loads(s)
        if isinstance(obj, dict) and obj:
            return obj
    except Exception:
        pass
    raise first_err


def extract_compute(text):
    code = _block(text, "compute")
    if code is None:
        m = re.search(r"```(?:js|javascript)\s*\n(.*?)```", text, re.S)
        code = m.group(1) if m else None
    if code is None:
        m = re.search(r"(function\s+compute\s*\(.*)$", text, re.S)
        code = m.group(1) if m else None
    if code is None:
        return None
    return _strip_fences(code)


def parse_generation(text):
    spec_txt = _block(text, "spec")
    if spec_txt is None:
        cut = text.find("<compute>")
        spec_txt = text[:cut] if cut > 0 else text
    spec = loads_lenient(spec_txt)
    code = extract_compute(text)
    return spec, code


# ----------------------------------------------------------------------------- brief understanding

def extract_requirements(focus):
    sents = [x.strip() for x in re.split(r"(?<=[.!?;])\s+(?=[A-Z])", focus.strip()) if x.strip()]
    checks = [x for x in sents if re.match(r"(check|verify|ensure|confirm|make sure)\b", x, re.I)]
    reqs = [x for x in sents if x not in checks]
    return reqs, checks


def build_user_prompt(case, reqs, checks):
    lines = ["BRIEF"]
    for k, v in case.items():
        if isinstance(v, str) and v.strip():
            lines.append(f"{k}: {v.strip()}")
        elif v not in (None, "", [], {}):
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)[:4000]}")
    if reqs:
        lines.append("\nREQUIREMENTS FROM THE BRIEF (each must be visibly satisfied):")
        lines += [f"- {r}" for r in reqs]
    if checks:
        lines.append("\nREQUIRED CHECKS (each must become a passing test):")
        lines += [f"- {c}" for c in checks]
    lines.append("\nWrite the <spec> and <compute> blocks.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- normalization

def _num(x, default=None):
    try:
        if isinstance(x, bool):
            return default
        v = float(x)
        return v if v == v and abs(v) != float("inf") else default
    except (TypeError, ValueError):
        return default


def _s(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list):
        return " ".join(_s(i) for i in x)
    return str(x)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_control(c, warns):
    c = dict(c)
    t = str(c.get("type", "slider")).lower().strip()
    t = CONTROL_ALIASES.get(t, t)
    c["type"] = t
    cid = re.sub(r"[^A-Za-z0-9_]", "_", _s(c.get("id")) or "c")
    if cid[0].isdigit():
        cid = "c_" + cid
    c["id"] = cid
    c["label"] = plain_label(_s(c.get("label")) or cid)
    c["help"] = plain_label(_s(c.get("help")))
    if t in ("slider", "number", "vector", "matrix"):
        lo, hi = _num(c.get("min"), None), _num(c.get("max"), None)
        if lo is None or hi is None or lo >= hi:
            dv = c.get("default")
            flat = []
            if isinstance(dv, list):
                for r in dv:
                    flat += r if isinstance(r, list) else [r]
            else:
                flat = [dv]
            flat = [f for f in (_num(x) for x in flat) if f is not None] or [0.0]
            lo = lo if lo is not None else min(min(flat), 0.0) - (0 if min(flat) >= 0 else abs(min(flat)))
            hi = hi if hi is not None and hi > lo else max(max(flat) * 2, lo + 1)
        c["min"], c["max"] = lo, hi
        step = _num(c.get("step"), None)
        ints = all(float(x).is_integer() for x in (lo, hi) + tuple(_flat_nums(c.get("default"))))
        c["step"] = step if step and step > 0 else (1 if ints or (hi - lo) >= 20 else 0.01 if (hi - lo) <= 2 else 0.1)
    if t in ("slider", "number"):
        c["default"] = _clamp(_num(c.get("default"), c["min"]), c["min"], c["max"])
    elif t == "toggle":
        d = c.get("default")
        c["default"] = d if isinstance(d, bool) else str(d).lower() in ("true", "1", "on", "yes")
    elif t == "select":
        opts = []
        for o in c.get("options") or []:
            if isinstance(o, dict):
                v = _typed_option(o.get("value") if o.get("value") is not None else o.get("label"))
                opts.append({"value": v, "label": plain_label(_s(o.get("label")) or _s(v))})
            else:
                opts.append({"value": _typed_option(o), "label": plain_label(_s(o))})
        if not opts:
            warns.append(f"select control '{cid}' has no options")
            opts = [{"value": "default", "label": "default"}]
        c["options"] = opts
        c["default"] = _match_option(opts, c.get("default"))
    elif t == "vector":
        d = c.get("default")
        d = [_clamp(_num(x, c["min"]), c["min"], c["max"]) for x in (d if isinstance(d, list) else [d])]
        c["default"] = d or [c["min"]]
        n = len(c["default"])
        mn = int(_num(c.get("min_len"), n))
        mx = int(_num(c.get("max_len"), n))
        c["min_len"], c["max_len"] = max(1, min(mn, n)), max(mx, n)
    elif t == "matrix":
        d = c.get("default")
        if not (isinstance(d, list) and d and all(isinstance(r, list) and r for r in d)):
            d = [[0, 0], [0, 0]]
            warns.append(f"matrix control '{cid}' had an invalid default")
        ncol = max(len(r) for r in d)
        d = [[_clamp(_num(x, 0.0), c["min"], c["max"]) for x in (r + [0] * (ncol - len(r)))] for r in d]
        c["default"] = d
        if c.get("resizable"):
            c["min_rows"] = int(_num(c.get("min_rows"), 1)); c["max_rows"] = int(_num(c.get("max_rows"), max(len(d), 4)))
            c["min_cols"] = int(_num(c.get("min_cols"), 1)); c["max_cols"] = int(_num(c.get("max_cols"), max(ncol, 4)))
            c["min_rows"] = min(c["min_rows"], len(d)); c["max_rows"] = max(c["max_rows"], len(d))
            c["min_cols"] = min(c["min_cols"], ncol); c["max_cols"] = max(c["max_cols"], ncol)
        else:
            c["resizable"] = False
    return c


def _flat_nums(v):
    if isinstance(v, list):
        return [y for x in v for y in _flat_nums(x)]
    x = _num(v)
    return [x] if x is not None else []


def _match_option(opts, v, fallback="first"):
    for o in opts:
        if o["value"] == v or _s(o["value"]) == _s(v):
            return o["value"]
    return opts[0]["value"] if fallback == "first" else fallback


def _typed_option(v):
    """Select option values keep a numeric type when they are numbers (template maps index -> typed value)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    return _s(v)


def coerce_value(c, v, clamp=True):
    """Coerce a preset/test value to the control's type (and range, for presets). Returns (value, ok)."""
    t = c["type"]
    if t in ("slider", "number"):
        x = _num(v)
        if x is None:
            return c["default"], False
        return (_clamp(x, c["min"], c["max"]) if clamp else x), True
    if t == "toggle":
        return (v if isinstance(v, bool) else str(v).lower() in ("true", "1", "on", "yes")), True
    if t == "select":
        m = _match_option(c["options"], v, None)
        return (m, True) if m is not None else (c["default"], False)
    if t == "vector":
        if not isinstance(v, list) or not v:
            return c["default"], False
        vec = [_clamp(_num(x, c["min"]), c["min"], c["max"]) for x in v]
        if not (c["min_len"] <= len(vec) <= c["max_len"]):
            c["min_len"], c["max_len"] = min(c["min_len"], len(vec)), max(c["max_len"], len(vec))
        return vec, True
    if t == "matrix":
        if not (isinstance(v, list) and v and all(isinstance(r, list) and r for r in v)):
            return c["default"], False
        ncol = max(len(r) for r in v)
        m = [[_clamp(_num(x, 0.0), c["min"], c["max"]) for x in (r + [0] * (ncol - len(r)))] for r in v]
        d = c["default"]
        if (len(m), ncol) != (len(d), len(d[0])) and not c.get("resizable"):
            c["resizable"] = True
            c["min_rows"], c["max_rows"] = min(len(m), len(d)), max(len(m), len(d))
            c["min_cols"], c["max_cols"] = min(ncol, len(d[0])), max(ncol, len(d[0]))
        return m, True
    return v, True


_LOC_NUM = re.compile(r"(?:section|sec\.?|§|eq(?:uation)?\.?|algorithm|theorem|proposition|lemma|table|figure|fig\.?)"
                      r"\s*\(?([0-9]+(?:\.[0-9]+)*)\)?", re.I)


def ground_locations(paper, focus):
    """Section/equation numbers the brief does not state are a hallucination risk: keep only verifiable ones."""
    stated = {m.group(1) for m in _LOC_NUM.finditer(focus or "")}
    out = dict(paper)
    for k in ("section", "equation"):
        val = out.get(k) or ""
        nums = [m.group(1) for m in _LOC_NUM.finditer(val)] or re.findall(r"\b\d+(?:\.\d+)*\b", val)
        if nums and not any(n in stated for n in nums):
            # strip the unverifiable number, keep a descriptive remainder if any
            desc = re.sub(r"\(([^()]*)\)", r"\1", _LOC_NUM.sub("", val))
            desc = re.sub(r"\b\d+(?:\.\d+)*\b", "", desc).strip(" ,;:-()")
            out[k] = desc if (k == "section" and len(desc) > 3) else ""
    if not out.get("section"):
        sec = [m.group(0) for m in _LOC_NUM.finditer(focus or "") if m.group(0).lower().startswith(("sec", "§"))]
        if sec:
            out["section"] = sec[0]
    return out


def normalize_spec(spec, warns):
    if not isinstance(spec, dict):
        raise ValueError("spec is not an object")
    out = dict(spec)
    for k in ("title", "subtitle", "idea", "why_it_matters"):
        out[k] = _s(out.get(k))
    out["plan"] = out.get("plan") if isinstance(out.get("plan"), dict) else {}
    p = out.get("paper") if isinstance(out.get("paper"), dict) else {}
    out["paper"] = {k: plain_label(_s(p.get(k))) for k in ("title", "authors", "year", "section", "equation")}
    if len(out["paper"]["equation"]) > 40 or "=" in out["paper"]["equation"]:
        out["paper"]["equation"] = ""  # a formula, not a location label: the key equation is shown separately
    eq = out.get("equation")
    if isinstance(eq, str):
        eq = {"latex": eq}
    eq = eq if isinstance(eq, dict) else {}
    out["equation"] = {"latex": _s(eq.get("latex")).strip("$ "), "caption": _s(eq.get("caption"))}
    out["symbols"] = [{"latex": _s(x.get("latex") or x.get("symbol")).strip("$ "), "meaning": _s(x.get("meaning")),
                       "where": _s(x.get("where"))} for x in (out.get("symbols") or []) if isinstance(x, dict)]
    out["steps"] = [{"title": _s(x.get("title")), "text": _s(x.get("text"))}
                    for x in (out.get("steps") or []) if isinstance(x, dict)]
    ctrls, seen, alias = [], set(), {}
    for c in out.get("controls") or []:
        if not isinstance(c, dict):
            continue
        orig_id = _s(c.get("id"))
        c = normalize_control(c, warns)
        alias[orig_id] = c["id"]
        if c["type"] not in CONTROL_TYPES:
            warns.append(f"control '{c['id']}' has unknown type '{c['type']}', treated as slider")
            c["type"] = "slider"
            c = normalize_control(c, warns)
        if c["id"] in seen:
            warns.append(f"duplicate control id '{c['id']}' dropped")
            continue
        seen.add(c["id"])
        ctrls.append(c)
    out["controls"] = ctrls
    out["readouts"] = [{"key": _s(r.get("key")), "label": plain_label(_s(r.get("label")) or _s(r.get("key"))),
                        "unit": plain_label(_s(r.get("unit"))), "digits": int(_num(r.get("digits"), 3))}
                       for r in (out.get("readouts") or []) if isinstance(r, dict) and _s(r.get("key"))]
    views = []
    for i, v in enumerate(out.get("views") or []):
        if not isinstance(v, dict):
            continue
        t = str(v.get("type", "")).lower().strip()
        t = VIEW_ALIASES.get(t, t)
        views.append({"id": re.sub(r"[^A-Za-z0-9_-]", "_", _s(v.get("id")) or f"view{i}"), "type": t,
                      "title": plain_label(_s(v.get("title"))), "caption": _s(v.get("caption")),
                      "data": _s(v.get("data"))})
    out["views"] = views
    out["invariants"] = [{"label": plain_label(_s(x.get("label"))), "expr": _s(x.get("expr"))}
                         for x in (out.get("invariants") or []) if isinstance(x, dict) and _s(x.get("expr"))]
    out["tests"] = [{"label": plain_label(_s(x.get("label"))), "state": x.get("state") if isinstance(x.get("state"), dict) else {},
                     "expect": _s(x.get("expect"))}
                    for x in (out.get("tests") or []) if isinstance(x, dict) and _s(x.get("expect"))]
    exps = []
    for x in out.get("explorations") or []:
        if not isinstance(x, dict):
            continue
        steps = [{"label": plain_label(_s(st.get("label"))),
                  "preset": st.get("preset") if isinstance(st.get("preset"), dict) else {},
                  "expect": _s(st.get("expect"))} for st in (x.get("steps") or []) if isinstance(st, dict)]
        if not steps and isinstance(x.get("preset"), dict) and x["preset"]:
            steps = [{"label": "", "preset": x["preset"], "expect": _s(x.get("expect"))}]
        exps.append({"title": plain_label(_s(x.get("title"))), "change": _s(x.get("change")),
                     "observe": _s(x.get("observe")), "why": _s(x.get("why")), "steps": steps[:3]})
    out["explorations"] = exps[:3]
    cv = out.get("caveat") if isinstance(out.get("caveat"), dict) else {}
    kind = _s(cv.get("kind")).lower()
    kind = kind if kind in ("limitation", "assumption", "misconception") else "misconception"
    out["caveat"] = {"kind": kind, "title": plain_label(_s(cv.get("title"))), "text": _s(cv.get("text"))}
    g = out.get("grounding") if isinstance(out.get("grounding"), dict) else {}
    out["grounding"] = {"from_paper": [_s(x) for x in (g.get("from_paper") or []) if _s(x)],
                        "simplifications": [_s(x) for x in (g.get("simplifications") or []) if _s(x)],
                        "not_claimed": _s(g.get("not_claimed"))}
    # presets / test states: coerce to control types & ranges, drop unknown ids
    cmap = {c["id"]: c for c in ctrls}
    step_items = [st for e in out["explorations"] for st in e["steps"]]
    for kind_name, items, key in (("exploration", step_items, "preset"), ("test", out["tests"], "state")):
        for it in items:
            clean = {}
            for k, v in (it.get(key) or {}).items():
                k = alias.get(k, k)
                if k not in cmap:
                    warns.append(f"{kind_name} '{it.get('title') or it.get('label')}' sets unknown control '{k}' (dropped)")
                    continue
                val, ok = coerce_value(cmap[k], v, clamp=(kind_name != "test"))
                if not ok:
                    warns.append(f"{kind_name} value for '{k}' was invalid; default used")
                clean[k] = val
            it[key] = clean
    return out


# ----------------------------------------------------------------------------- JS execution (validation)

JS_HARNESS = r"""
var console = (typeof console !== 'undefined') ? console : {log:function(){},warn:function(){},error:function(){},info:function(){}};
function __sig(v){ return JSON.stringify(v, function(k,x){ return (typeof x==='number' && !isFinite(x)) ? ('__NF__'+String(x)) : x; }); }
function __nonfinite(v, path, acc){
  if (acc.length > 12) return acc;
  if (typeof v === 'number') { if (!isFinite(v)) acc.push(path || '(root)'); }
  else if (Array.isArray(v)) { for (var i=0;i<v.length;i++) __nonfinite(v[i], path+'['+i+']', acc); }
  else if (v && typeof v === 'object') { for (var k in v) __nonfinite(v[k], path ? path+'.'+k : k, acc); }
  return acc;
}
function __runJobs(jobsJson){
  var jobs = JSON.parse(jobsJson), res = [];
  for (var j=0; j<jobs.length; j++){
    var job = jobs[j], r = {};
    try {
      if (typeof compute !== 'function') throw new Error('compute is not defined as a function');
      var out = compute(JSON.parse(JSON.stringify(job.state)));
      if (out === null || typeof out !== 'object' || Array.isArray(out)) throw new Error('compute must return a plain object, got ' + (Array.isArray(out) ? 'array' : typeof out));
      r.ok = true; r.sig = __sig(out); r.nonfinite = __nonfinite(out, '', []);
      r.warning = (typeof out.warning === 'string' && out.warning) ? out.warning : '';
      var prev = {};
      if (job.prev_state) { try { prev = compute(JSON.parse(JSON.stringify(job.prev_state))) || {}; } catch(e2) { prev = {}; } }
      r.vals = (job.exprs || []).map(function(e){
        try { var f = new Function('out','s','prev','"use strict"; return (' + e + ');');
              var v = f(out, JSON.parse(JSON.stringify(job.state)), prev);
              return {ok:true, pass: v === true, val: (typeof v === 'boolean') ? v : String(v).slice(0,60),
                      num: (typeof v === 'number' && isFinite(v)) ? v : null,
                      arr: (Array.isArray(v) && v.length <= 16 && v.every(function(x){ return typeof x === 'number' && isFinite(x); })) ? v : null,
                      str: (typeof v === 'string') ? v.slice(0,80) : null}; }
        catch(err){ return {ok:false, pass:false, err: String((err && err.message) || err).slice(0,200)}; }
      });
    } catch(err){ r.ok = false; r.err = String((err && err.message) || err).slice(0,300); }
    res.push(r);
  }
  return JSON.stringify(res);
}
"""


class JSRunner:
    """Executes LLM-written compute code: embedded V8 (mini-racer) first, Node.js as fallback."""

    def __init__(self):
        self.kind = None
        try:
            from py_mini_racer import MiniRacer  # noqa: F401
            self.kind = "v8"
        except Exception:
            import shutil
            if shutil.which("node"):
                self.kind = "node"

    def syntax_and_run(self, code, jobs):
        """Returns (results | None, error | None)."""
        if self.kind == "v8":
            from py_mini_racer import MiniRacer
            ctx = MiniRacer()
            try:
                ctx.eval(JS_HARNESS + "\n" + code + "\n;0", timeout=3000)
            except Exception as e:
                return None, "compute code failed to load: " + _short_err(e)
            try:
                raw = ctx.eval("__runJobs(" + json.dumps(json.dumps(jobs)) + ")", timeout=8000)
                return json.loads(raw), None
            except Exception as e:
                return None, "compute execution failed or timed out: " + _short_err(e)
        if self.kind == "node":
            import subprocess
            import tempfile
            src = (code + "\n" + JS_HARNESS +
                   "\nprocess.stdout.write(__runJobs(require('fs').readFileSync(0,'utf8')));\n")
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
                f.write(src)
                path = f.name
            try:
                p = subprocess.run(["node", path], input=json.dumps(jobs), capture_output=True,
                                   text=True, timeout=20, encoding="utf-8")
                if p.returncode != 0:
                    return None, "compute code failed: " + p.stderr.strip()[-400:]
                return json.loads(p.stdout), None
            except Exception as e:
                return None, "node execution failed: " + _short_err(e)
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        return None, "no JS engine available"


def _short_err(e):
    s = str(e).strip().splitlines()
    s = [x for x in s if x.strip()]
    return (" | ".join(s[:3]))[:300]


# ----------------------------------------------------------------------------- validation

def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _numlist(x):
    return isinstance(x, list) and all(_is_num(v) or v is None or (isinstance(v, str) and v.startswith("__NF__")) for v in x)


def adapt_view_data(t, d, title=""):
    """Mirror of the template's tolerance for common shape slips (bare arrays etc.)."""
    if isinstance(d, list) and d:
        if t == "heatmap" and isinstance(d[0], list):
            return {"matrix": d}
        if t == "bars" and _numlist(d):
            return {"labels": [str(i + 1) for i in range(len(d))], "values": d}
        if t == "pipeline":
            return {"stages": [x if isinstance(x, dict) else {"label": f"Step {i + 1}", "value": x} for i, x in enumerate(d)]}
        if t == "table" and isinstance(d[0], list):
            return {"columns": [f"c{i + 1}" for i in range(len(d[0]))], "rows": d}
    if t == "line" and isinstance(d, dict):
        d = dict(d)
        if not isinstance(d.get("series"), list):
            if isinstance(d.get("y"), list):
                d["series"] = [{"name": title or "y", "y": d["y"]}]
            else:
                n = len(d["x"]) if isinstance(d.get("x"), list) else None
                d["series"] = [{"name": k, "y": v} for k, v in d.items()
                               if k != "x" and _numlist(v) and len(v) > 1 and (n is None or len(v) == n)]
        if not isinstance(d.get("x"), list) and d["series"] and isinstance(d["series"][0], dict):
            y0 = d["series"][0].get("y") or []
            d["x"] = list(range(len(y0)))
        return d
    return d


def check_view_shape(v, d):
    t = v["type"]
    d = adapt_view_data(t, d, v.get("title", ""))
    if not isinstance(d, dict):
        return f"out.{v['data']} must be an object for a {t} view, got {type(d).__name__}"
    if t == "bars":
        labels = d.get("labels")
        if not isinstance(labels, list) or not labels:
            return "bars needs non-empty labels[]"
        if "values" in d:
            if not _numlist(d["values"]) or len(d["values"]) != len(labels):
                return "bars values[] must be numbers with the same length as labels[]"
        elif isinstance(d.get("series"), list) and d["series"]:
            for s in d["series"]:
                if not isinstance(s, dict) or not _numlist(s.get("values")) or len(s["values"]) != len(labels):
                    return "each bars series needs values[] with the same length as labels[]"
        else:
            return "bars needs values[] or series[]"
    elif t == "line":
        x = d.get("x")
        if not _numlist(x) or len(x) < 2:
            return "line needs numeric x[] with at least 2 points"
        ser = d.get("series")
        if not isinstance(ser, list) or not ser:
            return "line needs series:[{name,y}]"
        for s in ser:
            if not isinstance(s, dict) or not _numlist(s.get("y")) or len(s["y"]) != len(x):
                return "each line series needs y[] numbers with the same length as x[]"
    elif t == "heatmap":
        m = d.get("matrix")
        if not (isinstance(m, list) and m and all(isinstance(r, list) and r and _numlist(r) for r in m)):
            return "heatmap needs matrix as a non-empty 2-D numeric array"
        if len({len(r) for r in m}) != 1:
            return "heatmap matrix rows must have equal length"
        if len(m) > 64 or len(m[0]) > 64:
            return "heatmap too large (max 64x64)"
        for key, n in (("row_labels", len(m)), ("col_labels", len(m[0]))):
            if key in d and d[key] is not None and (not isinstance(d[key], list) or len(d[key]) != n):
                return f"heatmap {key} length must match the matrix"
    elif t == "plane":
        if not any(isinstance(d.get(k), list) and d.get(k) for k in ("vectors", "points", "paths")):
            return "plane needs vectors[], points[] or paths[]"
        for k in ("vectors", "points"):
            for p in d.get(k) or []:
                if not isinstance(p, dict) or not _is_num(p.get("x")) or not _is_num(p.get("y")):
                    return f"plane {k} items need numeric x and y"
    elif t == "graph":
        nodes = d.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            return "graph needs nodes[]"
        ids = {str(n.get("id")) for n in nodes if isinstance(n, dict)}
        for e in d.get("edges") or []:
            if not isinstance(e, dict) or str(e.get("from")) not in ids or str(e.get("to")) not in ids:
                return "graph edges must reference existing node ids"
    elif t == "pipeline":
        st = d.get("stages")
        if not isinstance(st, list) or not st or not all(isinstance(x, dict) and x.get("label") for x in st):
            return "pipeline needs stages:[{label,value,note}]"
    elif t == "table":
        cols, rows = d.get("columns"), d.get("rows")
        if not isinstance(cols, list) or not cols or not isinstance(rows, list):
            return "table needs columns[] and rows[]"
        for r in rows:
            if not isinstance(r, list) or len(r) != len(cols):
                return "every table row must have one cell per column"
    return None


def _alt_values(c, base_val):
    t = c["type"]
    if t in ("slider", "number"):
        lo, hi = c["min"], c["max"]
        st = c.get("step") or 0.01
        vals = [lo, hi, round(lo + round((hi - lo) * 0.37 / st) * st, 6)]
        if c.get("step", 1) >= 1 and lo <= 0 <= hi:
            vals.append(0)
        return [v for v in vals if v != base_val]
    if t == "toggle":
        return [not base_val]
    if t == "select":
        return [o["value"] for o in c["options"] if o["value"] != base_val]
    if t == "vector":
        n = len(base_val) if isinstance(base_val, list) and base_val else len(c["default"])
        lo, hi = c["min"], c["max"]
        zero = 0 if lo <= 0 <= hi else lo
        alts = [[zero] * n, [hi] * n, [hi] + [zero] * (n - 1),
                [round(lo + (hi - lo) * (i + 1) / (n + 1), 6) for i in range(n)]]
        if c["min_len"] < n:
            alts.append(list(base_val[:c["min_len"]]))
        if c["max_len"] > n:
            alts.append(list(base_val) + [base_val[-1]] * (c["max_len"] - n))
        return [a for a in alts if a != base_val]
    if t == "matrix":
        m = base_val if isinstance(base_val, list) and base_val else c["default"]
        r, k = len(m), len(m[0])
        lo, hi = c["min"], c["max"]
        zero = 0 if lo <= 0 <= hi else lo
        bumped = [row[:] for row in m]
        bumped[0][0] = hi if bumped[0][0] != hi else lo
        alts = [[[zero] * k for _ in range(r)], [[hi] * k for _ in range(r)], bumped]
        if c.get("resizable"):
            alts.append([[1] * c["min_cols"] for _ in range(c["min_rows"])])
            alts.append([[(i + j) % 3 - 1 for j in range(c["max_cols"])] for i in range(c["max_rows"])])
        return [a for a in alts if a != m]
    return []


def _random_value(c, rng):
    t = c["type"]
    step = c.get("step") or 0.01

    def rnd():
        v = rng.uniform(c["min"], c["max"])
        v = round(round((v - c["min"]) / step) * step + c["min"], 6)
        return _clamp(v, c["min"], c["max"])

    if t in ("slider", "number"):
        return rnd()
    if t == "toggle":
        return rng.random() < 0.5
    if t == "select":
        return rng.choice(c["options"])["value"]
    if t == "vector":
        n = rng.randint(c["min_len"], c["max_len"])
        return [rnd() for _ in range(n)]
    if t == "matrix":
        if c.get("resizable"):
            r, k = rng.randint(c["min_rows"], c["max_rows"]), rng.randint(c["min_cols"], c["max_cols"])
        else:
            r, k = len(c["default"]), len(c["default"][0])
        return [[rnd() for _ in range(k)] for _ in range(r)]
    return c.get("default")


class Issue:
    def __init__(self, sev, code, msg, ref=None):
        self.sev, self.code, self.msg, self.ref = sev, code, msg, ref

    def as_dict(self):
        return {"severity": self.sev, "check": self.code, "message": self.msg}


def _brief_state(st):
    s = json.dumps(st, ensure_ascii=False)
    return s if len(s) < 220 else s[:220] + "..."


def validate(spec, code, runner, required_checks):
    """Deterministic checks. Returns (issues, summary list of {name, ok, detail}, default_out)."""
    issues, summary = [], []

    def add(sev, code_, msg, ref=None):
        issues.append(Issue(sev, code_, msg, ref))

    # --- structure
    for k, label in (("title", "title"), ("idea", "idea"), ("why_it_matters", "why_it_matters")):
        if not spec.get(k):
            add("error", "structure", f"missing {label}")
    if not spec["equation"]["latex"]:
        add("error", "structure", "missing equation.latex")
    if len(spec["symbols"]) < 2:
        add("error", "structure", "need at least 2 symbols with meanings")
    if control_count(spec["controls"]) < 2:
        add("error", "controls", f"need at least 2 controls, found {len(spec['controls'])}")
    if not spec["views"]:
        add("error", "views", "need at least 1 view")
    if len(spec["explorations"]) < 2:
        add("error", "explorations", f"need exactly 2 guided explorations, found {len(spec['explorations'])}")
    for i, e in enumerate(spec["explorations"]):
        miss = [k for k in ("change", "observe", "why") if not e[k]]
        if miss:
            add("error", "explorations", f"exploration {i + 1} missing {', '.join(miss)}")
        if not e["steps"] or not all(st["preset"] for st in e["steps"]):
            add("error", "explorations", f"exploration {i + 1} needs steps with non-empty preset control values")
        if not any(st["expect"] for st in e["steps"]):
            add("warn", "explorations", f"exploration {i + 1} has no expect expression")
    if not spec["caveat"]["text"]:
        add("error", "structure", "missing caveat (limitation/assumption/misconception)")
    if not spec["grounding"]["from_paper"] or not spec["grounding"]["simplifications"]:
        add("error", "grounding", "grounding needs both from_paper[] and simplifications[]")
    if not (spec["paper"]["section"] or spec["paper"]["equation"]):
        add("error", "grounding", "paper.section or paper.equation must identify the relevant part of the paper")
    if not spec["readouts"]:
        add("warn", "readouts", "no readouts (key live numbers)")
    if required_checks and len(spec["tests"]) < len(required_checks):
        add("error", "tests", f"brief requires {len(required_checks)} checks but only {len(spec['tests'])} tests exist: "
            + " | ".join(required_checks))
    for v in spec["views"]:
        if v["type"] not in VIEW_TYPES:
            add("error", "views", f"view '{v['id']}' has unsupported type '{v['type']}' (use one of {sorted(VIEW_TYPES)})")
        if not v["data"]:
            add("error", "views", f"view '{v['id']}' has no data key")
    # latex
    try:
        latex_to_mathml(spec["equation"]["latex"], display=True, strict=True)
    except Exception as e:
        add("error", "latex", f"equation.latex does not parse ({_short_err(e)}); use standard LaTeX")
    bad_sym = []
    for s in spec["symbols"]:
        try:
            latex_to_mathml(s["latex"], strict=True)
        except Exception:
            bad_sym.append(s["latex"])
    if bad_sym:
        add("warn", "latex", f"symbols not parseable: {bad_sym}")
    summary.append({"name": "Spec structure (sections, ≥2 controls, 2 explorations, grounding)",
                    "ok": not any(i.sev == "error" and i.code in ("structure", "controls", "explorations", "grounding", "views")
                                  for i in issues), "detail": ""})

    # --- execution
    if not code:
        add("error", "compute", "no compute function was provided")
        return issues, summary, None
    if not re.search(r"function\s+compute\s*\(|compute\s*=\s*(function|\()", code):
        add("error", "compute", "code must define function compute(s)")
    if runner.kind is None:
        summary.append({"name": "Executable compute checks", "ok": True, "detail": "skipped: no JS engine"})
        return issues, summary, None

    ctrls = spec["controls"]
    defaults = {c["id"]: c["default"] for c in ctrls}
    inv_exprs = [x["expr"] for x in spec["invariants"]]
    jobs, meta = [], []

    def job(state, kind, label, exprs=(), prev_state=None):
        jobs.append({"state": state, "exprs": list(inv_exprs) + list(exprs), "prev_state": prev_state})
        meta.append((kind, label, len(exprs)))

    job(defaults, "default", "defaults")
    bases = [("defaults", defaults)]
    for i, e in enumerate(spec["explorations"]):
        for j, stp in enumerate(e["steps"]):
            st = {**defaults, **stp["preset"]}
            prev_st = {**defaults, **e["steps"][j - 1]["preset"]} if j > 0 else None
            job(st, "exploration", (i, j), [stp["expect"]] if stp["expect"] else [], prev_st)
            bases.append((f"exploration {i + 1} step {j + 1}", st))
    for i, t in enumerate(spec["tests"]):
        st = {**defaults, **t["state"]}
        job(st, "test", i, [t["expect"]])
        bases.append((f"test '{t['label']}'", st))
    # sensitivity + edge values
    for c in ctrls:
        for bname, bst in bases[:4]:
            for alt in _alt_values(c, bst[c["id"]])[:6]:
                job({**bst, c["id"]: alt}, "edge", (c["id"], bname, alt))
    rng = random.Random(20240601)
    fuzz_states = [{c["id"]: _random_value(c, rng) for c in ctrls} for _ in range(24)]
    for k, st in enumerate(fuzz_states):
        job(st, "fuzz", k)
        bases.append((f"random state {k}", st))
    # sensitivity on random states too (a control may only matter in some regimes)
    for c in ctrls:
        for k, st in enumerate(fuzz_states[:6]):
            for alt in _alt_values(c, st[c["id"]])[:2]:
                job({**st, c["id"]: alt}, "edge", (c["id"], f"random state {k}", alt))

    results, err = runner.syntax_and_run(code, jobs)
    if err:
        add("error", "compute", err)
        summary.append({"name": "compute() loads and runs", "ok": False, "detail": err})
        return issues, summary, None

    default_out = None
    base_sig = {}
    n_inv = len(inv_exprs)
    inv_fail, exceptions, nonfinite = {}, [], []
    sens = {c["id"]: False for c in ctrls}
    edge_sigs = []
    tests_ok, expl_ok = [], []
    for (kind, label, n_extra), job_, res in zip(meta, jobs, results):
        if not res.get("ok"):
            exceptions.append((kind, label, job_["state"], res.get("err")))
            if kind == "test":
                tests_ok.append(False)
            if kind == "exploration":
                expl_ok.append(False)
            continue
        vals = res.get("vals") or []
        warned = bool(res.get("warning"))
        if res.get("nonfinite") and not warned:
            nonfinite.append((kind, label, job_["state"], res["nonfinite"]))
        for k in range(n_inv):
            if k < len(vals) and not vals[k].get("pass") and not warned:
                inv_fail.setdefault(k, (kind, label, job_["state"], vals[k]))
        extra = vals[n_inv:]
        if kind == "default":
            default_out = json.loads(res["sig"])
            base_sig["defaults"] = res["sig"]
        elif kind == "exploration":
            i, j = label
            base_sig[f"exploration {i + 1} step {j + 1}"] = res["sig"]
            ok = all(v.get("pass") for v in extra)
            expl_ok.append(ok)
            if not ok:
                e, stp = spec["explorations"][i], spec["explorations"][i]["steps"][j]
                add("error", "explorations", f"exploration {i + 1} step {j + 1} ('{e['title']}' / '{stp['label']}') expect "
                    f"`{stp['expect']}` is not true at its preset {_brief_state(stp['preset'])}: "
                    f"{extra[0].get('err') or 'got ' + str(extra[0].get('val'))}. out = {res['sig'][:500]}")
        elif kind == "test":
            base_sig[f"test '{spec['tests'][label]['label']}'"] = res["sig"]
            ok = all(v.get("pass") for v in extra)
            tests_ok.append(ok)
            if not ok:
                t = spec["tests"][label]
                add("error", "tests", f"test '{t['label']}' failed: expect `{t['expect']}` with state {_brief_state(t['state'])} "
                    f"-> {extra[0].get('err') or 'got ' + str(extra[0].get('val'))}. out = {res['sig'][:500]}")
        elif kind == "fuzz":
            base_sig[f"random state {label}"] = res["sig"]
        elif kind == "edge":
            edge_sigs.append((label, res["sig"]))
    for (cid, bname, alt), sig in edge_sigs:
        if bname in base_sig and sig != base_sig[bname]:
            sens[cid] = True
    for i, e in enumerate(spec["explorations"]):
        sigs = [base_sig.get(f"exploration {i + 1} step {j + 1}") for j in range(len(e["steps"]))]
        sigs = [x for x in sigs if x]
        if sigs and all(x == base_sig.get("defaults") for x in sigs):
            add("error", "explorations", f"exploration {i + 1} ('{e['title']}'): none of its steps changes anything "
                "relative to the defaults; set presets that show the effect (e.g. a before/after pair)")
        elif len(e["steps"]) == 1 and sigs and sigs[0] == base_sig.get("defaults"):
            add("warn", "explorations", f"exploration {i + 1} ('{e['title']}'): its preset equals the default state, so "
                "setting it up changes nothing; choose a preset that visibly shows the effect")
        elif len(sigs) > 1 and len(set(sigs)) == 1:
            add("warn", "explorations", f"exploration {i + 1} ('{e['title']}'): all its steps produce identical outputs; "
                "each step must change the observed quantity")
    _, ph_errors = eval_placeholders(spec, code, runner)
    for field, msg in ph_errors[:6]:
        add("error", "placeholders", msg, ref=field)
    if exceptions:
        k, lab, st, e = exceptions[0]
        add("error", "robustness", f"compute threw for {len(exceptions)} input states, e.g. {_brief_state(st)}: {e}")
    if nonfinite:
        k, lab, st, paths = nonfinite[0]
        add("error", "robustness", f"NaN/Infinity in out ({', '.join(paths[:5])}) for {len(nonfinite)} input states, e.g. "
            f"{_brief_state(st)}; handle the degenerate case and set out.warning")
    for k, (kind, lab, st, v) in inv_fail.items():
        add("error", "invariants", f"invariant '{spec['invariants'][k]['label']}' (`{inv_exprs[k]}`) fails for state "
            f"{_brief_state(st)}: {v.get('err') or 'got ' + str(v.get('val'))}", ref=k)
    for cid, moved in sens.items():
        if not moved:
            add("error", "controls", f"control '{cid}' never changes any output; make it affect the computation or remove it",
                ref=cid)

    if default_out is not None:
        for r in spec["readouts"]:
            v = default_out.get(r["key"])
            if v is None:
                add("error", "readouts", f"readout key '{r['key']}' missing from out at defaults", ref=r["key"])
            elif not (_is_num(v) or isinstance(v, (str, bool)) or _numlist(v)):
                add("error", "readouts", f"readout '{r['key']}' must be a number, got {type(v).__name__}", ref=r["key"])
        for v in spec["views"]:
            if v["type"] in VIEW_TYPES and v["data"]:
                msg = check_view_shape(v, default_out.get(v["data"]))
                if msg:
                    add("error", "views", f"view '{v['id']}' ({v['type']}, data '{v['data']}'): {msg}", ref=v["id"])
    else:
        add("error", "compute", "compute(defaults) did not return an object")

    summary.append({"name": "compute() runs at defaults, all presets and test states", "ok": not exceptions,
                    "detail": f"{len(jobs)} executions"})
    summary.append({"name": "No NaN/Infinity on edge and random inputs", "ok": not nonfinite, "detail": f"{len(jobs)} states"})
    summary.append({"name": "Invariants hold on every state", "ok": not inv_fail, "detail": f"{n_inv} invariants"})
    summary.append({"name": "Every control changes the output", "ok": all(sens.values()),
                    "detail": ", ".join(f"{k}{'' if v else ' (no effect)'}" for k, v in sens.items())})
    summary.append({"name": "Brief-derived tests pass", "ok": bool(tests_ok) and all(tests_ok),
                    "detail": f"{sum(tests_ok)}/{len(tests_ok)}"})
    summary.append({"name": "Exploration presets produce the described effect", "ok": bool(expl_ok) and all(expl_ok),
                    "detail": f"{sum(expl_ok)}/{len(expl_ok)}"})
    summary.append({"name": "Readouts and view data have valid shapes", "ok": not any(i.code in ("readouts", "views") for i in issues),
                    "detail": ""})
    return issues, summary, default_out


SOFT_CODES = ("invariants", "controls", "readouts", "views")


def lock_sizes_fix(spec, code, issues, summary, default_out, runner, checks, trace, rnd):
    """compute() crashing or producing NaN when resizable matrices get mismatched shapes is the most common
    robustness failure; locking matrix sizes (the brief rarely needs resizing them) removes it without an LLM call."""
    if not any(i.sev == "error" and i.code == "robustness" for i in issues):
        return spec, issues, summary, default_out
    mats = [c["id"] for c in spec["controls"] if c["type"] == "matrix" and c.get("resizable")]
    if not mats:
        return spec, issues, summary, default_out
    fixed = json.loads(json.dumps(spec))
    for c in fixed["controls"]:
        if c["id"] in mats:
            c["resizable"] = False
    f_issues, f_summary, f_out = validate(fixed, code, runner, checks)
    ok = badness(f_issues) < badness(issues)
    trace.log("revise", "deterministic_fix", "accepted" if ok else "rejected", round=rnd,
              fixes=[f"locked the size of matrix '{m}' (mismatched shapes broke compute)" for m in mats],
              errors_before=n_errors(issues), errors_after=n_errors(f_issues))
    return (fixed, f_issues, f_summary, f_out) if ok else (spec, issues, summary, default_out)


def auto_fix(spec, code, issues, summary, default_out, runner, checks, trace, rnd, only_if_soft=True):
    """Deterministic fixes. While an LLM repair is still possible, apply them only when every remaining error
    is soft (so a real bug revealed by an invariant is shown to the model instead of being hidden)."""
    spec, issues, summary, default_out = lock_sizes_fix(spec, code, issues, summary, default_out, runner, checks,
                                                        trace, rnd)
    if only_if_soft and any(i.sev == "error" and i.code not in SOFT_CODES for i in issues):
        return spec, issues, summary, default_out
    fixed, fixes = deterministic_fix(spec, issues)
    if not fixes:
        return spec, issues, summary, default_out
    f_issues, f_summary, f_out = validate(fixed, code, runner, checks)
    ok = badness(f_issues) < badness(issues)
    trace.log("revise", "deterministic_fix", "accepted" if ok else "rejected", round=rnd, fixes=fixes,
              errors_before=n_errors(issues), errors_after=n_errors(f_issues))
    return (fixed, f_issues, f_summary, f_out) if ok else (spec, issues, summary, default_out)


def _err_keys(issues):
    return sorted((i.code, i.msg[:50]) for i in issues if i.sev == "error")


_WEIGHT = {"compute": 1000, "robustness": 50, "views": 20, "readouts": 10, "structure": 5, "controls": 5, "placeholders": 4,
           "tests": 4, "explorations": 4, "grounding": 3, "latex": 2, "invariants": 1}


def badness(issues):
    """Severity-weighted error score: a page whose compute cannot run is worse than any number of small issues."""
    return sum(_WEIGHT.get(i.code, 2) for i in issues if i.sev == "error")


_PH = re.compile(r"\{\{\s*(?:(\d+)\s*:)?\s*(.+?)\s*\}\}")


def _prose_sites(spec):
    """(container, key, exploration index or None, top-level field) for every prose string holding {{...}}."""
    sites = []

    def add(obj, key, ctx, field):
        try:
            if isinstance(obj[key], str) and "{{" in obj[key]:
                sites.append((obj, key, ctx, field))
        except (KeyError, IndexError, TypeError):
            pass
    for k in ("idea", "why_it_matters"):
        add(spec, k, None, k)
    add(spec["equation"], "caption", None, "equation")
    for st in spec["steps"]:
        add(st, "title", None, "steps")
        add(st, "text", None, "steps")
    for sy in spec["symbols"]:
        add(sy, "meaning", None, "symbols")
    for v in spec["views"]:
        add(v, "caption", None, "views")
    add(spec["caveat"], "text", None, "caveat")
    for i, e in enumerate(spec["explorations"]):
        for k in ("change", "observe", "why"):
            add(e, k, i, "explorations")
    g = spec["grounding"]
    for k in ("from_paper", "simplifications"):
        for idx in range(len(g[k])):
            add(g[k], idx, None, "grounding")
    add(g, "not_claimed", None, "grounding")
    return sites


def _ph_state(spec, defaults, ctx, step):
    if ctx is None:
        return "defaults", defaults
    steps = spec["explorations"][ctx]["steps"]
    if not steps:
        return "defaults", defaults
    j = (int(step) - 1) if step else len(steps) - 1
    j = max(0, min(j, len(steps) - 1))
    return f"e{ctx}s{j}", {**defaults, **steps[j]["preset"]}


def _fmt_ph(r):
    if r.get("num") is not None:
        v = r["num"]
        if float(v).is_integer() and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.4g}"
    if r.get("arr") is not None:
        return "[" + ", ".join(_fmt_ph({"num": x}) for x in r["arr"]) + "]"
    if r.get("str") is not None:
        return r["str"]
    return None


def eval_placeholders(spec, code, runner):
    """Evaluate every {{expr}} in prose with compute() at the right state.
    Returns ({(state_key, expr): text}, [(field, error message)])."""
    sites = _prose_sites(spec)
    if not sites or not code or runner.kind is None:
        return {}, []
    defaults = {c["id"]: c["default"] for c in spec["controls"]}
    wanted = {}
    for obj, key, ctx, field in sites:
        for m in _PH.finditer(obj[key]):
            steps = [m.group(1)]
            if ctx is not None and not m.group(1):
                steps = [str(k + 1) for k in range(max(1, len(spec["explorations"][ctx]["steps"])))]
            for st_ in steps:
                skey, state = _ph_state(spec, defaults, ctx, st_)
                wanted.setdefault(skey, (state, {}))[1].setdefault(m.group(2), field)
    keys = list(wanted)
    jobs = [{"state": wanted[k][0], "exprs": list(wanted[k][1])} for k in keys]
    results, err = runner.syntax_and_run(code, jobs)
    if err:
        return {}, [(next(iter(wanted[keys[0]][1].values())), "placeholders could not be evaluated: " + err)]
    values, errors = {}, []
    for k, res in zip(keys, results):
        exprs = list(wanted[k][1])
        for n, expr in enumerate(exprs):
            field = wanted[k][1][expr]
            if not res.get("ok"):
                errors.append((field, f"{{{{{expr}}}}} in {field}: compute threw: {res.get('err')}"))
                continue
            v = (res.get("vals") or [{}])[n] if n < len(res.get("vals") or []) else {}
            txt = _fmt_ph(v) if v.get("ok") else None
            if txt is None:
                errors.append((field, f"{{{{{expr}}}}} in {field} does not evaluate to a finite number or text "
                                      f"({v.get('err') or v.get('val')}); use an expression over out such as out.H"))
            else:
                values[(k, expr)] = txt
    return values, errors


def fill_placeholders(spec, code, runner):
    """Replace {{expr}} in prose by the value compute() actually produces (never typed by the model)."""
    values, errors = eval_placeholders(spec, code, runner)
    sites = _prose_sites(spec)
    if not sites:
        return spec, 0, errors
    defaults = {c["id"]: c["default"] for c in spec["controls"]}
    n = 0
    for obj, key, ctx, field in sites:
        def sub(m):
            nonlocal n
            n += 1
            nsteps = len(spec["explorations"][ctx]["steps"]) if ctx is not None else 0
            if ctx is not None and not m.group(1) and nsteps > 1:
                # no step given in a multi-step exploration: show the value at every step, in order
                vals = [values.get((_ph_state(spec, defaults, ctx, str(k + 1))[0], m.group(2)), "—") for k in range(nsteps)]
                return " → ".join(vals)
            skey, _ = _ph_state(spec, defaults, ctx, m.group(1))
            val = values.get((skey, m.group(2)), "—")
            if ctx is not None and nsteps > 1:
                # make the binding explicit so a sentence cannot silently show another step's value
                val += f" (step {skey.split('s')[-1] and int(skey.split('s')[-1]) + 1})"
            return val
        obj[key] = _PH.sub(sub, obj[key])
    return spec, n, errors


def n_errors(issues):
    return sum(1 for i in issues if i.sev == "error")


# ----------------------------------------------------------------------------- repair

def control_count(ctrls):
    """Interactive degrees of freedom: a resizable vector/matrix counts twice (its values and its size)."""
    n = 0
    for c in ctrls:
        resizable = (c["type"] == "vector" and c.get("min_len", 1) < c.get("max_len", 1)) or \
                    (c["type"] == "matrix" and c.get("resizable"))
        n += 2 if resizable else 1
    return n


def _facts_table(spec, code, runner):
    """Computed outputs at the defaults and at every exploration step, for the prose review."""
    defaults = {c["id"]: c["default"] for c in spec["controls"]}
    states = [("defaults", defaults)]
    for i, e in enumerate(spec["explorations"]):
        for j, st in enumerate(e["steps"]):
            states.append((f"exploration {i + 1} step {j + 1}", {**defaults, **st["preset"]}))
    res, err = runner.syntax_and_run(code, [{"state": st, "exprs": []} for _, st in states])
    if err or not res:
        return None
    view_keys = {v["data"] for v in spec["views"]}
    lines = []
    for (name, st), r in zip(states, res):
        if not r.get("ok"):
            continue
        out = json.loads(r["sig"])
        facts = {}
        for k, v in out.items():
            if k in view_keys or k == "warning":
                continue
            if _is_num(v):
                facts[k] = float(f"{v:.4g}")
            elif _numlist(v) and 0 < len(v) <= 8:
                facts[k] = [float(f"{x:.4g}") if _is_num(x) else x for x in v]
            elif isinstance(v, str) and len(v) < 60:
                facts[k] = v
            if len(facts) >= 18:
                break
        changed = {k: v for k, v in st.items() if v != defaults.get(k)}
        lines.append(f"{name}: controls changed vs defaults {json.dumps(changed, ensure_ascii=False)[:300]} -> "
                     f"{json.dumps(facts, ensure_ascii=False)[:900]}")
    return "\n".join(lines)


_PROSE_KEYS = ("idea", "why_it_matters", "steps", "symbols", "caveat", "grounding", "equation", "explorations")


def annotate_placeholders(spec, values):
    """Copy of spec where each {{expr}} shows the value it will render as: {{2:out.H}}⟨=1.5⟩ (for the reviewer)."""
    ann = json.loads(json.dumps(spec))
    defaults = {c["id"]: c["default"] for c in ann["controls"]}
    for obj, key, ctx, field in _prose_sites(ann):
        def sub(m):
            nsteps = len(ann["explorations"][ctx]["steps"]) if ctx is not None else 0
            if ctx is not None and not m.group(1) and nsteps > 1:
                v = " → ".join(values.get((_ph_state(ann, defaults, ctx, str(k + 1))[0], m.group(2)), "?")
                               for k in range(nsteps))
            else:
                v = values.get((_ph_state(ann, defaults, ctx, m.group(1))[0], m.group(2)), "?")
            return m.group(0) + "⟨=" + v + "⟩"
        obj[key] = _PH.sub(sub, obj[key])
    return ann


def review_prompt(case, spec, facts, values=None):
    if values:
        spec = annotate_placeholders(spec, values)
    prose = {k: spec[k] for k in ("title", "idea", "why_it_matters", "steps", "caveat", "grounding")}
    prose["paper"] = spec["paper"]["title"]
    prose["explorations"] = [{"title": e["title"], "change": e["change"], "observe": e["observe"], "why": e["why"],
                              "steps": [{"label": st["label"], "preset": st["preset"]} for st in e["steps"]]}
                             for e in spec["explorations"]]
    return "\n".join(["BRIEF", f"focus: {case.get('focus', '')}", f"audience: {case.get('audience', '')}", "",
                      "PAGE TEXT:", json.dumps(prose, ensure_ascii=False), "",
                      "COMPUTED VALUES (from the page's own compute function):", facts, "",
                      "In PAGE TEXT, {{...}}⟨=v⟩ shows the value each placeholder renders; check that it describes the "
                      "step the sentence talks about. Write placeholders without the ⟨=v⟩ part.", "",
                      "Return the <patch> block."])


def apply_review(spec, text):
    """Take only prose from the review patch; presets, expects, controls and code are never changed here."""
    ptxt = _block(text, "patch")
    if not ptxt or ptxt.strip() in ("", "{}"):
        return spec, []
    patch = loads_lenient(re.sub(r"⟨=[^⟩]*⟩", "", ptxt))
    if not isinstance(patch, dict):
        return spec, []
    new = json.loads(json.dumps(spec))
    changed = []
    for k, v in patch.items():
        if k not in _PROSE_KEYS:
            continue
        if k == "explorations" and isinstance(v, list):
            for i, e in enumerate(v[:len(new["explorations"])]):
                if isinstance(e, dict):
                    for f in ("title", "change", "observe", "why"):
                        if isinstance(e.get(f), str) and e[f].strip():
                            new["explorations"][i][f] = e[f].strip() if f != "title" else plain_label(e[f].strip())
            changed.append(k)
        elif k in ("idea", "why_it_matters") and isinstance(v, str) and v.strip():
            new[k] = v.strip()
            changed.append(k)
        elif k in ("steps", "symbols", "caveat", "grounding", "equation"):
            tmp = normalize_spec({**new, k: v}, [])
            if (k != "symbols" or len(tmp["symbols"]) >= 2) and (k != "steps" or tmp["steps"]):
                new[k] = tmp[k]
                changed.append(k)
    return new, changed


def deterministic_fix(spec, issues):
    """Fix low-value failures without an LLM call: drop invariants that do not hold for every reachable
    input (they are optional live checks) and display-only controls that never affect the computation."""
    fixes = []
    bad_inv = sorted({i.ref for i in issues if i.code == "invariants" and i.ref is not None})
    dead = [i.ref for i in issues if i.code == "controls" and i.ref]
    bad_ro = {i.ref for i in issues if i.code == "readouts" and i.ref}
    bad_views = {i.ref for i in issues if i.code == "views" and i.ref}
    if len(spec["views"]) - len(bad_views) < 2:
        bad_views = set()
    if not bad_inv and not dead and not bad_ro and not bad_views:
        return spec, fixes
    spec = json.loads(json.dumps(spec))
    if bad_ro:
        fixes += [f"dropped readout '{k}' (not a displayable number)" for k in sorted(bad_ro)]
        spec["readouts"] = [r for r in spec["readouts"] if r["key"] not in bad_ro]
    if bad_views:
        fixes += [f"dropped view '{k}' (its data has an invalid shape)" for k in sorted(bad_views)]
        spec["views"] = [v for v in spec["views"] if v["id"] not in bad_views]
    if bad_inv:
        fixes += [f"dropped invariant '{spec['invariants'][k]['label']}' (not true for every input)" for k in bad_inv]
        spec["invariants"] = [x for k, x in enumerate(spec["invariants"]) if k not in bad_inv]
    if dead and control_count([c for c in spec["controls"] if c["id"] not in dead]) >= 2:
        fixes += [f"dropped control '{c}' (it never changes any output)" for c in dead]
        spec["controls"] = [c for c in spec["controls"] if c["id"] not in dead]
        for t in spec["tests"]:
            t["state"] = {k: v for k, v in t["state"].items() if k not in dead}
        for e in spec["explorations"]:
            for st in e["steps"]:
                st["preset"] = {k: v for k, v in st["preset"].items() if k not in dead}
    return spec, fixes


_REPAIR_KEYS = {"controls": ("controls",), "readouts": ("readouts", "views"), "views": ("views",),
                "invariants": ("invariants",), "tests": ("tests", "controls"), "explorations": ("explorations", "controls"),
                "robustness": ("controls",), "compute": ("controls", "readouts", "views"),
                "structure": ("equation", "symbols", "caveat", "explorations"), "grounding": ("grounding", "paper"),
                "latex": ("equation", "symbols"), "placeholders": ("controls",)}


def repair_prompt(case, spec, code, issues, required_checks=()):
    errs = [i for i in issues if i.sev == "error"][:12]
    keys = []
    for i in errs:
        extra = (i.ref,) if i.code == "placeholders" and i.ref else ()
        for k in _REPAIR_KEYS.get(i.code, ("controls",)) + extra:
            if k not in keys:
                keys.append(k)
    slim = {k: spec[k] for k in keys if k in spec}
    lines = ["BRIEF", f"focus: {case.get('focus', '')}", f"audience: {case.get('audience', '')}", ""]
    if required_checks:
        lines += ["CHECKS REQUIRED BY THE BRIEF (authoritative facts from the paper; if a test encoding one of these fails, "
                  "the formula/code is wrong: re-derive it from the paper):"] + [f"- {c}" for c in required_checks] + [""]
    lines += ["FAILED CHECKS:"] + [f"- [{i.code}] {i.msg}" for i in errs]
    lines += ["", "CURRENT DATA (relevant keys only):", json.dumps(slim, ensure_ascii=False),
              "", "CURRENT CODE:", code or "(missing)", "", "Return the <patch> and/or <compute> blocks."]
    return "\n".join(lines)


def apply_repair(spec, code, text, warns):
    new_spec = dict(spec)
    changed = []
    ptxt = _block(text, "patch") or _block(text, "spec")
    if ptxt and ptxt.strip() and ptxt.strip() != "{}":
        patch = loads_lenient(ptxt)
        if isinstance(patch, dict):
            for k, v in patch.items():
                if k in spec or k in ("controls", "readouts", "views", "invariants", "tests", "explorations"):
                    new_spec[k] = v
                    changed.append(k)
    new_code = extract_compute(text) if "<compute>" in text.lower() or "function compute" in text else None
    if new_code:
        changed.append("compute")
    return normalize_spec(new_spec, warns), (new_code or code), changed


# ----------------------------------------------------------------------------- rendering

_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "theta": "θ", "lambda": "λ",
          "mu": "μ", "sigma": "σ", "tau": "τ", "phi": "φ", "pi": "π", "rho": "ρ", "omega": "ω", "eta": "η",
          "Sigma": "Σ", "Delta": "Δ", "sqrt": "√", "sum": "Σ", "top": "⊤", "times": "×", "cdot": "·",
          "infty": "∞", "leq": "≤", "geq": "≥", "neq": "≠", "approx": "≈", "log": "log", "in": "∈"}


def plain_label(s):
    """Labels must be plain text; strip stray LaTeX."""
    if "$" not in s and "\\" not in s:
        return s
    s = s.replace("$", "")
    s = re.sub(r"\\(?:mathrm|text|mathbf|operatorname|mathit)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\([A-Za-z]+)", lambda m: _GREEK.get(m.group(1), m.group(1)), s)
    s = s.replace("{", "").replace("}", "")
    return s


def latex_to_mathml(tex, display=False, strict=False):
    import latex2mathml.converter as l2m
    tex = (tex or "").strip().strip("$").strip()
    tex = re.sub(r"\\(?:displaystyle|limits|nolimits)\b", "", tex)
    tex = tex.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\operatorname*", "\\operatorname")
    tex = re.sub(r"\\(?:label|tag)\{[^}]*\}", "", tex)
    tex = tex.replace("\\\\[", "\\\\").replace("&=", "=") if "\\begin" not in tex else tex
    try:
        return l2m.convert(tex, display="block" if display else "inline")
    except Exception:
        if strict:
            raise
        return f'<code class="tex">{html.escape(tex)}</code>'


_BULLET = re.compile(r"^\s*[-*•]\s+")
_MATH_RE = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\\((.+?)\\\)|\$(?!\s)([^$\n]+?)(?<!\s)\$", re.S)


def _inline_fmt(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    return s


def rich(text):
    """Prose -> safe HTML: escapes text, converts $math$ to MathML, **bold**, paragraphs and simple lists."""
    if not text:
        return ""
    paras = re.split(r"\n\s*\n", text.strip())
    out = []
    for p in paras:
        lines = p.split("\n")
        if all(_BULLET.match(ln) for ln in lines if ln.strip()) and len(lines) > 1:
            items = [_rich_inline(_BULLET.sub("", ln)) for ln in lines if ln.strip()]
            out.append("<ul>" + "".join("<li>" + it + "</li>" for it in items) + "</ul>")
        else:
            out.append("<p>" + "<br>".join(_rich_inline(ln) for ln in lines) + "</p>")
    return "".join(out)


def _rich_inline(s):
    parts, pos = [], 0
    for m in _MATH_RE.finditer(s):
        parts.append(_inline_fmt(s[pos:m.start()]))
        disp = m.group(1) is not None or m.group(2) is not None
        tex = next(g for g in m.groups() if g is not None)
        parts.append(latex_to_mathml(tex, display=disp))
        pos = m.end()
    parts.append(_inline_fmt(s[pos:]))
    return "".join(parts)


def rich_inline(text):
    return _rich_inline(text) if text else ""


def build_page(case, model, spec, code, report):
    paper = ground_locations(spec["paper"], _s(case.get("focus")))
    return {
        "meta": {"title": spec["title"], "subtitle": plain_label(spec.get("subtitle", "")),
                 "audience": _s(case.get("audience")), "source_url": _s(case.get("source_url")), "model": model,
                 "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        "paper": paper,
        "idea_html": rich(spec["idea"]),
        "why_html": rich(spec["why_it_matters"]),
        "equation": {"mathml_html": latex_to_mathml(spec["equation"]["latex"], display=True),
                     "caption_html": rich_inline(spec["equation"]["caption"])},
        "symbols": [{"symbol_html": latex_to_mathml(s["latex"]), "meaning_html": rich_inline(s["meaning"]),
                     "where_html": rich_inline(s["where"])} for s in spec["symbols"]],
        "steps": [{"title_html": rich_inline(s["title"]), "text_html": rich(s["text"])} for s in spec["steps"]],
        "controls": spec["controls"],
        "readouts": spec["readouts"],
        "views": [{"id": v["id"], "type": v["type"], "title": v["title"], "caption_html": rich_inline(v["caption"]),
                   "data": v["data"]} for v in spec["views"] if v["type"] in VIEW_TYPES],
        "invariants": spec["invariants"],
        "tests": spec["tests"],
        "explorations": [{"title": e["title"], "change_html": rich_inline(e["change"]),
                          "observe_html": rich_inline(e["observe"]), "why_html": rich_inline(e["why"]),
                          "steps": e["steps"],
                          "preset": e["steps"][0]["preset"] if e["steps"] else {},
                          "expect": e["steps"][0]["expect"] if e["steps"] else ""} for e in spec["explorations"]],
        "caveat": {"kind": spec["caveat"]["kind"], "title": spec["caveat"]["title"],
                   "text_html": rich(spec["caveat"]["text"])},
        "grounding": {"from_paper_html": [rich_inline(x) for x in spec["grounding"]["from_paper"]],
                      "simplifications_html": [rich_inline(x) for x in spec["grounding"]["simplifications"]],
                      "not_claimed_html": rich_inline(spec["grounding"]["not_claimed"] or
                                                      "This is a small toy demonstration of the mechanism; it does not "
                                                      "reproduce the paper's experiments or results.")},
        "report": report,
    }


def _script_safe(s):
    return re.sub(r"</(script)", r"<\\/\1", s, flags=re.I).replace("<!--", "<\\!--")


_MINI_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>/*__TITLE__*/</title></head><body>
<script id="page-data" type="application/json">/*__PAGE_JSON__*/</script><script>/*__COMPUTE_JS__*/</script>
<pre id="o"></pre><script>document.getElementById('o').textContent=document.getElementById('page-data').textContent</script>
</body></html>"""


def render_html(page, code):
    tpl = TEMPLATE.read_text(encoding="utf-8") if TEMPLATE.exists() else _MINI_TEMPLATE
    page_json = json.dumps(page, ensure_ascii=False).replace("</", "<\\/").replace("<!--", "<\\u0021--")
    out = tpl.replace("/*__TITLE__*/", html.escape(page["meta"]["title"] or "Interactive explanation"))
    out = out.replace("/*__PAGE_JSON__*/", page_json)
    out = out.replace("/*__COMPUTE_JS__*/", _script_safe(code or ""))
    return out


def static_html_checks(doc):
    problems = []
    for pat, what in ((r"<script[^>]+src\s*=", "external script"), (r"<link[^>]+href\s*=\s*[\"']https?:", "external stylesheet"),
                      (r"@import", "CSS @import"), (r"url\(\s*[\"']?https?:", "remote CSS url()"),
                      (r"<img[^>]+src\s*=\s*[\"']https?:", "remote image"), (r"<iframe", "iframe")):
        if re.search(pat, doc, re.I):
            problems.append(what)
    if "/*__PAGE_JSON__*/" in doc or "/*__COMPUTE_JS__*/" in doc:
        problems.append("unfilled template placeholder")
    return problems


# ----------------------------------------------------------------------------- fallback page

def fallback_spec(case, reason):
    focus = _s(case.get("focus"))
    return {
        "title": "Explanation could not be generated", "subtitle": reason[:200], "idea": focus,
        "why_it_matters": "", "plan": {}, "paper": {"title": "", "authors": "", "year": "", "section": "", "equation": ""},
        "equation": {"latex": "", "caption": ""}, "symbols": [], "steps": [], "controls": [], "readouts": [],
        "views": [], "invariants": [], "tests": [], "explorations": [],
        "caveat": {"kind": "limitation", "title": "Generation failed", "text": reason},
        "grounding": {"from_paper": [], "simplifications": [], "not_claimed": ""},
    }


# ----------------------------------------------------------------------------- main

_RUN = {}


def _debug_write(path, text):
    try:
        Path(path).write_text(text, encoding="utf-8")
    except OSError:
        pass


def main():
    """Safety net: any unexpected error still leaves a trace event and a usable page."""
    try:
        return _main()
    except Exception as e:  # noqa: BLE001
        trace, outdir = _RUN.get("trace"), _RUN.get("outdir")
        msg = f"{type(e).__name__}: {e}"[:400]
        try:
            if trace:
                trace.log("finish", "unexpected_error", "failed", error=msg, elapsed_s=round(elapsed(), 2))
            if outdir and not (outdir / "index.html").exists():
                case = _RUN.get("case") or {}
                page = build_page(case, _RUN.get("model", ""), normalize_spec(fallback_spec(case, msg), []), "",
                                  {"checks": [], "revisions": 0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                   "elapsed_s": round(elapsed(), 1)})
                (outdir / "index.html").write_text(render_html(page, ""), encoding="utf-8")
        except Exception:
            pass
        print(f"error: {msg}", file=sys.stderr)
        return 1


def _main():
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Paper to Playground generator")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    trace = Trace(outdir / "trace.jsonl")
    _RUN.update(trace=trace, outdir=outdir, model=args.model)
    debug = os.environ.get("P2P_DEBUG") == "1"

    try:
        case = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
        if not isinstance(case, dict):
            raise ValueError("case.json must contain a JSON object")
    except Exception as e:
        trace.log("input", "load_case", "error", error=str(e)[:300])
        print(f"error: cannot read input: {e}", file=sys.stderr)
        return 2
    _RUN["case"] = case
    missing = [k for k in ("source_url", "focus", "audience") if not _s(case.get(k))]
    if not _s(case.get("audience")):
        case["audience"] = "Engineering undergraduates (assumed: no audience was given)"
    trace.log("input", "load_case", "ok" if not missing else "warn", fields=sorted(case.keys()), missing=missing,
              model=args.model)
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for ln in env.read_text(encoding="utf-8").splitlines():
                if ln.startswith("OPENROUTER_API_KEY="):
                    key = ln.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        trace.log("input", "api_key", "error", error="OPENROUTER_API_KEY not set")
        print("error: OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    runner = JSRunner()
    llm = LLM(args.model, key, trace)
    reqs, checks = extract_requirements(_s(case.get("focus")))
    trace.log("understand", "extract_requirements", "ok", requirements=len(reqs), required_checks=checks,
              js_engine=runner.kind)

    warns = []
    user = build_user_prompt(case, reqs, checks)

    def generate(tag, attempts):
        """One generation call (plus a retry on unparseable output). Returns (spec, code) or (None, None)."""
        for attempt in range(attempts):
            try:
                text, finish = llm.chat("generate", SYSTEM, user if attempt == 0 else
                                        user + "\n\nReturn ONLY the <spec> JSON block and the <compute> block. Keep prose short.",
                                        GEN_MAX_TOKENS, attempts=3)
            except (RuntimeError, BudgetExceeded) as e:
                trace.log("generate", "llm_generate", "error", candidate=tag, error=str(e)[:300])
                return None, None
            if debug:
                _debug_write(outdir / f"debug_generate_{tag}_{attempt}.txt", text)
            try:
                raw_spec, code_ = parse_generation(text)
                spec_ = normalize_spec(raw_spec, warns)
                trace.log("generate", "parse_output", "ok", candidate=tag, finish_reason=finish,
                          compute_chars=len(code_ or ""), controls=[c["id"] + ":" + c["type"] for c in spec_["controls"]],
                          views=[v["type"] for v in spec_["views"]], tests=len(spec_["tests"]),
                          explorations=len(spec_["explorations"]), normalization_warnings=warns[:10])
                trace.log("plan", "llm_plan", "ok", candidate=tag, plan=spec_.get("plan"))
                return spec_, code_
            except Exception as e:
                trace.log("generate", "parse_output", "error", candidate=tag, finish_reason=finish,
                          error=f"{type(e).__name__}: {e}"[:300])
        return None, None

    def check_and_repair(spec, code, tag, max_repairs, hint=False):
        """Deterministic checks, then targeted LLM repairs while errors keep decreasing."""
        issues, summary, default_out = validate(spec, code, runner, checks)
        trace.log("check", "validate", "pass" if n_errors(issues) == 0 else "fail", candidate=tag, round=0,
                  errors=n_errors(issues), issues=[i.as_dict() for i in issues][:20], checks=summary)
        spec, issues, summary, default_out = auto_fix(spec, code, issues, summary, default_out, runner, checks, trace, 0)
        revisions, retried_hard = 0, False
        for rnd in range(1, max_repairs + 2):
            if n_errors(issues) == 0 or (rnd > max_repairs and not retried_hard):
                break
            try:
                hard = any(i.code in ("tests", "explorations", "robustness", "compute") for i in issues if i.sev == "error")
                prompt = repair_prompt(case, spec, code, issues, checks)
                if retried_hard or (hint and hard):
                    prompt += ("\nA previous fix attempt did not change these failures. Before the code, re-derive the "
                               "paper's formula step by step as comments at the top of compute, then implement exactly that.")
                text, finish = llm.chat("revise", REPAIR_SYSTEM, prompt, REPAIR_MAX_TOKENS + (2000 if hard else 0))
            except (RuntimeError, BudgetExceeded) as e:
                trace.log("revise", "llm_repair", "skipped", candidate=tag, round=rnd, reason=str(e)[:300])
                break
            if debug:
                _debug_write(outdir / f"debug_repair_{tag}_{rnd}.txt", text)
            try:
                new_spec, new_code, changed = apply_repair(spec, code, text, warns)
            except Exception as e:
                trace.log("revise", "apply_patch", "error", candidate=tag, round=rnd, error=f"{type(e).__name__}: {e}"[:300])
                continue
            new_issues, new_summary, new_out = validate(new_spec, new_code, runner, checks)
            new_spec, new_issues, new_summary, new_out = auto_fix(new_spec, new_code, new_issues, new_summary, new_out,
                                                                  runner, checks, trace, rnd)
            accepted = badness(new_issues) <= badness(issues)
            stuck = _err_keys(new_issues) == _err_keys(issues)
            trace.log("revise", "apply_patch", "accepted" if accepted else "rejected", candidate=tag, round=rnd,
                      changed=changed, errors_before=n_errors(issues), errors_after=n_errors(new_issues),
                      issues=[i.as_dict() for i in new_issues][:20], checks=new_summary)
            if accepted:
                revisions += 1
                spec, code, issues, summary, default_out = new_spec, new_code, new_issues, new_summary, new_out
            if stuck:
                hard_left = any(i.code in ("tests", "explorations") for i in issues if i.sev == "error")
                if hard_left and not retried_hard and max_repairs > 1:
                    retried_hard = True  # one more attempt, asking for an explicit derivation first
                    trace.log("revise", "retry_with_derivation", "ok", candidate=tag, round=rnd)
                    continue
                trace.log("revise", "stop_repairs", "ok", candidate=tag, round=rnd,
                          reason="same errors persisted after a repair")
                break
        return spec, code, issues, summary, default_out, revisions

    spec, code = generate("A", 2)
    if spec is None:
        reason = "The language model did not return a usable explanation within the limits."
        page = build_page(case, args.model, normalize_spec(fallback_spec(case, reason), []), "", {
            "checks": [], "revisions": 0, "calls": llm.requests, "prompt_tokens": llm.prompt_tokens,
            "completion_tokens": llm.completion_tokens, "elapsed_s": round(elapsed(), 1)})
        (outdir / "index.html").write_text(render_html(page, ""), encoding="utf-8")
        trace.log("finish", "write_output", "failed", reason=reason, requests=llm.requests,
                  prompt_tokens=llm.prompt_tokens, completion_tokens=llm.completion_tokens,
                  total_tokens=llm.prompt_tokens + llm.completion_tokens, elapsed_s=round(elapsed(), 2))
        trace.close()
        return 1
    spec, code, issues, summary, default_out, revisions = check_and_repair(spec, code, "A", 1)

    # ---- if the computation still contradicts a required check or a claimed observation, try one fresh draft
    def correctness_errors(iss):
        return [i for i in iss if i.sev == "error" and i.code in ("tests", "explorations", "compute", "robustness")]

    if correctness_errors(issues) and llm.requests <= 3 and elapsed() < 300:
        trace.log("revise", "regenerate", "start", reason="correctness checks still failing after a repair",
                  failing=[i.msg[:120] for i in correctness_errors(issues)][:5])
        spec_b, code_b = generate("B", 1)
        if spec_b is not None:
            res_b = check_and_repair(spec_b, code_b, "B", 1)
            better = badness(res_b[2]) < badness(issues)
            trace.log("revise", "regenerate", "kept_B" if better else "kept_A",
                      badness_A=badness(issues), badness_B=badness(res_b[2]))
            if better:
                spec, code, issues, summary, default_out, rev_b = res_b
                revisions += rev_b + 1
    # a last repair only for failure kinds repairs reliably fix (measured: correctness failures that survived
    # a repair and a fresh draft were fixed by a further repair in only 2 of 19 runs)
    fixable = [i for i in issues if i.sev == "error" and i.code not in ("tests", "explorations", "invariants")]
    if fixable and llm.requests <= 4 and elapsed() < 360:
        spec, code, issues, summary, default_out, rev_c = check_and_repair(spec, code, "final", 1, hint=True)
        revisions += rev_c

    # ---- review: one prose pass against the page's own computed numbers (prose only; re-validated)
    if code and runner.kind and llm.requests <= 5 and elapsed() < 420 and badness(issues) < 1000:
        facts = _facts_table(spec, code, runner)
        if facts:
            try:
                ph_values, _ = eval_placeholders(spec, code, runner)
                text, _ = llm.chat("review", REVIEW_SYSTEM, review_prompt(case, spec, facts, ph_values), 5000)
                if debug:
                    _debug_write(outdir / "debug_review.txt", text)
                new_spec, changed = apply_review(spec, text)
                if changed:
                    r_issues, r_summary, r_out = validate(new_spec, code, runner, checks)
                    ok = badness(r_issues) <= badness(issues)
                    trace.log("review", "apply_prose_review", "accepted" if ok else "rejected", changed=changed,
                              errors_before=n_errors(issues), errors_after=n_errors(r_issues))
                    if ok:
                        spec, issues, summary, default_out = new_spec, r_issues, r_summary, r_out
                        revisions += 1
                else:
                    trace.log("review", "apply_prose_review", "no_change")
            except (RuntimeError, BudgetExceeded) as e:
                trace.log("review", "llm_review", "skipped", reason=str(e)[:300])
            except Exception as e:  # a malformed review never blocks the page
                trace.log("review", "apply_prose_review", "error", error=f"{type(e).__name__}: {e}"[:300])

    # ---- finalize: never show a learner a check that we could not make pass
    spec, issues, summary, default_out = auto_fix(spec, code, issues, summary, default_out, runner, checks, trace,
                                                  "final", only_if_soft=False)
    removed = []
    if runner.kind and code:
        bad_tests = {i.msg.split("'")[1] for i in issues if i.code == "tests" and i.msg.startswith("test '")}
        if bad_tests:
            removed += [f"test: {t}" for t in bad_tests]
            spec["tests"] = [t for t in spec["tests"] if t["label"] not in bad_tests]
        for i in issues:
            if i.code == "explorations" and "expect" in i.msg:
                m = re.match(r"exploration (\d+) step (\d+)", i.msg)
                if m:
                    k, j = int(m.group(1)) - 1, int(m.group(2)) - 1
                    if 0 <= k < len(spec["explorations"]) and 0 <= j < len(spec["explorations"][k]["steps"]):
                        removed.append(f"exploration {k + 1} step {j + 1} check")
                        spec["explorations"][k]["steps"][j]["expect"] = ""
            if i.code == "invariants":
                m = re.match(r"invariant '([^']*)'", i.msg)
                if m:
                    spec["invariants"] = [x for x in spec["invariants"] if x["label"] != m.group(1)]
                    removed.append(f"invariant: {m.group(1)}")
    if removed:
        trace.log("finalize", "remove_failing_checks", "ok", removed=removed)

    spec, n_ph, ph_err = fill_placeholders(spec, code, runner)
    if n_ph:
        trace.log("render", "fill_computed_numbers", "ok" if not ph_err else "warn", filled=n_ph,
                  errors=[m for _, m in ph_err][:5])
    report = {"checks": summary, "revisions": revisions, "calls": llm.requests,
              "prompt_tokens": llm.prompt_tokens, "completion_tokens": llm.completion_tokens,
              "elapsed_s": round(elapsed(), 1), "unresolved": [i.msg[:160] for i in issues if i.sev == "error"][:6],
              "removed_checks": removed}
    if debug:
        _debug_write(outdir / "debug_final_spec.json", json.dumps(spec, ensure_ascii=False, indent=1))
        _debug_write(outdir / "debug_final_compute.js", code or "")
        _debug_write(outdir / "debug_report.json", json.dumps(report, ensure_ascii=False))
    page = build_page(case, args.model, spec, code, report)
    doc = render_html(page, code)
    problems = static_html_checks(doc)
    (outdir / "index.html").write_text(doc, encoding="utf-8")
    trace.log("render", "write_index_html", "ok" if not problems else "warn", bytes=len(doc.encode("utf-8")),
              offline_problems=problems)
    status = "success" if n_errors(issues) == 0 else "success_with_unresolved_issues"
    trace.log("finish", "summary", status, requests=llm.requests, prompt_tokens=llm.prompt_tokens,
              completion_tokens=llm.completion_tokens, reasoning_tokens=llm.reasoning_tokens,
              total_tokens=llm.prompt_tokens + llm.completion_tokens, revisions=revisions,
              unresolved_errors=n_errors(issues), elapsed_s=round(elapsed(), 2))
    trace.close()
    print(f"wrote {outdir / 'index.html'} ({status}; {llm.requests} calls, "
          f"{llm.prompt_tokens + llm.completion_tokens} tokens, {elapsed():.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
