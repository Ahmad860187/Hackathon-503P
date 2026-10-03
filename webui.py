#!/usr/bin/env python3
"""Optional local web interface for Paper to Playground.

Fill in the three inputs in a browser, watch the generator work, and open the finished page.
It runs agent.py unchanged in a subprocess, so the graded command-line interface is not affected.

Usage: python webui.py [--port 8800] [--no-browser]
Needs OPENROUTER_API_KEY in the environment (or a .env file), or the key can be typed into the form;
the key is only passed to the generator process and is never written to disk.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "ui_runs"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
JOBS = {}
LOCK = threading.Lock()
ID_RE = re.compile(r"^[a-f0-9]{12}$")

STEPS = ["Read the brief", "Draft the explanation", "Run the checks", "Fix problems", "Review the wording",
         "Build the page"]
STAGE_STEP = {"input": 0, "understand": 0, "generate": 1, "plan": 1, "check": 2, "revise": 3, "review": 4,
              "finalize": 5, "render": 5, "finish": 5}


def env_key():
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    env = ROOT / ".env"
    if not key and env.exists():
        for ln in env.read_text(encoding="utf-8").splitlines():
            if ln.startswith("OPENROUTER_API_KEY="):
                key = ln.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def read_trace(path):
    events = []
    try:
        for ln in path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(ln))
            except ValueError:
                pass
    except OSError:
        pass
    return events


def describe(ev):
    """One plain-language line per notable trace event (None for events not worth showing)."""
    st, ac, res = ev.get("stage"), ev.get("action"), ev.get("result")
    if ac == "llm_call":
        if res == "ok":
            what = {"generate": "Drafted the page", "revise": "Asked the model for a fix",
                    "review": "Reviewed the wording"}.get(st, "Model call")
            return f"{what} ({ev.get('completion_tokens', 0):,} tokens written, {ev.get('elapsed_s', 0):.0f} s)"
        return "A model call failed and was retried"
    if ac == "validate":
        n = ev.get("errors", 0)
        return "All checks passed" if n == 0 else f"Checks found {n} problem{'s' if n != 1 else ''}"
    if ac == "deterministic_fix" and res == "accepted":
        return "Fixed small problems automatically"
    if ac == "apply_patch":
        if res == "accepted":
            return f"Applied a fix (problems {ev.get('errors_before')} → {ev.get('errors_after')})"
        return "Rejected a fix that made things worse"
    if ac == "regenerate" and res == "start":
        return "Starting a fresh draft"
    if ac == "regenerate":
        return "Kept the better of the two drafts"
    if ac == "apply_prose_review":
        return {"accepted": "Corrected the wording against the computed numbers",
                "no_change": "Wording already matched the computed numbers"}.get(res)
    if ac == "remove_failing_checks":
        return "Removed checks that could not be made to pass"
    if ac == "write_index_html":
        return "Wrote the page"
    return None


class Job:
    def __init__(self, inputs, model, key):
        self.id = uuid.uuid4().hex[:12]
        self.dir = RUNS / self.id
        self.out = self.dir / "out"
        self.inputs, self.model = inputs, model
        self.started = time.time()
        self.ended = None
        self.state = "running"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "case.json").write_text(json.dumps(inputs, ensure_ascii=False, indent=1), encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        env.pop("P2P_DEBUG", None)
        if key:
            env["OPENROUTER_API_KEY"] = key
        self.log = open(self.dir / "log.txt", "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, str(ROOT / "agent.py"), "--input", str(self.dir / "case.json"), "--output", str(self.out),
             "--model", model], cwd=str(ROOT), env=env, stdout=self.log, stderr=subprocess.STDOUT)
        threading.Thread(target=self._wait, daemon=True).start()

    def _wait(self):
        code = self.proc.wait()
        self.log.close()
        self.ended = time.time()
        if self.state != "cancelled":
            self.state = "done" if (self.out / "index.html").exists() and code == 0 else "failed"
        self._save_meta()

    def cancel(self):
        if self.state == "running":
            self.state = "cancelled"
            self.proc.terminate()

    def _save_meta(self):
        meta = {"id": self.id, "inputs": self.inputs, "model": self.model, "state": self.state,
                "started": self.started, "title": page_title(self.out / "index.html"), "stats": finish_stats(self.out)}
        (self.dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def status(self):
        events = read_trace(self.out / "trace.jsonl")
        step, seen = 0, set()
        for ev in events:
            k = STAGE_STEP.get(ev.get("stage"))
            if k is not None:
                step = max(step, k)
                seen.add(k)
        lines = [d for d in (describe(e) for e in events) if d]
        data = {"id": self.id, "state": self.state, "step": step, "seen": sorted(seen), "steps": STEPS, "log": lines[-12:],
                "elapsed": round((self.ended or time.time()) - self.started, 1)}
        if self.state != "running":
            data["stats"] = finish_stats(self.out)
            data["title"] = page_title(self.out / "index.html")
            if self.state == "failed":
                data["error"] = failure_reason(self.dir, events)
        return data


def page_title(path):
    try:
        m = re.search(r"<title>(.*?)</title>", path.read_text(encoding="utf-8")[:20000], re.S)
        if m:
            import html as _h
            return _h.unescape(m.group(1)).strip()[:160]
    except OSError:
        pass
    return ""


def finish_stats(out):
    events = read_trace(out / "trace.jsonl")
    fin = next((e for e in reversed(events) if e.get("stage") == "finish"), None)
    checks = next((e.get("checks") for e in reversed(events) if e.get("checks")), None) or []
    if not fin:
        return None
    return {"calls": fin.get("requests"), "tokens": fin.get("total_tokens"), "seconds": fin.get("elapsed_s"),
            "unresolved": fin.get("unresolved_errors"), "result": fin.get("result"),
            "checks_passed": sum(1 for c in checks if c.get("ok")), "checks_total": len(checks)}


def failure_reason(jobdir, events):
    for e in reversed(events):
        if e.get("error"):
            return str(e["error"])[:300]
    try:
        tail = (jobdir / "log.txt").read_text(encoding="utf-8").strip().splitlines()[-3:]
        if tail:
            return " ".join(tail)[:300]
    except OSError:
        pass
    return "The generator stopped without writing a page."


def history():
    items = []
    if RUNS.exists():
        for d in RUNS.iterdir():
            if not (d.is_dir() and ID_RE.match(d.name)):
                continue
            with LOCK:
                job = JOBS.get(d.name)
            if job and job.state == "running":
                items.append({"id": d.name, "state": "running", "title": job.inputs.get("focus", "")[:80],
                              "started": job.started})
                continue
            try:
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                items.append({"id": d.name, "state": meta.get("state"), "started": meta.get("started", 0),
                              "title": meta.get("title") or meta.get("inputs", {}).get("focus", "")[:80],
                              "stats": meta.get("stats")})
            except (OSError, ValueError):
                pass
    items.sort(key=lambda x: x.get("started", 0), reverse=True)
    return items[:20]


def examples():
    out = []
    for p in sorted((ROOT / "examples").glob("case_*.json")):
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            out.append({"name": p.stem.replace("case_", "").replace("_", " ").title(), **c})
        except (OSError, ValueError):
            pass
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        data = body if isinstance(body, bytes) else (json.dumps(body) if not isinstance(body, str) else body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if path == "/api/config":
            return self._send(200, {"has_key": bool(env_key()), "model": DEFAULT_MODEL, "examples": examples(),
                                    "history": history()})
        m = re.match(r"^/api/status/([a-f0-9]{12})$", path)
        if m:
            with LOCK:
                job = JOBS.get(m.group(1))
            if job:
                return self._send(200, job.status())
            meta = RUNS / m.group(1) / "meta.json"
            if meta.exists():
                mj = json.loads(meta.read_text(encoding="utf-8"))
                seen = sorted({STAGE_STEP[e["stage"]] for e in read_trace(RUNS / mj["id"] / "out" / "trace.jsonl")
                               if e.get("stage") in STAGE_STEP})
                return self._send(200, {"id": mj["id"], "state": mj["state"], "step": len(STEPS) - 1, "steps": STEPS,
                                        "seen": seen,
                                        "log": [], "stats": mj.get("stats"), "title": mj.get("title"),
                                        "inputs": mj.get("inputs")})
            return self._send(404, {"error": "No page with that id."})
        m = re.match(r"^/runs/([a-f0-9]{12})/(index\.html|trace\.jsonl)$", path)
        if m:
            f = RUNS / m.group(1) / "out" / m.group(2)
            if f.exists():
                ctype = "text/html; charset=utf-8" if f.suffix == ".html" else "text/plain; charset=utf-8"
                extra = {}
                if self.path.endswith("?download=1"):
                    extra["Content-Disposition"] = 'attachment; filename="playground.html"'
                return self._send(200, f.read_bytes(), ctype, extra)
        return self._send(404, {"error": "Not found."})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(min(n, 100_000)) or b"{}")
        except ValueError:
            return self._send(400, {"error": "The request was not valid JSON."})
        if path == "/api/generate":
            inputs = {k: str(body.get(k, "")).strip()[:4000] for k in ("source_url", "focus", "audience")}
            missing = [k for k, v in inputs.items() if not v]
            if missing:
                labels = {"source_url": "paper link", "focus": "what to explain", "audience": "who it is for"}
                names = [labels[k] for k in missing]
                text = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
                return self._send(400, {"error": "Fill in the " + text + "."})
            key = str(body.get("api_key") or "").strip() or env_key()
            if not key:
                return self._send(400, {"error": "Add an OpenRouter API key to generate a page."})
            model = str(body.get("model") or DEFAULT_MODEL).strip()[:120] or DEFAULT_MODEL
            job = Job(inputs, model, key)
            with LOCK:
                JOBS[job.id] = job
            return self._send(200, {"id": job.id})
        m = re.match(r"^/api/cancel/([a-f0-9]{12})$", path)
        if m:
            with LOCK:
                job = JOBS.get(m.group(1))
            if job:
                job.cancel()
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "No running page with that id."})
        return self._send(404, {"error": "Not found."})


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper to Playground</title>
<style>
:root{
  --paper:#f7f9fb; --grid:#e4eaf1; --ink:#1b2b44; --ink2:#4a5b74; --muted:#7a889c;
  --pen:#2448b8; --pen-dark:#1a3591; --hl:#f6d55c; --ok:#1f7a4d; --bad:#b42318; --line:#d5dde8; --card:#ffffff;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0}
body{font-family:var(--sans);color:var(--ink);font-size:16px;line-height:1.5;
  background-color:var(--paper);
  background-image:linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px);
  background-size:24px 24px;min-height:100vh}
a{color:var(--pen)}
.wrap{max-width:1320px;margin:0 auto;padding:28px 24px 48px}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:22px}
h1{font-family:var(--serif);font-weight:600;font-size:2.15rem;line-height:1.1;margin:0;letter-spacing:-.01em}
.tagline{margin:6px 0 0;color:var(--ink2);font-size:1.02rem;max-width:60ch}
.keystate{font-size:.88rem;color:var(--ink2);background:var(--card);border:1px solid var(--line);border-radius:999px;padding:5px 12px}
.keystate b{color:var(--ok)} .keystate.missing b{color:var(--bad)}
.grid{display:grid;grid-template-columns:minmax(320px,410px) minmax(0,1fr);gap:22px;align-items:start}
.grid>*{min-width:0}
.sheet{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:22px;box-shadow:0 1px 0 rgba(27,43,68,.04)}
.sheet h2{font-family:var(--serif);font-weight:600;font-size:1.2rem;margin:0 0 4px}
.hint{color:var(--muted);font-size:.88rem;margin:0 0 16px}
label{display:block;font-weight:600;font-size:.95rem;margin:16px 0 6px}
label .opt{font-weight:400;color:var(--muted)}
input[type=text],input[type=url],input[type=password],textarea{width:100%;font:inherit;color:var(--ink);background:#fbfcfe;
  border:1px solid #c5cfdc;border-radius:7px;padding:10px 12px;transition:border-color .15s, box-shadow .15s}
textarea{min-height:150px;resize:vertical;line-height:1.45}
input:focus,textarea:focus{outline:none;border-color:var(--pen);box-shadow:0 0 0 3px rgba(36,72,184,.16)}
.help{color:var(--muted);font-size:.84rem;margin-top:5px}
.examples{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px;align-items:center;font-size:.9rem;color:var(--ink2)}
.chip{font:inherit;font-size:.88rem;border:1px solid #c5cfdc;background:#fff;color:var(--ink);border-radius:999px;padding:5px 12px;cursor:pointer}
.chip:hover{border-color:var(--pen);color:var(--pen)}
details.adv{margin-top:16px;font-size:.92rem}
details.adv summary{cursor:pointer;color:var(--ink2)}
.actions{display:flex;gap:10px;margin-top:20px;align-items:center}
.btn{font:inherit;font-weight:600;border-radius:8px;padding:11px 18px;cursor:pointer;border:1px solid transparent}
.btn.primary{background:var(--pen);color:#fff;flex:1}
.btn.primary:hover{background:var(--pen-dark)}
.btn.primary:disabled{background:#8fa2d6;cursor:not-allowed}
.btn.quiet{background:#fff;border-color:#c5cfdc;color:var(--ink)}
.btn.quiet:hover{border-color:var(--ink2)}
:focus-visible{outline:3px solid rgba(36,72,184,.45);outline-offset:2px}
.err{margin-top:12px;color:var(--bad);font-size:.92rem;min-height:1.2em}
.recent{margin-top:22px}
.recent h3{font-family:var(--serif);font-weight:600;font-size:1.02rem;margin:0 0 8px}
.recent ul{list-style:none;margin:0;padding:0;border-top:1px solid var(--line)}
.recent li{border-bottom:1px solid var(--line)}
.recent button{all:unset;display:block;width:100%;box-sizing:border-box;padding:9px 2px;cursor:pointer;font-size:.92rem}
.recent button:hover .rt{color:var(--pen)}
.recent button:focus-visible{outline:3px solid rgba(36,72,184,.45)}
.rt{display:block;color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rm{display:block;color:var(--muted);font-size:.8rem}
.stage{min-height:620px;display:flex;flex-direction:column}
.empty{margin:auto;max-width:520px;text-align:center;padding:40px 10px}
.empty h2{font-family:var(--serif);font-weight:600;font-size:1.45rem;margin:0 0 10px}
.empty p{color:var(--ink2);margin:0 0 8px}
.empty ol{text-align:left;color:var(--ink2);display:inline-block;margin:12px 0 0;padding-left:1.2em}
.run-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.run-head h2{font-family:var(--serif);font-weight:600;font-size:1.35rem;margin:0;line-height:1.25}
.clock{color:var(--ink2);font-variant-numeric:tabular-nums}
ol.steps{list-style:none;margin:0 0 14px;padding:0;display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:6px;counter-reset:s}
ol.steps li{counter-increment:s;font-size:.86rem;color:var(--muted);padding:8px 8px 8px 30px;position:relative;border-radius:6px;background:#f2f5f9;line-height:1.25}
ol.steps li::before{content:counter(s);position:absolute;left:9px;top:8px;font-weight:700}
ol.steps li.done{color:var(--ink2);background:#eaf2ec}
ol.steps li.done::before{content:"✓";color:var(--ok)}
ol.steps li.now{color:var(--ink);background:var(--hl);font-weight:600}
ol.steps li.skip{color:var(--muted);background:transparent;border:1px dashed var(--line)}
ol.steps li.skip::before{content:"–"}
.log{font-size:.9rem;color:var(--ink2);margin:0 0 14px;padding-left:1.1em;max-height:150px;overflow:auto}
.log li{margin:2px 0}
.stats{display:flex;flex-wrap:wrap;gap:8px 22px;margin:0 0 14px;color:var(--ink2);font-size:.92rem}
.stats b{color:var(--ink);font-variant-numeric:tabular-nums}
.result-actions{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.result-actions a{text-decoration:none}
.verdict{font-weight:600}
.verdict.ok{color:var(--ok)} .verdict.warn{color:#9a6700} .verdict.bad{color:var(--bad)}
iframe{width:100%;flex:1;min-height:640px;border:1px solid var(--line);border-radius:8px;background:#fff}
.failbox{border:1px solid #f1c4bf;background:#fdf1f0;color:var(--bad);border-radius:8px;padding:12px 14px;margin-bottom:12px}
@media (max-width:900px){.grid{grid-template-columns:1fr}.stage{min-height:0}ol.steps{grid-template-columns:repeat(3,minmax(0,1fr))}iframe{min-height:520px}}
@media (max-width:520px){.wrap{padding:18px 14px 36px}h1{font-size:1.7rem}ol.steps{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Paper to Playground</h1>
      <p class="tagline">Turn one idea from a research paper into an interactive page that a student can explore.</p>
    </div>
    <span class="keystate" id="keystate">Checking the API key…</span>
  </header>
  <div class="grid">
    <section class="sheet" aria-labelledby="form-title">
      <h2 id="form-title">Describe the explanation</h2>
      <p class="hint">Three short inputs. Generation takes about half a minute.</p>
      <form id="form" novalidate>
        <label for="source_url">Paper link</label>
        <input type="url" id="source_url" name="source_url" placeholder="https://arxiv.org/abs/1706.03762" autocomplete="off">
        <label for="focus">What to explain</label>
        <textarea id="focus" name="focus" placeholder="Name the paper and section, the idea to explain, what the learner should be able to change, and anything to check."></textarea>
        <div class="help">Mention the section, the interactions you want, and checks such as "check that the weights sum to one".</div>
        <label for="audience">Who it is for</label>
        <input type="text" id="audience" name="audience" placeholder="Second-year engineering undergraduates" autocomplete="off">
        <div class="examples"><span>Try an example:</span><span id="examples"></span></div>
        <div id="keybox" hidden>
          <label for="api_key">OpenRouter API key</label>
          <input type="password" id="api_key" autocomplete="off" placeholder="sk-or-…">
          <div class="help">Used only for this generation and never saved.</div>
        </div>
        <details class="adv">
          <summary>Model</summary>
          <label for="model">OpenRouter model id</label>
          <input type="text" id="model" autocomplete="off">
        </details>
        <div class="actions">
          <button class="btn primary" id="go" type="submit">Generate page</button>
          <button class="btn quiet" id="cancel" type="button" hidden>Cancel</button>
        </div>
        <div class="err" id="err" role="alert"></div>
      </form>
      <div class="recent" id="recentbox" hidden>
        <h3>Recent pages</h3>
        <ul id="recent"></ul>
      </div>
    </section>
    <section class="sheet stage" id="stage" aria-live="polite">
      <div class="empty" id="empty">
        <h2>Your page will appear here</h2>
        <p>The generator drafts the explanation, runs its calculations through automatic checks, fixes what fails, and builds a page with live controls.</p>
        <ol>
          <li>Paste a paper link, or pick an example.</li>
          <li>Say what to explain and who it is for.</li>
          <li>Select Generate page.</li>
        </ol>
      </div>
      <div id="run" hidden>
        <div class="run-head"><h2 id="run-title">Generating your page</h2><span class="clock" id="clock"></span></div>
        <ol class="steps" id="steps"></ol>
        <ul class="log" id="log"></ul>
        <div id="result" hidden>
          <div class="stats" id="stats"></div>
          <div class="result-actions">
            <a class="btn quiet" id="open" target="_blank" rel="noopener">Open in a new tab</a>
            <a class="btn quiet" id="download">Download HTML</a>
            <a class="btn quiet" id="trace" target="_blank" rel="noopener">View the trace</a>
          </div>
        </div>
        <div class="failbox" id="fail" hidden></div>
      </div>
      <iframe id="frame" title="Generated page" hidden></iframe>
    </section>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
let CFG = null, current = null, timer = null;
const fmt = n => (n == null ? "—" : Number(n).toLocaleString());

function setBusy(b) { $("go").disabled = b; $("go").textContent = b ? "Generating…" : "Generate page"; $("cancel").hidden = !b; }

async function loadConfig() {
  const r = await fetch("/api/config"); CFG = await r.json();
  const ks = $("keystate");
  if (CFG.has_key) { ks.innerHTML = "API key: <b>found</b>"; }
  else { ks.innerHTML = "API key: <b>not set</b>"; ks.classList.add("missing"); $("keybox").hidden = false; }
  $("model").value = CFG.model;
  const ex = $("examples"); ex.textContent = "";
  CFG.examples.forEach(e => {
    const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = e.name;
    b.addEventListener("click", () => { $("source_url").value = e.source_url || ""; $("focus").value = e.focus || ""; $("audience").value = e.audience || ""; $("err").textContent = ""; });
    ex.appendChild(b); ex.appendChild(document.createTextNode(" "));
  });
  renderHistory(CFG.history);
}

function renderHistory(items) {
  const ul = $("recent"); ul.textContent = "";
  $("recentbox").hidden = !items.length;
  items.forEach(it => {
    const li = document.createElement("li"), b = document.createElement("button");
    const t = document.createElement("span"); t.className = "rt"; t.textContent = it.title || "Untitled page";
    const m = document.createElement("span"); m.className = "rm";
    const when = it.started ? new Date(it.started * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}) : "";
    const state = {running: "generating", done: "ready", failed: "failed", cancelled: "cancelled"}[it.state] || it.state;
    m.textContent = [when, state, it.stats && it.stats.seconds ? Math.round(it.stats.seconds) + " s" : ""].filter(Boolean).join(", ");
    b.appendChild(t); b.appendChild(m); b.addEventListener("click", () => watch(it.id));
    li.appendChild(b); ul.appendChild(li);
  });
}

function renderSteps(st) {
  const ol = $("steps"); ol.textContent = "";
  st.steps.forEach((name, i) => {
    const li = document.createElement("li"); li.textContent = name;
    const seen = (st.seen || []).includes(i), running = st.state === "running";
    if (running && i === st.step) li.className = "now";
    else if (seen && (i < st.step || !running)) li.className = "done";
    else if (i < st.step || (!running && st.state === "done")) { li.className = "skip"; li.title = "Not needed for this page"; }
    ol.appendChild(li);
  });
}

function render(st) {
  $("empty").hidden = true; $("run").hidden = false;
  renderSteps(st);
  $("clock").textContent = Math.round(st.elapsed || (st.stats && st.stats.seconds) || 0) + " s";
  const log = $("log"); log.textContent = "";
  (st.log || []).forEach(t => { const li = document.createElement("li"); li.textContent = t; log.appendChild(li); });
  log.hidden = !(st.log || []).length; log.scrollTop = log.scrollHeight;
  if (st.state === "running") { $("run-title").textContent = "Generating your page"; $("result").hidden = true; $("fail").hidden = true; $("frame").hidden = true; return; }
  if (st.state === "done") {
    $("run-title").textContent = st.title || "Page ready";
    const s = st.stats || {}, stats = $("stats"); stats.textContent = "";
    const verdict = document.createElement("span");
    const clean = !s.unresolved;
    verdict.className = "verdict " + (clean ? "ok" : "warn");
    verdict.textContent = clean ? "Page ready, every check passed" : "Page ready, " + s.unresolved + " issue" + (s.unresolved === 1 ? "" : "s") + " could not be fixed";
    stats.appendChild(verdict);
    [["Model calls", s.calls], ["Tokens", fmt(s.tokens)], ["Time", s.seconds != null ? Math.round(s.seconds) + " s" : "—"]].forEach(([k, v]) => {
      const sp = document.createElement("span"); sp.appendChild(document.createTextNode(k + " ")); const b = document.createElement("b"); b.textContent = v; sp.appendChild(b); stats.appendChild(sp);
    });
    $("open").href = "/runs/" + st.id + "/index.html"; $("download").href = "/runs/" + st.id + "/index.html?download=1";
    $("trace").href = "/runs/" + st.id + "/trace.jsonl";
    $("result").hidden = false; $("fail").hidden = true;
    const fr = $("frame"); const src = "/runs/" + st.id + "/index.html";
    if (fr.getAttribute("src") !== src) fr.setAttribute("src", src);
    fr.hidden = false;
  } else {
    $("run-title").textContent = st.state === "cancelled" ? "Generation cancelled" : "The page could not be generated";
    $("result").hidden = true; $("frame").hidden = true;
    const f = $("fail"); f.hidden = st.state === "cancelled";
    f.textContent = (st.error || "The generator stopped without writing a page.") + " Check the inputs and the API key, then try again.";
  }
}

async function poll() {
  if (!current) return;
  try {
    const r = await fetch("/api/status/" + current); const st = await r.json();
    if (!r.ok) throw new Error(st.error || "Status unavailable.");
    render(st);
    if (st.state === "running") { timer = setTimeout(poll, 1000); return; }
    setBusy(false); refreshHistory();
  } catch (e) { timer = setTimeout(poll, 2000); }
}

async function refreshHistory() { try { const r = await fetch("/api/config"); const c = await r.json(); renderHistory(c.history); } catch (e) {} }

function watch(id) { clearTimeout(timer); current = id; $("frame").removeAttribute("src"); poll(); }

$("form").addEventListener("submit", async ev => {
  ev.preventDefault(); $("err").textContent = "";
  const body = {source_url: $("source_url").value, focus: $("focus").value, audience: $("audience").value, model: $("model").value, api_key: $("api_key").value};
  const labels = {source_url: "paper link", focus: "what to explain", audience: "who it is for"};
  const missing = Object.keys(labels).filter(k => !body[k].trim());
  if (missing.length) { const n = missing.map(k => labels[k]); const text = n.length === 1 ? n[0] : n.slice(0, -1).join(", ") + " and " + n[n.length - 1]; $("err").textContent = "Fill in the " + text + "."; $(missing[0]).focus(); return; }
  setBusy(true);
  try {
    const r = await fetch("/api/generate", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "The generator could not start.");
    watch(d.id); refreshHistory();
  } catch (e) { $("err").textContent = e.message; setBusy(false); }
});

$("cancel").addEventListener("click", async () => { if (current) { await fetch("/api/cancel/" + current, {method: "POST"}); } });

loadConfig();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Local web interface for Paper to Playground")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        print(f"Port {args.port} is in use. Start with another port, e.g. python webui.py --port 8801")
        return 1
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Paper to Playground is running at {url}  (press Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with LOCK:
            for job in JOBS.values():
                job.cancel()
    return 0


if __name__ == "__main__":
    sys.exit(main())
