"""Dev helper: run agent.py on many cases in parallel and summarize traces."""
import json, subprocess, sys, time, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
PY = sys.executable
tag = sys.argv[1]
cases = sys.argv[2:]
def run(case):
    name = Path(case).stem
    out = Path("runs") / tag / name
    t = time.time()
    p = subprocess.run([PY, str(Path(__file__).resolve().parent.parent / "agent.py"), "--input", case, "--output", str(out), "--model", os.environ.get("MODEL", "deepseek/deepseek-v4.1-flash")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.time() - t
    fin = {}
    errs = []
    try:
        for ln in open(out / "trace.jsonl", encoding="utf-8"):
            ev = json.loads(ln)
            if ev["stage"] == "finish": fin = ev
            if ev["stage"] in ("check", "revise") and ev.get("issues"):
                errs.append((ev["stage"], ev.get("round"), [i["message"][:110] for i in ev["issues"] if i["severity"] == "error"]))
    except Exception as e:
        fin = {"result": f"no trace: {e}"}
    return name, p.returncode, wall, fin, errs, p.stderr[-500:]
with ThreadPoolExecutor(12) as ex:
    for name, rc, wall, fin, errs, err in ex.map(run, cases):
        print(f"{name:12s} rc={rc} wall={wall:5.1f}s {fin.get('result')} calls={fin.get('requests')} tok={fin.get('total_tokens')} (p={fin.get('prompt_tokens')} c={fin.get('completion_tokens')}) unresolved={fin.get('unresolved_errors')}")
        for e in errs: print("    ", e)
        if rc != 0: print("    STDERR:", err)
