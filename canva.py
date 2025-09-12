#!/usr/bin/env python3
# -- coding: utf-8 --

"""
Integrate slide concepts -> code generator -> images, with parallel calls.

Two ways to run:
1) As a function from your notebook/script (uses in-memory inputs and outputs_context_based)
     from gen_images_from_slides import run_from_memory
     run_from_memory(inputs, outputs_context_based, outdir="viz_outputs", max_parallel=6)

2) As a CLI reading a JSON file:
     python gen_images_from_slides.py --json slides.json --outdir viz_outputs --max-parallel 6
   Where slides.json looks like:
     [
       {"input": "explain me projectile motion using football",
        "concepts": [{"id":1,"name":"...","detailed_explanation":"..."}, ...]
       },
       ...
     ]

Requirements:
 - Python 3.9+
 - openai>=1.0.0
 - For LaTeX or R outputs, have pdflatex / Rscript installed if the model chooses them.
 - Matplotlib + numpy for Python outputs (the model will request them when needed).
"""

import os
import sys
import json
import time
import math
import argparse
import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from openai._types import NotGiven

# -------------------- Tunables --------------------

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
SYSTEM_INSTRUCTIONS = """
You output STRICT JSON ONLY — a single JSON object with this schema:
{
  "language": "python" | "latex" | "r",
  "filename": "<str: .py for python, .tex/.tikz for LaTeX, .R for R>",
  "code": "<full runnable code>",
  "run_instructions": "<how to run/compile>",
  "python_packages": ["..."],
  "r_packages": ["..."],
  "latex_requires": ["..."]
}

Guidelines:
- Produce a clear, self-contained FIGURE illustrating the concept.
- *Absolutely avoid text/label overlap*:
  - Prefer a readable figsize (e.g., ≥ (8,5)).
  - Keep legends/titles concise; place crowded legends *below* or to the side.
  - Leave margins so labels/ticks aren’t clipped.
- PYTHON:
  - Use matplotlib + numpy only (no seaborn).
  - Before saving, call plt.tight_layout(). Prefer constrained_layout=True in figure/subplots.
  - If many labels, consider bbox_to_anchor or move legend to bottom.
  - Save image (PNG or PDF) in working directory.
- LaTeX:
  - Provide fully compilable standalone doc; add a small border (e.g., border=5pt) to avoid clipping.
- R:
  - Use base or ggplot2. Ensure margins and avoid clipping (e.g., theme(plot.margin=unit(c(10,10,10,10),'pt')),
    coord_cartesian(clip='off')). Save with ggsave().
- No external data/files; everything self-contained.
- Keep dependency arrays minimal but sufficient.
- Return ONLY valid JSON (no prose).
"""

USER_TEMPLATE = """
Task: Generate runnable code to create ONE figure that best illustrates this concept for learners.

Context:
- Topic input: {topic_input}
- Slide:
  - id: {sid}
  - name: {sname}
  - detailed_explanation: {sexpl}

Preferences:
- Preferred language: auto (choose the best)
- The code must save an image/PDF to the working directory.
- Make the figure clean, labeled, and pedagogically useful.

Return ONLY a single JSON object per the schema. Ensure the file extension matches the language.
"""

# -------------------- Helpers --------------------
import re

def _massage_code_for_layout(lang: str, code: str) -> str:
    """
    Best-effort, non-invasive fixes to reduce text overlap in generated code.
    Currently only patches Python/matplotlib. Other languages unchanged.
    """
    if lang != "python":
        return code

    patched = code

    # Ensure non-interactive backend for headless runs
    if "matplotlib.use(" not in patched:
        patched = "import matplotlib as _mpl\n_mpl.use('Agg')\n" + patched

    # Ensure matplotlib imported (some snippets rely on implicit import)
    if "import matplotlib.pyplot as plt" not in patched:
        lines = patched.splitlines()
        insert_idx = 0
        for i, ln in enumerate(lines[:15]):
            if "import numpy as np" in ln:
                insert_idx = i + 1
        lines.insert(insert_idx, "import matplotlib.pyplot as plt")
        patched = "\n".join(lines)

    # Add constrained_layout=True to figure/subplots if not already there
    def _add_constrained_layout_to_figure(m):
        inner = m.group(1) or ""
        if "constrained_layout" in inner:
            return f"plt.figure({inner})"
        inner = inner.strip()
        if inner == "":
            return "plt.figure(constrained_layout=True)"
        return f"plt.figure(constrained_layout=True, {inner})"

    patched = re.sub(r"plt\.figure\((.*?)\)", _add_constrained_layout_to_figure, patched)

    def _add_constrained_layout_to_subplots(m):
        inner = m.group(1) or ""
        if "constrained_layout" in inner:
            return f"plt.subplots({inner})"
        inner = inner.strip()
        if inner == "":
            return "plt.subplots(constrained_layout=True)"
        return f"plt.subplots(constrained_layout=True, {inner})"

    patched = re.sub(r"plt\.subplots\((.*?)\)", _add_constrained_layout_to_subplots, patched)

    # Insert tight_layout() immediately before each savefig(...)
    def _tight_before_savefig(m):
        return f"plt.tight_layout()\n{m.group(0)}"
    patched = re.sub(r"plt\.savefig\([^\)]*\)", _tight_before_savefig, patched)

    # Make legends safer if bare
    patched = re.sub(r"plt\.legend\(\s*\)", "plt.legend(loc='best', frameon=True)", patched)

    return patched


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text if text is not None else "", encoding="utf-8")

def _safe_tail(s: Optional[str], n: int = 240) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= n else s[-n:]

def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _validate_payload(d: Dict[str, Any]) -> str:
    required = ["language", "filename", "code", "run_instructions",
                "python_packages", "r_packages", "latex_requires"]
    for k in required:
        if k not in d:
            raise ValueError(f"Missing key in model JSON: {k}")

    lang = d["language"].lower().strip()
    fname = d["filename"]
    lower = fname.lower()

    if lang == "python" and not lower.endswith(".py"):
        raise ValueError("For python, filename must end with .py")
    if lang == "latex" and not (lower.endswith(".tex") or lower.endswith(".tikz")):
        raise ValueError("For latex, filename must end with .tex or .tikz")
    if lang == "r" and not lower.endswith(".r"):
        raise ValueError("For r, filename must end with .R or .r")
    return lang

async def _run_subprocess(cmd: List[str], cwd: Path, timeout: int = 300) -> Tuple[int, str, str]:
    """Run a command asynchronously and capture output."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, "", f"Timeout after {timeout}s"
    rc = proc.returncode
    return rc, stdout.decode(errors="ignore"), stderr.decode(errors="ignore")

async def _execute_generated_code(lang: str, workdir: Path, filename: str) -> Tuple[int, str, str]:
    if lang == "python":
        os.environ.setdefault("MPLBACKEND", "Agg")
        cmd = ["python", filename]
    elif lang == "r":
        cmd = ["Rscript", filename]
    elif lang == "latex":
        rc1, so1, se1 = await _run_subprocess(["pdflatex", "-interaction=nonstopmode", filename], workdir)
        if rc1 != 0:
            return rc1, so1, se1
        rc2, so2, se2 = await _run_subprocess(["pdflatex", "-interaction=nonstopmode", filename], workdir)
        return rc2, so1 + so2, se1 + se2
    else:
        return 2, "", f"Unsupported language: {lang}"
    return await _run_subprocess(cmd, workdir)

def _sanitize_name(s: str) -> str:
    keep = "".join(c if c.isalnum() or c in ("-", "") else "" for c in s.strip())
    return keep[:80] if keep else "untitled"

def _slide_dir(base: Path, topic: str, slide_id: Any, slide_name: str) -> Path:
    tdir = _sanitize_name(topic)
    sdir = f"{str(slide_id).zfill(2)}_{_sanitize_name(slide_name)}"
    return base / tdir / sdir

def _extract_saved_files(workdir: Path) -> List[str]:
    exts = {".png", ".pdf", ".jpg", ".jpeg", ".svg"}
    return [f.name for f in workdir.iterdir() if f.is_file() and f.suffix.lower() in exts]

# -------------------- Core task --------------------

async def _gen_one_figure(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    topic_input: str,
    slide: Dict[str, Any],
    outdir: Path,
    attempt: int = 0,
    max_attempts: int = 3
) -> Dict[str, Any]:
    sid   = slide.get("id", "NA")
    sname = slide.get("name", "Concept")
    sexpl = slide.get("detailed_explanation", "")

    workdir = _slide_dir(outdir, topic_input, sid, sname)
    _ensure_dir(workdir)

    user_msg = USER_TEMPLATE.format(
        topic_input=topic_input,
        sid=sid,
        sname=sname,
        sexpl=sexpl
    )

    backoff = 2 ** attempt

    async with semaphore:
        if attempt > 0:
            await asyncio.sleep(backoff)  # simple backoff before retry

        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            err = f"OpenAI API error (attempt {attempt+1}/{max_attempts}): {e}"
            # persist a minimal failure record
            mini = {
                "ok": False,
                "topic": topic_input,
                "slide_id": sid,
                "slide_name": sname,
                "workdir": str(workdir),
                "language": None,
                "filename": None,
                "run_exit_code": None,
                "stdout": "",
                "stderr": "",
                "saved_artifacts": [],
                "model_json": None,
                "error": err,
            }
            _write_text(workdir / "result.json", json.dumps(mini, ensure_ascii=False, indent=2))
            _write_text(workdir / "stderr.log", err)
            return mini

    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
        lang = _validate_payload(data)
    except Exception as e:
        err = f"Bad JSON or schema (attempt {attempt+1}/{max_attempts}): {e}"
        mini = {
            "ok": False,
            "topic": topic_input,
            "slide_id": sid,
            "slide_name": sname,
            "workdir": str(workdir),
            "language": None,
            "filename": None,
            "run_exit_code": None,
            "stdout": "",
            "stderr": "",
            "saved_artifacts": [],
            "model_json": None,
            "error": err,
        }
        _write_text(workdir / "raw_content.txt", content or "")
        _write_text(workdir / "result.json", json.dumps(mini, ensure_ascii=False, indent=2))
        _write_text(workdir / "stderr.log", err + "\n\n" + (content or ""))
        return mini

    # Write code to file (with gentle layout patch for Python)
    code_text = (data["code"].rstrip() + "\n")
    code_text = _massage_code_for_layout(lang, code_text)
    codefile = workdir / data["filename"]
    codefile.write_text(code_text, encoding="utf-8")

    # Save the raw model JSON for reproducibility
    _write_text(workdir / "model_response.json", json.dumps(data, ensure_ascii=False, indent=2))

    # Execute
    rc, so, se = await _execute_generated_code(lang, workdir, codefile.name)
    images = _extract_saved_files(workdir)

    result: Dict[str, Any] = {
        "ok": rc == 0 and len(images) > 0,
        "topic": topic_input,
        "slide_id": sid,
        "slide_name": sname,
        "workdir": str(workdir),
        "language": lang,
        "filename": codefile.name,
        "run_exit_code": rc,
        "stdout": so,
        "stderr": se,
        "saved_artifacts": images,
        "model_json": data
    }

    if not result["ok"]:
        err_tail = _safe_tail(se, 480) or "Non-zero exit or no image artifacts produced."
        result["error"] = f"exec failed (exit={rc}): {err_tail}"

    # Persist per-slide logs
    _write_text(workdir / "stdout.log", so or "")
    _write_text(workdir / "stderr.log", se or "")
    _write_text(workdir / "result.json", json.dumps(result, ensure_ascii=False, indent=2))

    return result

# -------------------- Public runners --------------------

async def _run_async(
    items: List[Dict[str, Any]],
    outdir: str,
    model: str,
    max_parallel: int
) -> List[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var is not set")
    client = AsyncOpenAI(api_key=api_key)

    sem = asyncio.Semaphore(max_parallel)
    base = Path(outdir).resolve()
    _ensure_dir(base)
    runlog = base / "pipeline_results.jsonl"

    tasks = []
    for item in items:
        topic = item["input"]
        for slide in item["concepts"]:
            tasks.append(_gen_one_figure(client, sem, model, topic, slide, base))

    results = []
    for fut in asyncio.as_completed(tasks):
        res = await fut
        tag = f"[{res.get('topic','?')}] slide={res.get('slide_id','?')} «{res.get('slide_name','')}»"

        # Append to global JSONL
        _append_jsonl(runlog, {
            "ts": int(time.time()),
            **res
        })

        if res.get("ok"):
            arts = ", ".join(res.get("saved_artifacts", []))
            print(f"✓ Generated: {tag}  ->  {res.get('workdir')}  [{arts}]")
        else:
            err = res.get("error") or _safe_tail(res.get("stderr"), 240) or "no stderr"
            code = res.get("run_exit_code")
            print(f"✗ Failed:    {tag}  ->  exit={code}  err={err}", file=sys.stderr)

        results.append(res)

    return results

def run_from_memory(
    inputs: List[str],
    outputs_context_based: List[Dict[str, Any]],
    outdir: str = "viz_outputs",
    model: str = DEFAULT_MODEL,
    max_parallel: int = 6
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for i, topic in enumerate(inputs):
        concepts = outputs_context_based[i].get("concepts") or \
                   outputs_context_based[i].get("high_school", {}).get("concepts") or []
        items.append({"input": topic, "concepts": concepts})
    return asyncio.run(_run_async(items, outdir, model, max_parallel))

def run_from_json(
    json_path: str,
    outdir: str = "viz_outputs",
    model: str = DEFAULT_MODEL,
    max_parallel: int = 6
) -> List[Dict[str, Any]]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of {input, concepts}")
    return asyncio.run(_run_async(data, outdir, model, max_parallel))

# -------------------- CLI --------------------

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--json", help="Path to slides JSON (list of {input, concepts})")
    p.add_argument("--outdir", default="viz_outputs")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-parallel", type=int, default=6)
    args = p.parse_args()

    if not args.json:
        print("ERROR: --json is required when using CLI. Or import run_from_memory().", file=sys.stderr)
        sys.exit(2)

    results = run_from_json(args.json, outdir=args.outdir, model=args.model, max_parallel=args.max_parallel)
    # Save a summary
    summary_path = Path(args.outdir) / "summary_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[✓] Wrote {summary_path}")
    print(f"[i] Run-wide log (JSONL): {Path(args.outdir) / 'pipeline_results.jsonl'}")

if _name_ == "_main_":
    _cli()