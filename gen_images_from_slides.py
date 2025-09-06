#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrate slide concepts -> code generator -> images, with parallel calls.

Two ways to run:
1) As a function from your notebook/script (uses in-memory `inputs` and `outputs_context_based`)
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
 - For LaTeX or R outputs, have `pdflatex` / `Rscript` installed if the model chooses them.
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
- Choose the best language to produce a clear, self-contained FIGURE that illustrates the provided concept and explanation.
- For PYTHON: use matplotlib + numpy; DO NOT use seaborn. Save an image (PNG or PDF) in the working directory.
- For LaTeX: provide a fully compilable standalone .tex (with \\documentclass, \\usepackage, etc.). Compiling with `pdflatex` should produce a PDF figure.
- For R: use base or ggplot2; call `ggsave()` or similar to save an image.
- No external data/files; everything self-contained.
- The figure must be saved to the working directory with a sensible name.
- Include minimal dependencies in the arrays.
- Return ONLY valid JSON (no prose, no markdown, no backticks).
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

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _validate_payload(d: Dict[str, Any]) -> str:
    required = ["language", "filename", "code", "run_instructions",
                "python_packages", "r_packages", "latex_requires"]
    for k in required:
        if k not in d:
            raise ValueError(f"Missing key in model JSON: {k}")

    lang = d["language"].lower().strip()
    fname = d["filename"]
    if lang == "python" and not fname.endswith(".py"):
        raise ValueError("For python, filename must end with .py")
    if lang == "latex" and not (fname.endswith(".tex") or fname.endswith(".tikz")):
        raise ValueError("For latex, filename must end with .tex or .tikz")
    if lang == "r" and not fname.endswith(".R"):
        raise ValueError("For r, filename must end with .R")
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
        cmd = ["python", filename]
    elif lang == "r":
        cmd = ["Rscript", filename]
    elif lang == "latex":
        # compile twice for references; keep it simple here
        rc1, so1, se1 = await _run_subprocess(["pdflatex", "-interaction=nonstopmode", filename], workdir)
        if rc1 != 0:
            return rc1, so1, se1
        rc2, so2, se2 = await _run_subprocess(["pdflatex", "-interaction=nonstopmode", filename], workdir)
        return rc2, so1 + so2, se1 + se2
    else:
        return 2, "", f"Unsupported language: {lang}"
    return await _run_subprocess(cmd, workdir)

def _sanitize_name(s: str) -> str:
    keep = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s.strip())
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
            if attempt + 1 < max_attempts:
                return await _gen_one_figure(client, semaphore, model, topic_input, slide, outdir, attempt+1, max_attempts)
            return {"ok": False, "topic": topic_input, "slide_id": sid, "slide_name": sname, "error": err}

    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
        lang = _validate_payload(data)
    except Exception as e:
        err = f"Bad JSON or schema (attempt {attempt+1}/{max_attempts}): {e}\n{content}"
        if attempt + 1 < max_attempts:
            return await _gen_one_figure(client, semaphore, model, topic_input, slide, outdir, attempt+1, max_attempts)
        return {"ok": False, "topic": topic_input, "slide_id": sid, "slide_name": sname, "error": err}

    # Write code to file
    codefile = workdir / data["filename"]
    codefile.write_text(data["code"].rstrip() + "\n", encoding="utf-8")

    # Execute
    rc, so, se = await _execute_generated_code(lang, workdir, codefile.name)
    images = _extract_saved_files(workdir)

    return {
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

    tasks = []
    for item in items:
        topic = item["input"]
        for slide in item["concepts"]:
            tasks.append(_gen_one_figure(client, sem, model, topic, slide, base))

    results = []
    for fut in asyncio.as_completed(tasks):
        res = await fut
        # Pretty progress
        tag = f"[{res.get('topic','?')}] slide={res.get('slide_id','?')} «{res.get('slide_name','')}»"
        if res.get("ok"):
            print(f"✓ Generated: {tag}  ->  {res.get('workdir')}")
        else:
            print(f"✗ Failed:    {tag}  ->  {res.get('error','unknown error')}", file=sys.stderr)
        results.append(res)

    return results

def run_from_memory(
    inputs: List[str],
    outputs_context_based: List[Dict[str, Any]],
    outdir: str = "viz_outputs",
    model: str = DEFAULT_MODEL,
    max_parallel: int = 6
) -> List[Dict[str, Any]]:
    """
    Use your existing Python variables: `inputs` and `outputs_context_based`.
    Returns a list of result dicts.
    """
    # Normalize into items [{"input": str, "concepts": [...]}, ...]
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

if __name__ == "__main__":
    _cli()
