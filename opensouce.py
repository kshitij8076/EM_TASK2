#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import base64
import asyncio
import pickle
import requests
from datetime import datetime
from tempfile import mkdtemp
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===========================
# Config: Ollama (open-source)
# ===========================
USE_OLLAMA = True
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

STAGE1_MODEL = os.getenv("OLLAMA_MODEL_STAGE1", "llama3.1:8b")
STAGE2_MODEL = os.getenv("OLLAMA_MODEL_STAGE2", "llama3.1:8b")
STAGE3_MODEL = os.getenv("OLLAMA_MODEL_STAGE3", "qwen3-coder:30b")
# STAGE4_MODEL = os.getenv("OLLAMA_MODEL_STAGE4", "llava:13b")  # vision model
STAGE4_MODEL = os.getenv("OLLAMA_MODEL_STAGE4", "qwen3-coder:30b")  # vision model

OLLAMA_OPTIONS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "num_ctx": 8192,  # adjust per model capacity
}

OUTDIR = os.getenv("OUTDIR", "viz_outputs_temps")
MAX_PARALLEL_STAGE3 = int(os.getenv("MAX_PARALLEL_STAGE3", "6"))

def ollama_chat(model: str, messages: list, fmt: Optional[str] = None,
                stream: bool = False, options: Optional[dict] = None) -> str:
    """Call Ollama /api/chat (blocking), return assistant text."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {**OLLAMA_OPTIONS, **(options or {})},
    }
    if fmt:
        payload["format"] = fmt  # "json" for structured outputs
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]

# ===========================
# RAG (Docling + Chroma) - unchanged from your script
# ===========================
from typing import List as _List, Dict as _Dict, Optional as _Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

class LangChainDoclingRAG:
    """RAG system using LangChain with Docling for PDF extraction and semantic search"""
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        export_type: ExportType = ExportType.DOC_CHUNKS,
        persist_directory: _Optional[str] = None,
        max_token_length: int = 450  # ~512 for MiniLM
    ):
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.export_type = export_type
        self.persist_directory = persist_directory or mkdtemp()
        self.max_token_length = max_token_length

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.retriever = None
        self.documents = []

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def _truncate_long_documents(self, documents: _List[Document], max_length: int = None) -> _List[Document]:
        if max_length is None:
            max_length = self.max_token_length * 4
        processed_docs = []
        for doc in documents:
            if self._estimate_tokens(doc.page_content) > self.max_token_length:
                content = doc.page_content.strip()
                sentences = content.split('. ')
                chunks, current = [], ""
                for sentence in sentences:
                    sentence = sentence.strip() + '. '
                    if self._estimate_tokens(current + sentence) > self.max_token_length:
                        if current:
                            chunks.append(current.strip())
                            overlap = current.split('. ')[-2:]
                            current = ('. '.join(overlap) + '. ' if overlap and overlap[0] else '') + sentence
                        else:
                            current = sentence
                    else:
                        current += sentence
                if current.strip():
                    chunks.append(current.strip())
                for i, chunk_content in enumerate(chunks):
                    if len(chunk_content.strip()) > 20:
                        processed_docs.append(
                            Document(page_content=chunk_content, metadata={**doc.metadata, 'chunk_index': i, 'is_split': True})
                        )
            else:
                processed_docs.append(doc)
        return processed_docs

    def extract_documents_from_pdfs(self, pdf_paths: _List[str]) -> _List[Document]:
        all_docs = []
        for pdf_path in pdf_paths:
            try:
                if self.export_type == ExportType.DOC_CHUNKS:
                    loader = DoclingLoader(
                        file_path=[pdf_path],
                        export_type=self.export_type,
                        chunker=HybridChunker(tokenizer=self.embedding_model_name)
                    )
                else:
                    loader = DoclingLoader(file_path=[pdf_path], export_type=self.export_type)

                docs = loader.load()
                for doc in docs:
                    doc.metadata.update({'source_file': os.path.basename(pdf_path), 'full_path': pdf_path})
                    if 'doc_items' in doc.metadata:
                        try:
                            doc_items = doc.metadata['doc_items']
                            if isinstance(doc_items, list) and doc_items:
                                first = doc_items[0]
                                if 'prov' in first and isinstance(first['prov'], list) and first['prov']:
                                    prov = first['prov'][0]
                                    if 'page_no' in prov: doc.metadata['page_number'] = prov['page_no']
                                    if 'bbox' in prov:
                                        bbox = prov['bbox']
                                        doc.metadata['bbox_left'] = bbox.get('l', 0)
                                        doc.metadata['bbox_top'] = bbox.get('t', 0)
                                        doc.metadata['bbox_right'] = bbox.get('r', 0)
                                        doc.metadata['bbox_bottom'] = bbox.get('b', 0)
                                if 'label' in first: doc.metadata['content_type'] = first['label']
                                if 'content_layer' in first: doc.metadata['content_layer'] = first['content_layer']
                        except Exception:
                            pass
                    if 'origin' in doc.metadata and isinstance(doc.metadata['origin'], dict):
                        origin = doc.metadata['origin']
                        if 'filename' in origin: doc.metadata['original_filename'] = origin['filename']
                        if 'mimetype' in origin: doc.metadata['mimetype'] = origin['mimetype']

                if self.export_type == ExportType.DOC_CHUNKS:
                    processed_docs = docs
                elif self.export_type == ExportType.MARKDOWN:
                    from langchain_text_splitters import MarkdownHeaderTextSplitter
                    header_splitter = MarkdownHeaderTextSplitter(
                        headers_to_split_on=[("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]
                    )
                    processed_docs = []
                    for doc in docs:
                        splits = header_splitter.split_text(doc.page_content)
                        for split in splits:
                            processed_docs.append(Document(page_content=split.page_content, metadata={**doc.metadata, **split.metadata}))
                else:
                    processed_docs = self.text_splitter.split_documents(docs)

                all_docs.extend(processed_docs)
            except Exception as e:
                print(f"✗ Error processing {pdf_path}: {e}")
                continue

        if all_docs:
            all_docs = filter_complex_metadata(all_docs)
            all_docs = self._truncate_long_documents(all_docs)

        self.documents = all_docs
        return all_docs

    def build_vector_store(self, documents: _Optional[_List[Document]] = None):
        if documents is None:
            documents = self.documents
        if not documents:
            raise ValueError("No documents to index. Extract documents first.")

        documents = filter_complex_metadata(documents)
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="docling_rag"
        )
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def add_pdf_to_index(self, pdf_path: str):
        new_docs = self.extract_documents_from_pdfs([pdf_path])
        if new_docs:
            new_docs = filter_complex_metadata(new_docs)
            if self.vectorstore is None:
                self.build_vector_store(new_docs)
            else:
                self.vectorstore.add_documents(new_docs)

    def search(self, query: str, top_k: int = 5) -> _List[_Dict]:
        if self.vectorstore is None:
            raise ValueError("Vector store not built/loaded. Call load_existing_vectorstore() first.")
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        results = []
        for doc, score in docs_with_scores:
            results.append({
                'text': doc.page_content,
                'source': doc.metadata.get('source_file', 'Unknown'),
                'score': float(1 - score),
                'metadata': doc.metadata
            })
        return results

    def setup_rag_chain(self, llm, prompt_template: _Optional[str] = None):
        if self.retriever is None:
            raise ValueError("Retriever not set up. Call build_vector_store() first.")
        if prompt_template is None:
            prompt_template = """Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {input}
Answer:"""
        prompt = PromptTemplate.from_template(prompt_template)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        return self.rag_chain

    def ask(self, question: str) -> _Dict:
        if not hasattr(self, 'rag_chain'):
            raise ValueError("RAG chain not set up. Call setup_rag_chain() first.")
        return self.rag_chain.invoke({"input": question})

    def save_index(self, path: str):
        metadata = {
            'embedding_model_name': self.embedding_model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'export_type': self.export_type,
            'persist_directory': self.persist_directory,
            'num_documents': len(self.documents)
        }
        with open(path, 'wb') as f:
            pickle.dump(metadata, f)

    def load_existing_vectorstore(self) -> bool:
        try:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name="docling_rag"
                )
                self.retriever = self.vectorstore.as_retriever()
                return True
            return False
        except Exception as e:
            print(f"Could not load existing vector store: {e}")
            return False

    def get_processed_files(self) -> set:
        if self.vectorstore is None:
            return set()
        try:
            collection = self.vectorstore._collection
            results = collection.get(include=['metadatas'])
            processed_files = set()
            for metadata in results['metadatas']:
                if 'source_file' in metadata:
                    processed_files.add(metadata['source_file'])
            return processed_files
        except Exception:
            return set()

    def process_pdfs_incrementally(self, pdf_paths: _List[str]) -> _List[Document]:
        self.load_existing_vectorstore()
        processed_files = self.get_processed_files()
        new_pdf_paths = []
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            if filename not in processed_files:
                new_pdf_paths.append(pdf_path)
        if not new_pdf_paths:
            print("All PDFs already processed!")
            return []
        new_docs = self.extract_documents_from_pdfs(new_pdf_paths)
        if new_docs:
            if self.vectorstore is None:
                self.build_vector_store(new_docs)
            else:
                new_docs = filter_complex_metadata(new_docs)
                new_docs = self._truncate_long_documents(new_docs)
                self.vectorstore.add_documents(new_docs)
                self.documents.extend(new_docs)
        return new_docs

# ===========================
# Pipeline helpers & prompts
# ===========================

_RAG_CACHE: Dict[str, Optional[LangChainDoclingRAG]] = {}

def _normalize_class_label(raw: str) -> str:
    s = str(raw).strip().lower().replace("grade", "").replace("class", "").strip()
    roman = {"ix": "9", "x": "10", "xi": "11", "xii": "12", "viii": "8", "vii": "7", "vi": "6"}
    if s in roman:
        s = roman[s]
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return f"class_{digits}"
    return f"class_{s.replace(' ', '_')}"

def load_rag_for_class(class_level: str,
                       base_dir: str = "rag",
                       embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Optional[LangChainDoclingRAG]:
    key = _normalize_class_label(class_level)
    print("key is", key)
    if key in _RAG_CACHE:
        return _RAG_CACHE[key]

    persist_dir = os.path.join(base_dir, key)
    print(persist_dir)
    rag = LangChainDoclingRAG(
        embedding_model=embedding_model,
        chunk_size=800,
        chunk_overlap=100,
        persist_directory=persist_dir
    )

    if not os.path.exists(persist_dir):
        _RAG_CACHE[key] = None
        return None

    if rag.load_existing_vectorstore():
        _RAG_CACHE[key] = rag
        return rag
    else:
        _RAG_CACHE[key] = None
        return None

PROMPT1_TEXT = r"""
Analyze an educational prompt by extracting and structuring content for a high school audience only. Focus exclusively on providing a thorough, comprehensive, and detailed breakdown of the main topic, context, measurable objectives (using Bloom's taxonomy), prerequisites, and all relevant key concepts. Ensure every concept necessary to fully understand the topic at the high school level is included with in-depth detail.

- Use advanced terminology, include mathematical and theoretical aspects, and reference real-world applications where appropriate.
- Each concept should be listed by name and explained with a stand-alone, detailed description as a list of clear points (never paragraphs). Each "brief" array should be as thorough as possible, fully unpacking the concept for a high school audience.
- Explanations must be independent of any specific example or scenario, but remain logically connected to the main topic.
- All elements must be reasoned out in a stepwise manner: first determine the main topic and context, then develop high school-appropriate learning objectives and prerequisites, and finally construct a logically ordered list of every relevant concept from foundational to advanced.

Return only the high school section in the following JSON structure (do not use markdown or code blocks):

{
  "high_school": {
    "main_topic": "",
    "context_object": "",
    "learning_objectives": [],
    "prerequisites": [],
    "concepts": [
      {
        "id": 1,
        "name": "",
        "brief": [
          "Detailed point 1",
          "Detailed point 2"
        ]
      }
    ]
  }
}

# Steps

1. Analyze the prompt to determine the main topic and context example.
2. Identify and clearly state 3–5 measurable learning objectives (Bloom’s taxonomy) for the high school level.
3. Specify all necessary prerequisites for high school students to access the topic.
4. List every relevant concept relating to the topic, ordered from foundation to advanced, ensuring thorough coverage.
5. For each concept, present a detailed "brief" as an array of clear, unpacked explanatory points suitable for high school comprehension.

# Output Format

Provide a single JSON object as shown above, containing only the "high_school" key, with all fields thoroughly and completely filled. Do not include markdown formatting or code blocks. Every "brief" should be a multi-point list (not a paragraph), as detailed and comprehensive as possible for the high school level.

# Notes
- Omit the elementary section entirely—only output high school content.
- Concepts must comprehensively cover the topic; persist in generating and explaining as many as are necessary.
- Each "brief" must have at least 2–3 points (often more if needed for rigor).
- Follow the reasoning first, answer last model: reason about each classification and explanation fully before producing the final JSON output.
- Be concise, highly detailed, and use high school-appropriate academic language throughout.
- Repeat: final output is one JSON object, high school section only, with complete and detailed educational content.
"""

PROMPT2_TEXT = r"""
Develop detailed, context-rich, engaging explanations for a set of educational concepts anchored in a provided real-world context object—analyzing and leveraging an additional RAG_CONTEXT when it is available—tailored to a specified educational level.

For each concept provided:
- Begin by analyzing both the {context_object} and, if present, the {RAG_CONTEXT}, considering how each can inform, illustrate, or deepen the explanation.
- Expand the concept into a thorough, 4-6 sentence explanation, ensuring clarity, connection, and depth appropriate for the specified level.
- Ensure all reasoning and examples meaningfully incorporate the {context_object} and, if available, information or details from the {RAG_CONTEXT}.
- Provide a specific real-world example involving the {context_object} (and RAG_CONTEXT when relevant).
- Include relevant formulas or mathematical relationships (in LaTeX) for high school; omit for elementary.
- For all but the first concept, explain explicitly how this concept builds on or connects to the previous concept.
- Address a common misconception per concept and provide a correction.

Persistence clause: Continue step-by-step until every concept has a complete, correct explanation, fully integrating both context_object and, if provided, RAG_CONTEXT, for the specified level.

Guidelines by Level:
- Elementary: Prioritize visual/intuitive understanding using analogies and accessible language; omit complex formulas.
- High School: Incorporate scientific principles, quantitative reasoning, and relevant formulas (in LaTeX), closely linked to both context_object and RAG_CONTEXT where possible.

# Steps
1. For each concept, first analyze how both {context_object} and {RAG_CONTEXT} (if provided) can illustrate or exemplify the concept.
2. Use insights from both contexts to construct a detailed explanation, ensuring educational completeness, specificity, and clear real-world relevance.
3. Integrate the RAG_CONTEXT into explanations and examples, not just the main context_object, whenever it enriches understanding.
4. Complete all requested fields for each concept, including connections and misconceptions.

# Output Format

Return a single JSON object with the following structure and fields:
{
  "concepts": [
    {
      "id": [sequential number, starting from 1],
      "name": [the concept name as given],
      "detailed_explanation": [4-6 sentence comprehensive explanation, explicitly referencing both contexts if RAG_CONTEXT is present],
      "formula": [mathematical formula in LaTeX for high school, null for elementary],
      "real_world_example": [specific example using the context object, and RAG_CONTEXT where possible],
      "connection_to_previous": [how this builds on the previous concept; null for first concept],
      "misconception_addressed": [common misconception and correction]
    }
  ]
}

# Notes
- If RAG_CONTEXT is provided, explanations and examples must meaningfully reference or utilize it alongside the main context_object.
- Output ONLY the finalized JSON structure—no extra text.
"""

def _parse_json_strict_or_repair(s: str) -> dict:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    s2 = s.strip()
    if s2.startswith("```"):
        s2 = re.sub(r"^```[a-zA-Z0-9_-]*\n|\n```$", "", s2, flags=re.S)
    s2 = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s2)
    return json.loads(s2)

def assemble_rag_query(hs_obj: Dict) -> str:
    topic = hs_obj.get("main_topic", "")
    concept_names = [c.get("name", "") for c in hs_obj.get("concepts", [])]
    top_concepts = ", ".join([c for c in concept_names if c][:10])
    return f"{topic}. Focus on: {top_concepts}"

def format_rag_context(results: List[Dict], max_chars: int = 2000) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        text = r.get("text", "").strip().replace("\n", " ")
        snippet = text if len(text) <= 600 else text[:600] + "..."
        parts.append(f"[{i}] {snippet}")
    ctx = "\n\n".join(parts)
    return ctx[:max_chars]

def get_rag_context_for_class(class_level: str, hs_obj: Dict, top_k: int = 5) -> Optional[str]:
    rag = load_rag_for_class(class_level)
    if rag is None:
        return None
    query = assemble_rag_query(hs_obj)
    try:
        results = rag.search(query, top_k=1)
        if not results:
            return None
        return format_rag_context(results)
    except Exception as e:
        print(f"RAG search failed for class '{class_level}': {e}")
        return None

def build_context_input(hs_obj: Dict, rag_context: Optional[str]) -> str:
    try:
        main_topic = hs_obj.get("main_topic", "")
        context_object = hs_obj.get("context_object", "")
        concepts = [c.get("name", "") for c in hs_obj.get("concepts", [])]
    except Exception:
        main_topic, context_object, concepts = "", "", []
    base = (
        f"main_topic: {main_topic}\n"
        f"context_object: {context_object}\n"
        f"level: High School\n"
        f"concepts: {concepts}\n"
    )
    if rag_context and rag_context.strip():
        base += "RAG_CONTEXT_BEGIN\n" + rag_context.strip() + "\nRAG_CONTEXT_END\n"
    return base

# ===========================
# Stage 1 & 2 using Ollama
# ===========================
def process_one_stage1(user_input: str) -> dict:
    try:
        messages = [
            {"role": "system", "content": PROMPT1_TEXT.strip()},
            {"role": "user",   "content": f'user_prompt: "{user_input}"'}
        ]
        txt = ollama_chat(STAGE1_MODEL, messages, fmt="json")
        return _parse_json_strict_or_repair(txt)
    except Exception as e:
        print(f"Stage 1 (Ollama) failed for '{user_input}': {e}")
        return {"high_school": {}}

def process_one_stage2(hs_wrapper: Dict, class_level: str) -> Dict:
    hs = hs_wrapper.get("high_school", {})
    rag_context = get_rag_context_for_class(class_level, hs)
    ctx_input = build_context_input(hs, rag_context)
    try:
        messages = [
            {"role": "system", "content": PROMPT2_TEXT.strip()},
            {"role": "user",   "content": ctx_input}
        ]
        txt = ollama_chat(STAGE2_MODEL, messages, fmt="json")
        return _parse_json_strict_or_repair(txt)
    except Exception as e:
        print(f"Stage 2 (Ollama) failed: {e}")
        return {"concepts": []}

# ===========================
# Stage 3: JSON → code → image
# ===========================
SYSTEM_INSTRUCTIONS_STAGE3 = """
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
- For LaTeX: provide a fully compilable standalone .tex. Compiling with `pdflatex` should produce a PDF figure.
- For R: use base or ggplot2; call `ggsave()` or similar to save an image.
- No external data/files; everything self-contained.
- The figure must be saved to the working directory with a sensible name.
- Include minimal dependencies in the arrays.
- Return ONLY valid JSON (no prose, no markdown, no backticks).
"""

USER_TEMPLATE_STAGE3 = """
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

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _validate_payload_stage3(d: Dict[str, Any]) -> str:
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
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(cwd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, "", f"Timeout after {timeout}s"
    return proc.returncode, stdout.decode(errors="ignore"), stderr.decode(errors="ignore")

async def _execute_generated_code(lang: str, workdir: Path, filename: str) -> Tuple[int, str, str]:
    if lang == "python":
        cmd = ["python", filename]
        return await _run_subprocess(cmd, workdir)
    if lang == "r":
        cmd = ["Rscript", filename]
        return await _run_subprocess(cmd, workdir)
    if lang == "latex":
        rc1, so1, se1 = await _run_subprocess(["pdflatex", "-interaction=nonstopmode", filename], workdir)
        if rc1 != 0:
            return rc1, so1, se1
        rc2, so2, se2 = await _run_subprocess(["pdflatex", "-interaction=nonstopmode", filename], workdir)
        return rc2, so1 + so2, se1 + se2
    return 2, "", f"Unsupported language: {lang}"

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

async def _gen_one_figure(
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

    user_msg = USER_TEMPLATE_STAGE3.format(
        topic_input=topic_input, sid=sid, sname=sname, sexpl=sexpl
    )

    backoff = 2 ** attempt
    async with semaphore:
        if attempt > 0:
            await asyncio.sleep(backoff)
        try:
            content = await asyncio.to_thread(
                ollama_chat,
                model,
                [{"role": "system", "content": SYSTEM_INSTRUCTIONS_STAGE3.strip()},
                 {"role": "user",   "content": user_msg}],
                "json"
            )
        except Exception as e:
            err = f"Ollama error (attempt {attempt+1}/{max_attempts}): {e}"
            if attempt + 1 < max_attempts:
                return await _gen_one_figure(semaphore, model, topic_input, slide, outdir, attempt+1, max_attempts)
            return {"ok": False, "topic": topic_input, "slide_id": sid, "slide_name": sname, "error": err}

    try:
        data = json.loads(content)
        lang = _validate_payload_stage3(data)
    except Exception as e:
        err = f"Bad JSON/schema (attempt {attempt+1}/{max_attempts}): {e}\n{content[:5000]}"
        if attempt + 1 < max_attempts:
            return await _gen_one_figure(semaphore, model, topic_input, slide, outdir, attempt+1, max_attempts)
        return {"ok": False, "topic": topic_input, "slide_id": sid, "slide_name": sname, "error": err}

    codefile = workdir / data["filename"]
    codefile.write_text(data["code"].rstrip() + "\n", encoding="utf-8")

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

async def stage3_run_async(
    items: List[Dict[str, Any]],
    outdir: str,
    model: str,
    max_parallel: int
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(max_parallel)
    base = Path(outdir).resolve()
    _ensure_dir(base)

    tasks = []
    for item in items:
        topic = item["input"]
        for slide in item["concepts"]:
            tasks.append(_gen_one_figure(sem, model, topic, slide, base))

    results = []
    for fut in asyncio.as_completed(tasks):
        res = await fut
        tag = f"[{res.get('topic','?')}] slide={res.get('slide_id','?')} «{res.get('slide_name','')}»"
        if res.get("ok"):
            print(f"✓ Generated: {tag}  ->  {res.get('workdir')}")
        else:
            print(f"✗ Failed:    {tag}  ->  {res.get('error','unknown error')}", file=sys.stderr)
        results.append(res)
    return results

def stage3_items_from_stage2(
    inputs_with_class: List[Tuple[str, str]],
    stage1_outputs: List[Dict],
    stage2_outputs: List[Dict]
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for i, (topic, _cls) in enumerate(inputs_with_class):
        concepts = stage2_outputs[i].get("concepts") or stage1_outputs[i].get("high_school", {}).get("concepts") or []
        items.append({"input": topic, "concepts": concepts})
    return items

# ===========================
# Stage 4: Modification (vision)
# ===========================
MOD_USER_TEMPLATE = """
You are given an existing slide figure (provided below as an image) that was generated for:
- Topic: {topic}
- Slide id: {sid}
- Slide name: {sname}

Modify the figure according to these instructions:

MODIFICATION_REQUEST:
{mod_instructions}

Constraints and goals:
- Output STRICT JSON ONLY with the same schema used previously (language, filename, code, run_instructions, python_packages, r_packages, latex_requires).
- Produce new code that recreates the original figure BUT with the requested modifications applied.
- If the original appears to be matplotlib: keep matplotlib + numpy (no seaborn). If LaTeX: return a fully compilable .tex. If R: use base/ggplot2.
- Ensure the code SAVES the final figure to the working directory. Use a new filename that includes '_modified' before the extension.
- Keep dependencies minimal.

Return only the JSON object; no extra text.
"""

def _encode_image_as_data_url(img_path: Path) -> str:
    ext = img_path.suffix.lower().lstrip(".")
    if ext not in {"png", "jpg", "jpeg", "webp"}:
        ext = "png"
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"

def _load_last_run_summary(summary_path: Path) -> List[Dict[str, Any]]:
    if not summary_path.exists():
        raise FileNotFoundError(f"summary_results.json not found at {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))

def _find_slide_entry(summary: List[Dict[str, Any]], slide_id: int, image_name: str) -> Dict[str, Any]:
    for entry in summary:
        if entry.get("slide_id") == slide_id:
            wd = Path(entry["workdir"])
            candidate = wd / image_name
            if candidate.exists():
                entry["_image_path"] = str(candidate)
                return entry
    for entry in summary:
        wd = Path(entry["workdir"])
        candidate = wd / image_name
        if candidate.exists():
            entry["_image_path"] = str(candidate)
            return entry
    raise FileNotFoundError(f"Could not find image '{image_name}' for slide id={slide_id}.")

def _propose_modified_code_from_image(
    model: str,
    image_path: Path,
    topic: str,
    slide_id: Any,
    slide_name: str,
    mod_instructions: str
) -> Dict[str, Any]:
    # Vision models on Ollama: base64 in 'images' field
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    user_payload = MOD_USER_TEMPLATE.format(
        topic=topic, sid=slide_id, sname=slide_name, mod_instructions=mod_instructions
    ).strip()
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS_STAGE3.strip()},
        {"role": "user",   "content": user_payload, "images": [b64]}
    ]
    try:
        content = ollama_chat(model, messages, fmt="json")
        data = json.loads(content)
    except Exception:
        content = ollama_chat(model, messages, fmt=None)
        data = _parse_json_strict_or_repair(content)
    _ = _validate_payload_stage3(data)
    return data

async def _execute_stage4_and_save(
    data: Dict[str, Any],
    workdir: Path
) -> Tuple[int, str, str, List[str]]:
    codefile = workdir / data["filename"]
    if codefile.exists():
        stem, suf = codefile.stem, codefile.suffix
        codefile = workdir / f"{stem}_{int(datetime.now().timestamp())}{suf}"
    codefile.write_text(data["code"].rstrip() + "\n", encoding="utf-8")
    rc, so, se = await _execute_generated_code(data["language"].lower(), workdir, codefile.name)
    images = _extract_saved_files(workdir)
    return rc, so, se, images

def _append_mod_log(base_outdir: Path, record: Dict[str, Any]) -> None:
    logf = base_outdir / "modifications.jsonl"
    with logf.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ===========================
# Orchestration
# ===========================
def build_slide_plans(inputs_with_class: List[Tuple[str, str]],
                      max_workers: int = 5) -> Tuple[List[Dict], List[Dict]]:
    stage1_outputs: List[Dict] = [None] * len(inputs_with_class)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(process_one_stage1, q): i for i, (q, _cls) in enumerate(inputs_with_class)}
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            stage1_outputs[idx] = fut.result()

    print("Done Process 1 ---------------------------------------")
    stage2_outputs: List[Dict] = [None] * len(inputs_with_class)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {
            ex.submit(process_one_stage2, stage1_outputs[i], inputs_with_class[i][1]): i
            for i in range(len(inputs_with_class))
        }
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            stage2_outputs[idx] = fut.result()
    return stage1_outputs, stage2_outputs

# ===========================
# Main
# ===========================
if __name__ == "__main__":
    # Example input
    inputs_with_class: List[Tuple[str, str]] = [
        ("explain me projectile motion using cricket", "10"),
    ]

    stage1, stage2 = build_slide_plans(inputs_with_class, max_workers=5)

    # Display Stage-2 plan
    for i, input_text in enumerate(inputs_with_class):
        print("\n\n")
        print("========================================")
        print(f"Input: {input_text}")
        print("========================================")
        for slide in stage2[i].get('concepts', []):
            print('-' * 200)
            print(f"\tSlide No. - {slide['id']}")
            print(f"\tName : {slide['name']}")
            print(f"\tDetailed Explanation :-\n\t{slide.get('detailed_explanation', '')}")
            print('-' * 200)

    items = stage3_items_from_stage2(inputs_with_class, stage1, stage2)

    # Ensure outdir exists
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

    # Run Stage 3 and persist summary
    try:
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except Exception:
            pass

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            stage3_run_async(items, OUTDIR, STAGE3_MODEL, MAX_PARALLEL_STAGE3)
        )
    except RuntimeError:
        results = asyncio.run(stage3_run_async(items, OUTDIR, STAGE3_MODEL, MAX_PARALLEL_STAGE3))

    summary_path = Path(OUTDIR) / "summary_results.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    print(f"\n[✓] Finished. OK: {len(ok)}  Failed: {len(bad)}  (details: {summary_path})")

    for i, (inp, cls) in enumerate(inputs_with_class):
        print("\n" + "=" * 80)
        print(f"INPUT: {inp!r}  |  CLASS: {cls}")
        print("- Stage 1 (scaffold) keys:", list(stage1[i].get("high_school", {}).keys()))
        print("- Stage 2 (expanded) keys:", list(stage2[i].keys()))

    # ------------------------- Stage 4: Optional modifications -------------------------
    summary = _load_last_run_summary(summary_path)

    print("\n--- Stage 4: Modify generated slide images (optional) ---")
    print("You can edit slides by giving (slide id) and (image filename), e.g., '1 projectile_motion_cricket.png'.")
    print("Leave slide id empty to finish.\n")

    while True:
        try:
            raw = input("Enter: <slide_id> <image_name> (or blank to exit): ").strip()
        except EOFError:
            break
        if not raw:
            break

        parts = raw.split()
        if len(parts) < 2:
            print("Please provide both slide id and image filename.")
            continue
        try:
            slide_id = int(parts[0])
        except ValueError:
            print("Slide id must be an integer (the numeric id shown in your slides).")
            continue
        image_name = " ".join(parts[1:])

        mod_instructions = input("Describe the modification you want: ").strip()
        if not mod_instructions:
            print("No modification text provided; skipping.")
            continue

        try:
            entry = _find_slide_entry(summary, slide_id, image_name)
        except FileNotFoundError as e:
            print(str(e))
            continue

        image_path = Path(entry["_image_path"])
        workdir = Path(entry["workdir"])
        topic   = entry.get("topic", "unknown topic")
        sname   = entry.get("slide_name", f"slide_{slide_id}")

        print(image_path)
        print(workdir)
        print(f"\n→ Modifying slide {slide_id} «{sname}» with image {image_path.name}")
        try:
            data = _propose_modified_code_from_image(
                model=STAGE4_MODEL,
                image_path=image_path,
                topic=topic,
                slide_id=slide_id,
                slide_name=sname,
                mod_instructions=mod_instructions
            )
        except Exception as e:
            print(f"Model failed to produce valid JSON: {e}")
            continue

        try:
            try:
                loop = asyncio.get_event_loop()
                rc, so, se, images = loop.run_until_complete(_execute_stage4_and_save(data, workdir))
            except RuntimeError:
                rc, so, se, images = asyncio.run(_execute_stage4_and_save(data, workdir))

            ok_exec = (rc == 0)
            print(f"Execution {'succeeded' if ok_exec else 'failed'} (exit={rc}). Saved files: {images}")
            _append_mod_log(Path(OUTDIR), {
                "ts": datetime.now().isoformat(),
                "topic": topic,
                "slide_id": slide_id,
                "slide_name": sname,
                "workdir": str(workdir),
                "source_image": image_path.name,
                "mod_instructions": mod_instructions,
                "model_json": data,
                "run_exit_code": rc,
                "stdout": so[-5000:],
                "stderr": se[-5000:],
                "final_artifacts": images
            })
        except Exception as e:
            print(f"Could not execute modified code: {e}")

    print("\n[Stage 4] done.")
