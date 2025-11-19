#!/usr/bin/env python
import argparse
import os
import re
from pathlib import Path
from typing import List

import shutil
from tqdm import tqdm
from transformers import AutoTokenizer

# ------------- CONFIG -------------

TOKEN_LIMIT = 450
MIN_CHUNK_TOKENS = 100  # to avoid one tiny chunk + one huge chunk

MODEL_NAME = "intfloat/multilingual-e5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# <a>...</a> tags (allow attributes, multiline content)
A_TAG_RE = re.compile(r"<a\b[^>]*>.*?</a>", re.IGNORECASE | re.DOTALL)


# ------------- TOKEN COUNT -------------

def count_tokens(text: str) -> int:
    """Count tokens using the multilingual-e5-base tokenizer."""
    ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    return len(ids)


# ------------- SPLITTING LOGIC -------------

def split_by_a_tags(text: str) -> List[str]:
    """
    Split a text into multiple smaller texts based on <a>...</a> tags.

    - Always keeps the first two lines as a header (repeated in each output).
    - Splits the remainder into segments anchored at <a> tags.
    - Groups segments into chunks so that header+chunk <= TOKEN_LIMIT tokens
      where possible, or at least one segment per chunk.
    - If splitting would create only two chunks and one is very tiny
      (< MIN_CHUNK_TOKENS including header), returns [text] (no split).

    Returns a list of one or more texts.
    """
    total_tokens = count_tokens(text)
    if total_tokens <= TOKEN_LIMIT:
        return [text]

    # First two lines become the header
    lines = text.splitlines(keepends=True)
    if len(lines) >= 2:
        header = "".join(lines[:2])
        body = "".join(lines[2:])
    else:
        header = "".join(lines)
        body = ""
    header_tokens = count_tokens(header)

    # If header alone already exceeds the limit, splitting by <a> won't help
    if header_tokens >= TOKEN_LIMIT or not body.strip():
        return [text]

    matches = list(A_TAG_RE.finditer(body))
    if len(matches) <= 1:
        # Need at least two <a> tags to split meaningfully
        return [text]

    # Build segments: each contains one <a> block plus the following text
    segments: List[str] = []
    for i, m in enumerate(matches):
        if i == 0:
            start = 0
        else:
            start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        segments.append(body[start:end])

    # Token budget for body segments in each chunk
    allowed_for_segments = TOKEN_LIMIT - header_tokens
    if allowed_for_segments <= 0:
        return [text]

    # Greedy grouping of segments into chunks
    chunks: List[List[str]] = []
    current_chunk: List[str] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = count_tokens(seg)

        if not current_chunk:
            current_chunk.append(seg)
            current_tokens = seg_tokens
            continue

        if current_tokens + seg_tokens > allowed_for_segments:
            # Start a new chunk
            chunks.append(current_chunk)
            current_chunk = [seg]
            current_tokens = seg_tokens
        else:
            current_chunk.append(seg)
            current_tokens += seg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    # If only one chunk, splitting didn't help
    if len(chunks) <= 1:
        return [text]

    # Check for the "one tiny chunk" case (only when there are exactly two)
    if len(chunks) == 2:
        chunk_tokens = []
        for chunk in chunks:
            chunk_body = "".join(chunk)
            chunk_tokens.append(count_tokens(header + chunk_body))

        if min(chunk_tokens) < MIN_CHUNK_TOKENS:
            # Splitting would create a nearly empty file; skip splitting
            return [text]

    # Build new texts
    new_texts: List[str] = []
    for chunk in chunks:
        chunk_body = "".join(chunk)
        new_text = header + chunk_body
        new_texts.append(new_text)

    return new_texts


def process_file(src_path: Path, out_dir: Path) -> None:
    """Process a single .txt file with the <a>-based splitting."""
    text = src_path.read_text(encoding="utf-8", errors="replace")

    versions = split_by_a_tags(text)

    # If we got only one version and it's the same as original, just copy
    if len(versions) == 1 and versions[0] == text:
        dst_path = out_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        return

    # Otherwise, write split versions
    stem = src_path.stem
    suffix = src_path.suffix

    if len(versions) == 1:
        out_name = f"{stem}_a1{suffix}"
        (out_dir / out_name).write_text(versions[0], encoding="utf-8")
    else:
        for idx, content in enumerate(versions, start=1):
            out_name = f"{stem}_a{idx}{suffix}"
            (out_dir / out_name).write_text(content, encoding="utf-8")


# ------------- MAIN -------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Second-stage splitter:\n"
            "- Reads .txt files from an input folder (e.g. output of previous step).\n"
            "- Files with <= 450 tokens are copied unchanged to output.\n"
            "- Larger files are split based on <a>...</a> tags:\n"
            "    * first two lines kept in every file,\n"
            "    * segments grouped so each result is <= 450 tokens when possible,\n"
            "    * avoids creating one tiny file + one huge file.\n"
            "- All results (copies + split files) are written to the output folder."
        )
    )
    parser.add_argument("input_dir", help="Folder containing .txt files to split")
    parser.add_argument("output_dir", help="Folder to write new/split files")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = [p for p in in_dir.glob("*.txt") if p.is_file()]

    for src in tqdm(txt_files, desc="Splitting large files by <a> tags"):
        process_file(src, out_dir)


if __name__ == "__main__":
    main()