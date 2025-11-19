#!/usr/bin/env python
import argparse
import re
from pathlib import Path
from typing import List

import shutil
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------- CONFIG ----------

# Max tokens per final file (approximate, using embedding tokenizer)
TOKEN_LIMIT = 450

# Embedding / tokenizer model
MODEL_NAME = "intfloat/multilingual-e5-base"

# Load tokenizer once (same as you use for embeddings)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# tbody open tag anywhere in text, e.g. <tbody_1>, <tbody_1.1>, ...
TBODY_OPEN_RE = re.compile(r"<tbody_([0-9.]+)>", re.IGNORECASE)
# tr open tag, e.g. <tr_1,1,1> , <tr_1.2> ...
TR_OPEN_RE = re.compile(r"<tr_[^>]*>", re.IGNORECASE)
# <a>...</a> tags (allow attributes, multiline content)
A_TAG_RE = re.compile(r"<a\b[^>]*>.*?</a>", re.IGNORECASE | re.DOTALL)


# ---------- Token counting ----------

def count_tokens(text: str) -> int:
    """
    Count tokens using the multilingual-e5-base tokenizer.

    NOTE: we don't add "query:" / "passage:" prefixes here; if you
    will always embed with a prefix, you can optionally add it here.
    """
    # Set add_special_tokens=True so this matches embedding-time length.
    ids = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False
    )["input_ids"]
    return len(ids)


# ---------- Step 1: tbody detection ----------

def find_leaf_tbody_blocks(text: str):
    """
    Find all <tbody_X>...</tbody_X> ranges and return only the *leaf* ones.

    Returns a list of dicts: {"id": str, "start": int, "end": int},
    sorted by "start".

    Leaf = a tbody range that does NOT contain the start of another tbody.
    """
    open_matches = list(TBODY_OPEN_RE.finditer(text))
    if not open_matches:
        return []

    blocks = []
    for m in open_matches:
        tbody_id = m.group(1)  # e.g. "1", "1.1", "1.1.1"
        start = m.start()

        # closing tag: </tbody_X>
        close_pat = re.compile(r"</tbody_" + re.escape(tbody_id) + r">",
                               re.IGNORECASE)
        close_match = close_pat.search(text, m.end())
        if close_match:
            end = close_match.end()
        else:
            end = len(text)

        blocks.append({"id": tbody_id, "start": start, "end": end})

    # keep only leaf blocks (no other block starts inside)
    leaf_blocks = []
    for i, b in enumerate(blocks):
        has_child = any(
            j != i and blocks[j]["start"] > b["start"] and blocks[j]["start"] < b["end"]
            for j in range(len(blocks))
        )
        if not has_child:
            leaf_blocks.append(b)

    leaf_blocks.sort(key=lambda b: b["start"])
    return leaf_blocks


# ---------- Step 2: <tr_...> splitting based on tokens ----------

def split_tr_blocks(inner: str) -> List[str]:
    """
    Split tbody inner content into blocks per <tr_...>.

    Each block runs from its <tr_...> tag up to (but not including)
    the next <tr_...>, or to the end of 'inner'.
    """
    matches = list(TR_OPEN_RE.finditer(inner))
    if not matches:
        return [inner] if inner.strip() else []

    blocks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(inner)
        blocks.append(inner[start:end])
    return blocks


def split_part_on_tr(text: str) -> List[str]:
    """
    Given a *part* file text (first two lines + maybe <a> + one tbody),
    split it further on <tr_...> so each resulting text has at most
    TOKEN_LIMIT tokens if possible, or at least one <tr> per file.

    Always returns *at least one* version (never None).
    """
    # Only handle if there is a tbody
    m_open = TBODY_OPEN_RE.search(text)
    if not m_open:
        return [text]

    total_tokens = count_tokens(text)
    if total_tokens <= TOKEN_LIMIT:
        # Already small enough
        return [text]

    tbody_id = m_open.group(1)
    open_start, open_end = m_open.span()

    # Find matching closing tag </tbody_X>
    close_pat = re.compile(r"</tbody_" + re.escape(tbody_id) + r">",
                           re.IGNORECASE)
    m_close = close_pat.search(text, open_end)
    if not m_close:
        # Malformed: do not try to split
        return [text]

    close_start, close_end = m_close.span()

    prefix = text[:open_start]
    open_tag = text[open_start:open_end]
    inner = text[open_end:close_start]
    close_tag = text[close_start:close_end]
    suffix = text[close_end:]

    tr_blocks = split_tr_blocks(inner)

    # If there are 0 or 1 <tr> blocks, nothing useful to split
    if len(tr_blocks) <= 1:
        return [text]

    # Tokens contributed by everything except the rows themselves
    constant_part = prefix + open_tag + close_tag + suffix
    constant_tokens = count_tokens(constant_part)
    allowed_for_rows = TOKEN_LIMIT - constant_tokens

    chunks: List[List[str]] = []

    if allowed_for_rows <= 0:
        # Even without any <tr>, we already exceed the token limit.
        # We still split by <tr>, but force exactly one <tr> per file.
        for block in tr_blocks:
            chunks.append([block])
    else:
        # Normal case: pack <tr> blocks into chunks under the limit when possible
        current_chunk: List[str] = []
        current_tokens = 0

        for block in tr_blocks:
            block_tokens = count_tokens(block)

            if not current_chunk:
                # Always start a chunk with at least one row, even if large
                current_chunk.append(block)
                current_tokens = block_tokens
                continue

            # If adding this row would exceed the limit and we already have
            # at least one row in the chunk, start a new chunk.
            if current_tokens + block_tokens > allowed_for_rows:
                chunks.append(current_chunk)
                current_chunk = [block]
                current_tokens = block_tokens
            else:
                current_chunk.append(block)
                current_tokens += block_tokens

        if current_chunk:
            chunks.append(current_chunk)

    # If after all this we still only have one chunk, leave as one file
    if len(chunks) <= 1:
        return [text]

    # Build the full file texts for each chunk
    new_texts: List[str] = []
    for chunk in chunks:
        inner_chunk = "".join(chunk)
        new_text = prefix + open_tag + inner_chunk + close_tag + suffix
        new_texts.append(new_text)

    return new_texts


# ---------- Combined processing per source file ----------

def process_file(src_path: Path, dst_dir: Path) -> None:
    """
    For each source .txt file:

    - If no <tbody_...> exists:
        * copy the file unchanged to dst_dir.

    - If there are leaf <tbody_...> blocks:
        * Create <stem>_base.txt with full text but each leaf tbody
          replaced by a placeholder "__TBODY_PART_n__".
        * For each leaf tbody block:
            - Build a part text containing:
                first two lines
                + all <a>...</a> between first two lines and that tbody
                + that tbody block
            - Then apply <tr_...> + token-based splitting
              (<= TOKEN_LIMIT tokens if possible, or at least one <tr> per file).
            - Write resulting files:
                * if only one version: <stem>_part{i}.txt
                * if several versions: <stem>_part{i}_s{j}.txt
    """
    text = src_path.read_text(encoding="utf-8", errors="replace")
    blocks = find_leaf_tbody_blocks(text)

    # If no tbody markers: just copy unchanged
    if not blocks:
        dst_path = dst_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        return

    # Build first two lines and their length
    lines = text.splitlines(keepends=True)
    if len(lines) >= 2:
        first_two = "".join(lines[:2])
    else:
        first_two = "".join(lines)
    first_two_len = len(first_two)

    # ---- Create base file (everything except tbody content) ----
    segments = []
    last = 0
    for idx, b in enumerate(blocks, start=1):
        segments.append(text[last:b["start"]])
        segments.append(f"__TBODY_PART_{idx}__\n")  # placeholder
        last = b["end"]
    segments.append(text[last:])
    base_text = "".join(segments)

    base_name = f"{src_path.stem}_base{src_path.suffix}"
    (dst_dir / base_name).write_text(base_text, encoding="utf-8")

    # ---- Create part files per tbody, then split them on <tr_...> if needed ----
    for idx, b in enumerate(blocks, start=1):
        # Text between first two lines and this tbody
        before_tbody = text[first_two_len:b["start"]]

        # Extract only <a>...</a> segments from that region
        a_segments = A_TAG_RE.findall(before_tbody)
        a_text = "".join(a_segments)

        tbody_text = text[b["start"]:b["end"]]

        part_text = first_two + a_text + tbody_text

        # Now apply the TOKEN_LIMIT + <tr_...> logic
        versions = split_part_on_tr(part_text)

        if len(versions) == 1:
            out_name = f"{src_path.stem}_part{idx}{src_path.suffix}"
            out_path = dst_dir / out_name
            out_path.write_text(versions[0], encoding="utf-8")
        else:
            for j, content in enumerate(versions, start=1):
                out_name = f"{src_path.stem}_part{idx}_s{j}{src_path.suffix}"
                out_path = dst_dir / out_name
                out_path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "1) Split .txt files on leaf <tbody_X>...</tbody_X> blocks, creating:\n"
            "   - one *_base.txt with placeholders __TBODY_PART_n__,\n"
            "   - one or more *_partN*.txt files per tbody, containing\n"
            "     the first two lines + any <a>...</a> before that tbody + the tbody.\n"
            "2) Further split each part on <tr_...> rows using the "
            "intfloat/multilingual-e5-base tokenizer so that each final file\n"
            "   has at most ~450 tokens when possible, or at least one <tr> per file.\n"
            "Files without tbody are copied unchanged to the output folder."
        )
    )
    parser.add_argument("input_dir", help="Folder containing .txt files")
    parser.add_argument("output_dir", help="Folder for output files")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = [p for p in in_dir.glob("*.txt") if p.is_file()]

    for src in tqdm(txt_files, desc="Processing .txt files"):
        process_file(src, out_dir)


if __name__ == "__main__":
    main()