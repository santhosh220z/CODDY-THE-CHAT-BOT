from __future__ import annotations

import csv
import io
import os
import re
from typing import Any, Dict, List

from backend.utils.config import settings
from backend.utils.logging import logger


class DocumentProcessingError(ValueError):
    """The file could not be turned into indexable text."""


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split into chunks of at most `chunk_size` with `chunk_overlap` characters
    of overlap between neighbours.

    Boundaries are tried in order: blank line, newline, sentence end,
    whitespace. A unit too large for one chunk (minified JSON, an unbroken
    token) is hard-sliced while still preserving overlap.

    The overlap is real: each new chunk restarts with the tail of the previous
    one, so a sentence spanning a boundary stays retrievable.
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
    # Keep separators so the concatenation of units reproduces the input.
    units = [u for u in re.split(r"(\n{2,}|\n|(?<=[.!?])\s+|\s+)", text) if u]

    chunks: List[str] = []
    current = ""

    for unit in units:
        candidate = current + unit
        if len(candidate) <= chunk_size or not current.strip():
            current = candidate
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - chunk_overlap:] if chunk_overlap else ""
            continue

        chunks.append(current)
        # Restart from the tail of what we just emitted.
        tail = current[-chunk_overlap:] if chunk_overlap else ""
        current = (tail + unit) if len(tail) + len(unit) <= chunk_size else unit

    if current.strip():
        chunks.append(current)

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Extractors - each returns [{"text": str, "page_number": int}, ...]
# ---------------------------------------------------------------------------
def _extract_pdf(path: str) -> List[Dict[str, Any]]:
    from pypdf import PdfReader

    pages: List[Dict[str, Any]] = []
    with open(path, "rb") as fh:
        reader = PdfReader(fh)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise DocumentProcessingError(
                    "This PDF is password protected and cannot be read."
                ) from exc

        for index, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception as exc:
                logger.warning(f"Page {index} of {path} failed to extract: {exc}")
                continue
            text = clean_text(raw)
            if text:
                pages.append({"text": text, "page_number": index})

    if not pages:
        raise DocumentProcessingError(
            "No selectable text found. This looks like a scanned PDF - it needs OCR."
        )
    return pages


def _extract_docx(path: str) -> List[Dict[str, Any]]:
    import docx

    try:
        document = docx.Document(path)
    except Exception as exc:
        raise DocumentProcessingError(f"Could not read the DOCX file: {exc}") from exc

    blocks: List[str] = []
    for para in document.paragraphs:
        if para.text.strip():
            blocks.append(para.text)

    # python-docx does not surface text inside tables via .paragraphs, and
    # tabular content is often the point of the document.
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))

    text = clean_text("\n\n".join(blocks))
    if not text:
        raise DocumentProcessingError("The DOCX file contains no text.")
    # DOCX has no reliable page boundaries without rendering it.
    return [{"text": text, "page_number": 1}]


def _extract_csv(path: str) -> List[Dict[str, Any]]:
    # Read as utf-8 first, then fall back to latin-1 which never raises, so a
    # stray byte does not fail an otherwise valid upload.
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=encoding, newline="") as fh:
                rows = list(csv.reader(fh))
            break
        except UnicodeDecodeError:
            continue
    else:
        raise DocumentProcessingError("Could not decode the CSV file.")

    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        raise DocumentProcessingError("The CSV file is empty.")

    header, body = rows[0], rows[1:]
    # Repeat the header on every record so each chunk is self-describing when
    # retrieved in isolation.
    lines = ["Columns: " + " | ".join(h.strip() for h in header)]
    for index, row in enumerate(body, start=1):
        cells = [cell.strip() for cell in row]
        lines.append(f"Row {index}: " + " | ".join(cells))

    return [{"text": clean_text("\n".join(lines)), "page_number": 1}]


def _extract_plain(path: str) -> List[Dict[str, Any]]:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=encoding, errors="strict") as fh:
                text = clean_text(fh.read())
            break
        except UnicodeDecodeError:
            continue
    else:
        raise DocumentProcessingError("Could not decode the file as text.")

    if not text:
        raise DocumentProcessingError("The file contains no readable text.")
    return [{"text": text, "page_number": 1}]


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".csv": _extract_csv,
    ".txt": _extract_plain,
    ".md": _extract_plain,
    ".markdown": _extract_plain,
}


def process_document(
    path: str,
    file_name: str,
    owner_id: int,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Dict[str, Any]]:
    """Extract, chunk, and stamp tenant metadata onto every chunk."""
    ext = os.path.splitext(file_name)[1].lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise DocumentProcessingError(f"Unsupported file type '{ext}'.")

    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    pages = extractor(path)

    chunks: List[Dict[str, Any]] = []
    for page in pages:
        for text in split_text(page["text"], chunk_size, chunk_overlap):
            index = len(chunks) + 1
            chunks.append(
                {
                    "chunk_index": index,
                    "text": text,
                    "metadata": {
                        "file_name": file_name,
                        "page_number": page["page_number"],
                        # Tenant scope. The vector store filters on this, so one
                        # user's chat can never return another user's chunks.
                        "user_id": int(owner_id),
                    },
                }
            )

    if not chunks:
        raise DocumentProcessingError("No indexable text could be extracted.")

    logger.info(f"Indexed {file_name}: {len(pages)} page(s) -> {len(chunks)} chunks")
    return chunks