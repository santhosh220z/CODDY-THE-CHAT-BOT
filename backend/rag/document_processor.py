import os
import re
import pandas as pd
from typing import List, Dict, Any
from pypdf import PdfReader
import pdfplumber
import docx
from backend.utils.logging import logger
from backend.utils.config import settings

class DocumentProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        # Normalize whitespace (replace multiple spaces/newlines with single ones)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\r\n|\r', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
        pages = []
        # Try pdfplumber first for better tabular/layout extraction
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        pages.append({
                            "text": DocumentProcessor.clean_text(text),
                            "page_number": i + 1
                        })
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed, falling back to pypdf: {e}")
            # Fallback to pypdf
            try:
                reader = PdfReader(file_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        pages.append({
                            "text": DocumentProcessor.clean_text(text),
                            "page_number": i + 1
                        })
            except Exception as ex:
                logger.error(f"pypdf extraction failed as well: {ex}")
                raise ex
        return pages

    @staticmethod
    def extract_text_from_docx(file_path: str) -> List[Dict[str, Any]]:
        pages = []
        try:
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            
            cleaned_text = DocumentProcessor.clean_text("\n".join(full_text))
            # DOCX doesn't have native page numbers easily extractable via python-docx,
            # so we treat the whole document as a single "page" (or segment) or split by sections.
            pages.append({
                "text": cleaned_text,
                "page_number": 1
            })
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise e
        return pages

    @staticmethod
    def extract_text_from_csv(file_path: str) -> List[Dict[str, Any]]:
        pages = []
        try:
            df = pd.read_csv(file_path)
            # Convert tabular data to readable text format
            # Option: Row by row description
            rows_text = []
            for idx, row in df.iterrows():
                row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
                rows_text.append(f"Row {idx + 1}: {row_str}")
            
            full_text = "\n".join(rows_text)
            pages.append({
                "text": full_text,
                "page_number": 1
            })
        except Exception as e:
            logger.error(f"CSV extraction failed: {e}")
            raise e
        return pages

    @staticmethod
    def extract_text_from_md(file_path: str) -> List[Dict[str, Any]]:
        pages = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            cleaned_text = DocumentProcessor.clean_text(text)
            pages.append({
                "text": cleaned_text,
                "page_number": 1
            })
        except Exception as e:
            logger.error(f"Markdown extraction failed: {e}")
            raise e
        return pages

    @staticmethod
    def extract_text_from_txt(file_path: str) -> List[Dict[str, Any]]:
        pages = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            cleaned_text = DocumentProcessor.clean_text(text)
            pages.append({
                "text": cleaned_text,
                "page_number": 1
            })
        except Exception as e:
            logger.error(f"TXT extraction failed: {e}")
            raise e
        return pages

    @staticmethod
    def split_text_recursive(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Splits text recursively by trying common boundaries: paragraphs, sentences, words."""
        if len(text) <= chunk_size:
            return [text]

        separators = ["\n\n", "\n", " ", ""]
        final_chunks = []
        
        # Simple recursive character splitting logic
        def _split(text_to_split: str, current_seps: List[str]) -> List[str]:
            if len(text_to_split) <= chunk_size:
                return [text_to_split]
            
            if not current_seps:
                # Fallback: hard slice
                return [text_to_split[i:i+chunk_size] for i in range(0, len(text_to_split), chunk_size - chunk_overlap)]
            
            sep = current_seps[0]
            parts = text_to_split.split(sep)
            chunks = []
            current_chunk = ""
            
            for part in parts:
                # Re-add separator if it isn't empty string
                candidate = (current_chunk + sep + part) if current_chunk else part
                
                if len(candidate) <= chunk_size:
                    current_chunk = candidate
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # If single part is larger than chunk size, split it further with remaining separators
                    if len(part) > chunk_size:
                        sub_chunks = _split(part, current_seps[1:])
                        # Merge overlap if possible
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = part
            
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks

        raw_chunks = _split(text, separators)
        
        # Merge chunks with overlap support
        merged_chunks = []
        for rc in raw_chunks:
            if not rc.strip():
                continue
            if not merged_chunks:
                merged_chunks.append(rc)
            else:
                last_chunk = merged_chunks[-1]
                # If they can fit together with overlap, or if we want to create overlapping pieces
                # For simplicity, we just keep the split items as chunks, ensuring they are roughly within size
                # Let's ensure proper overlap. If chunk is small, merge it.
                merged_chunks.append(rc)
                
        # Re-verify and apply overlap manually if simple split didn't capture overlap
        # To guarantee exact chunking alignment:
        guaranteed_chunks = []
        for i, chunk in enumerate(merged_chunks):
            # If it's the first chunk, just append
            if i == 0:
                guaranteed_chunks.append(chunk)
                continue
            
            # Prepend overlap from previous chunk
            prev_chunk = merged_chunks[i-1]
            overlap_text = prev_chunk[-chunk_overlap:] if len(prev_chunk) >= chunk_overlap else prev_chunk
            combined = overlap_text + "\n" + chunk
            
            if len(combined) <= chunk_size:
                guaranteed_chunks.append(combined)
            else:
                guaranteed_chunks.append(chunk)
                
        return guaranteed_chunks

    @classmethod
    def process_document(cls, file_path: str, file_name: str, uploaded_by_email: str) -> List[Dict[str, Any]]:
        ext = os.path.splitext(file_name)[1].lower()
        logger.info(f"Processing document {file_name} (ext: {ext})...")
        
        if ext == ".pdf":
            pages = cls.extract_text_from_pdf(file_path)
        elif ext == ".docx":
            pages = cls.extract_text_from_docx(file_path)
        elif ext == ".csv":
            pages = cls.extract_text_from_csv(file_path)
        elif ext in [".md", ".markdown"]:
            pages = cls.extract_text_from_md(file_path)
        elif ext == ".txt":
            pages = cls.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        processed_chunks = []
        chunk_id_counter = 0
        
        chunk_size = settings.CHUNK_SIZE
        chunk_overlap = settings.CHUNK_OVERLAP
        
        for p in pages:
            text = p["text"]
            page_num = p["page_number"]
            
            # Split page text into chunks
            chunks = cls.split_text_recursive(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            for chunk_text in chunks:
                chunk_id_counter += 1
                chunk_meta = {
                    "file_name": file_name,
                    "page_number": page_num,
                    "chunk_id": f"{file_name}_chunk_{chunk_id_counter}",
                    "source": file_name,
                    "uploaded_by": uploaded_by_email
                }
                
                processed_chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_meta,
                    "chunk_index": chunk_id_counter
                })
                
        logger.info(f"Document {file_name} split into {len(processed_chunks)} chunks.")
        return processed_chunks
