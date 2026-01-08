"""Shared text chunking utilities for RAG ingestion."""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
    prefer_paragraph_breaks: bool = True
) -> List[str]:
    """Split text into overlapping chunks, preferring natural break points."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            if prefer_paragraph_breaks:
                para_break = text.rfind("\n\n", start + chunk_size // 2, end)
                if para_break > start:
                    end = para_break + 2
                else:
                    end = _find_sentence_break(text, start, end)
            else:
                end = _find_sentence_break(text, start, end)
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def _find_sentence_break(text: str, start: int, end: int) -> int:
    """Find the best sentence break point within the range."""
    separators = [". ", ".\n", "? ", "?\n", "! ", "!\n"]
    
    best_break = end
    min_pos = start + (end - start) // 2
    
    for sep in separators:
        pos = text.rfind(sep, min_pos, end)
        if pos > min_pos:
            candidate = pos + len(sep)
            if candidate > min_pos:
                best_break = candidate
                break
    
    return best_break


def chunk_conversation(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 150
) -> List[str]:
    """Chunk conversation text, optimized for dialogue format."""
    return chunk_text(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap,
        prefer_paragraph_breaks=True
    )
