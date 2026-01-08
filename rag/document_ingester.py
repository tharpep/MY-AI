"""Document Ingestion System"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Union
from .rag_setup import ContextEngine


class DocumentIngester:
    """Handles document ingestion and preprocessing"""
    
    def __init__(self, rag_system: ContextEngine):
        """Initialize document ingester."""
        self.rag = rag_system
        self.supported_extensions = ['.txt', '.md']
    
    def ingest_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Ingest a single file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        if file_path.suffix.lower() not in self.supported_extensions:
            return {"error": f"Unsupported file type: {file_path.suffix}"}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            from core.config import get_config
            config = get_config()
            
            is_markdown = file_path.suffix.lower() == '.md'
            document_type = "markdown" if is_markdown else "text"
            
            if is_markdown:
                from rag.chunking import chunk_markdown
                chunk_results = chunk_markdown(
                    content,
                    chunk_size=config.library_chunk_size,
                    overlap=config.library_chunk_overlap
                )
                
                total_indexed = 0
                for text, section_title in chunk_results:
                    metadata = {
                        "document_type": document_type,
                        "source_file": str(file_path.name),
                        "section_title": section_title if section_title else None
                    }
                    total_indexed += self.rag.add_documents([text], metadata=metadata)
                
                return {
                    "success": True,
                    "file": str(file_path),
                    "chunks": len(chunk_results),
                    "indexed": total_indexed
                }
            else:
                processed_content = self._preprocess_text(content)
                from rag.chunking import chunk_text
                chunks = chunk_text(
                    processed_content, 
                    chunk_size=config.library_chunk_size,
                    overlap=config.library_chunk_overlap
                )
                
                metadata = {
                    "document_type": document_type,
                    "source_file": str(file_path.name)
                }
                count = self.rag.add_documents(chunks, metadata=metadata)
                
                return {
                    "success": True,
                    "file": str(file_path),
                    "chunks": len(chunks),
                    "indexed": count
                }
            
        except Exception as e:
            return {"error": f"Failed to process {file_path}: {str(e)}"}
    
    def ingest_folder(self, folder_path: Union[str, Path]) -> Dict[str, Any]:
        """Ingest all supported files in a folder."""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            return {"error": f"Folder not found: {folder_path}"}
        
        results = {
            "success": True,
            "processed": 0,
            "failed": 0,
            "files": [],
            "errors": []
        }
        
        for ext in self.supported_extensions:
            pattern = str(folder_path / f"**/*{ext}")
            files = glob.glob(pattern, recursive=True)
            
            for file_path in files:
                result = self.ingest_file(file_path)
                
                if result.get("success"):
                    results["processed"] += 1
                    results["files"].append({
                        "file": result["file"],
                        "chunks": result["chunks"],
                        "indexed": result["indexed"]
                    })
                else:
                    results["failed"] += 1
                    results["errors"].append(result["error"])
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        text = ' '.join(text.split())
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def get_supported_files(self, folder_path: Union[str, Path]) -> List[str]:
        """Get list of supported files in a folder."""
        folder_path = Path(folder_path)
        files = []
        
        for ext in self.supported_extensions:
            pattern = str(folder_path / f"**/*{ext}")
            files.extend(glob.glob(pattern, recursive=True))
        
        return files


def main():
    """Demo of document ingestion"""
    from core.config import get_config
    
    print("=== Document Ingestion Demo ===\n")
    
    print("1. Initializing RAG with persistent storage...")
    rag = ContextEngine(use_persistent=True)
    
    ingester = DocumentIngester(rag)
    
    config = get_config()
    documents_folder = config.library_documents_folder
    print(f"Using documents folder: {documents_folder}")
    
    supported_files = ingester.get_supported_files(documents_folder)
    
    print(f"2. Found {len(supported_files)} supported files:")
    for file_path in supported_files:
        print(f"   - {file_path}")
    
    if supported_files:
        print(f"\n3. Ingesting files from {documents_folder}...")
        result = ingester.ingest_folder(documents_folder)
        
        if result["success"]:
            print(f"   Processed: {result['processed']} files")
            print(f"   Failed: {result['failed']} files")
            print(f"   Total chunks indexed: {sum(f['chunks'] for f in result['files'])}")
            
            if result["errors"]:
                print(f"   Errors: {result['errors']}")
        else:
            print(f"   Error: {result['error']}")
    else:
        print(f"\n3. No supported files found in {documents_folder}")
        print("   Add some .txt or .md files to test ingestion")
    
    stats = rag.get_stats()
    print(f"\n4. Final RAG stats: {stats}")


if __name__ == "__main__":
    main()
