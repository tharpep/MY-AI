"""
Document Ingestion System
Handles file processing and indexing for RAG system
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Union
from .rag_setup import BasicRAG


class DocumentIngester:
    """Handles document ingestion and preprocessing"""
    
    def __init__(self, rag_system: BasicRAG):
        """
        Initialize document ingester
        
        Args:
            rag_system: RAG system instance to index documents into
        """
        self.rag = rag_system
        self.supported_extensions = ['.txt', '.md']
    
    def ingest_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Ingest a single file
        
        Args:
            file_path: Path to the file to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        if file_path.suffix.lower() not in self.supported_extensions:
            return {"error": f"Unsupported file type: {file_path.suffix}"}
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic preprocessing
            processed_content = self._preprocess_text(content)
            
            # Chunk long documents using config values
            from core.config import get_config
            config = get_config()
            chunks = self._chunk_text(
                processed_content, 
                max_chunk_size=config.rag_chunk_size,
                overlap=config.rag_chunk_overlap
            )
            
            # Index chunks
            count = self.rag.add_documents(chunks)
            
            return {
                "success": True,
                "file": str(file_path),
                "chunks": len(chunks),
                "indexed": count
            }
            
        except Exception as e:
            return {"error": f"Failed to process {file_path}: {str(e)}"}
    
    def ingest_folder(self, folder_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Ingest all supported files in a folder
        
        Args:
            folder_path: Path to folder containing documents
            
        Returns:
            Dictionary with ingestion results
        """
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
        
        # Find all supported files
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
        """
        Basic text preprocessing
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        return '\n'.join(lines)
    
    def _chunk_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + max_chunk_size // 2, end - 200), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def get_supported_files(self, folder_path: Union[str, Path]) -> List[str]:
        """
        Get list of supported files in a folder
        
        Args:
            folder_path: Path to folder
            
        Returns:
            List of file paths
        """
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
    
    # Initialize RAG with persistent storage
    print("1. Initializing RAG with persistent storage...")
    rag = BasicRAG(use_persistent=True)
    
    # Initialize ingester
    ingester = DocumentIngester(rag)
    
    # Get folder from config
    config = get_config()
    documents_folder = config.rag_documents_folder
    print(f"Using documents folder: {documents_folder}")
    
    # Check what files are available
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
    
    # Show final stats
    stats = rag.get_stats()
    print(f"\n4. Final RAG stats: {stats}")


if __name__ == "__main__":
    main()
