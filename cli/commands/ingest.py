"""Ingest command - ingest documents into RAG system"""
from typing import Optional
from pathlib import Path

import typer

from ..utils import check_venv


def ingest(
    folder_path: Optional[str] = typer.Option(None, "--folder", "-f", help="Path to folder containing documents (uses config default if not provided)"),
) -> None:
    """Ingest documents from a folder into the RAG system"""
    if not check_venv():
        raise typer.Exit(1)

    try:
        from rag.rag_setup import BasicRAG
        from rag.document_ingester import DocumentIngester
        from core.config import get_config

        config = get_config()
        
        # Use config folder if not provided
        if folder_path is None:
            folder_path = config.rag_documents_folder
        
        # Validate folder path
        folder = Path(folder_path)
        if not folder.exists():
            typer.echo(f"Error: Folder not found: {folder_path}", err=True)
            raise typer.Exit(1)
        
        if not folder.is_dir():
            typer.echo(f"Error: Path is not a directory: {folder_path}", err=True)
            raise typer.Exit(1)
        
        typer.echo("Initializing RAG system...")
        rag = BasicRAG()
        typer.echo("RAG system ready!")
        typer.echo("")
        
        # Initialize ingester
        ingester = DocumentIngester(rag)
        
        # Check what files are available
        typer.echo(f"Scanning folder: {folder_path}")
        supported_files = ingester.get_supported_files(folder_path)
        
        if not supported_files:
            typer.echo(f"No supported files found in {folder_path}")
            typer.echo("Supported formats: .txt, .md")
            raise typer.Exit(1)
        
        typer.echo(f"Found {len(supported_files)} supported files:")
        for file_path in supported_files[:10]:  # Show first 10
            typer.echo(f"  - {file_path}")
        if len(supported_files) > 10:
            typer.echo(f"  ... and {len(supported_files) - 10} more")
        typer.echo("")
        
        # Ingest documents
        typer.echo(f"Ingesting files from {folder_path}...")
        result = ingester.ingest_folder(folder_path)
        
        if result["success"]:
            typer.echo("")
            typer.echo("=== Ingestion Results ===")
            typer.echo(f"✅ Processed: {result['processed']} files")
            typer.echo(f"❌ Failed: {result['failed']} files")
            typer.echo(f"📄 Total chunks indexed: {sum(f.get('chunks', 0) for f in result.get('files', []))}")
            
            if result.get("errors"):
                typer.echo("")
                typer.echo("Errors encountered:")
                for error in result["errors"]:
                    typer.echo(f"  - {error}")
            
            # Show final stats
            typer.echo("")
            typer.echo("=== RAG Statistics ===")
            stats = rag.get_stats()
            for key, value in stats.items():
                typer.echo(f"  {key}: {value}")
        else:
            typer.echo(f"❌ Ingestion failed: {result.get('error', 'Unknown error')}", err=True)
            raise typer.Exit(1)
        
        raise typer.Exit(0)
        
    except ImportError as e:
        typer.echo(f"Error: Could not import required modules: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

