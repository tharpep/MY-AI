"""Query command - RAG-powered question answering"""
from typing import Optional

import typer

from ..utils import check_venv


def query(
    question: Optional[str] = typer.Argument(None, help="Question to ask (optional, will prompt if not provided)"),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Number of documents to retrieve"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Similarity threshold (0.0-1.0)"),
) -> None:
    """Query the RAG system with a question - uses retrieved context to answer"""
    if not check_venv():
        raise typer.Exit(1)

    try:
        from rag.rag_setup import BasicRAG
        from core.config import get_config

        config = get_config()
        
        typer.echo("Initializing RAG system...")
        rag = BasicRAG()
        typer.echo("RAG system ready!")
        typer.echo("")
        
        # Interactive mode if no question provided
        if question is None:
            typer.echo("=== RAG Query Mode ===")
            typer.echo("Type 'quit', 'exit', or 'q' to exit")
            typer.echo("")
            
            while True:
                question = typer.prompt("Question", default="").strip()
                
                if not question:
                    continue
                
                if question.lower() in ["quit", "exit", "q"]:
                    typer.echo("Exiting query mode.")
                    break
                
                # Use config defaults if not specified
                context_limit = top_k if top_k is not None else config.rag_top_k
                
                typer.echo(f"\nQuerying: {question}")
                typer.echo(f"Retrieving top {context_limit} documents...")
                typer.echo("")
                
                # Query with RAG
                answer, context_docs, context_scores = rag.query(
                    question=question,
                    context_limit=context_limit
                )
                
                # Display answer
                typer.echo("=== Answer ===")
                typer.echo(answer)
                typer.echo("")
                
                # Display context if logging enabled
                if config.log_output and context_docs:
                    typer.echo(f"=== Retrieved Context ({len(context_docs)} documents) ===")
                    for i, (doc, score) in enumerate(zip(context_docs, context_scores), 1):
                        doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
                        typer.echo(f"\n[{i}] Score: {score:.3f}")
                        typer.echo(f"    {doc_preview}")
                
                typer.echo("")
        else:
            # Single query mode
            # Use config defaults if not specified
            context_limit = top_k if top_k is not None else config.rag_top_k
            
            typer.echo(f"Querying: {question}")
            typer.echo(f"Retrieving top {context_limit} documents...")
            typer.echo("")
            
            # Query with RAG
            answer, context_docs, context_scores = rag.query(
                question=question,
                context_limit=context_limit
            )
            
            # Display answer
            typer.echo("=== Answer ===")
            typer.echo(answer)
            typer.echo("")
            
            # Display context if logging enabled
            if config.log_output and context_docs:
                typer.echo(f"=== Retrieved Context ({len(context_docs)} documents) ===")
                for i, (doc, score) in enumerate(zip(context_docs, context_scores), 1):
                    doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
                    typer.echo(f"\n[{i}] Score: {score:.3f}")
                    typer.echo(f"    {doc_preview}")
        
        raise typer.Exit(0)
        
    except ImportError as e:
        typer.echo(f"Error: Could not import required modules: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

