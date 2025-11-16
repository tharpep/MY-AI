"""Config command - display current configuration"""
import typer

from ..utils import check_venv


def config() -> None:
    """Show current configuration settings"""
    if not check_venv():
        raise typer.Exit(1)

    typer.echo("=== Current Configuration ===")
    typer.echo("")

    try:
        from core.config import get_config

        config = get_config()

        typer.echo("=== Primary Configuration ===")
        typer.echo("")
        
        typer.echo("Provider Configuration:")
        typer.echo(f"  Type: {config.provider_type}")
        typer.echo(f"  Name: {config.provider_name}")
        if config.provider_fallback:
            typer.echo(f"  Fallback: {config.provider_fallback}")
        typer.echo("")

        typer.echo("Model Configuration:")
        typer.echo(f"  Default Model: {config.model_default}")
        typer.echo(f"  Active Model: {config.model_name}")
        typer.echo("")

        typer.echo("Ollama Configuration:")
        typer.echo(f"  Base URL: {config.ollama_base_url}")
        typer.echo(f"  Timeout: {config.ollama_timeout}s")
        typer.echo("")

        typer.echo("RAG Configuration:")
        typer.echo(f"  Storage: {'Persistent' if config.rag_use_persistent else 'In-memory'}")
        typer.echo(f"  Collection: {config.rag_collection_name}")
        typer.echo(f"  Top-K: {config.rag_top_k}")
        typer.echo(f"  Similarity Threshold: {config.rag_similarity_threshold}")
        typer.echo(f"  Max Tokens: {config.rag_max_tokens}")
        typer.echo(f"  Temperature: {config.rag_temperature}")
        typer.echo("")

        typer.echo("Tuning Configuration:")
        typer.echo(f"  Device: {config.tuning_device}")
        typer.echo(f"  Batch Size: {config.tuning_batch_size}")
        typer.echo(f"  Epochs: {config.tuning_num_epochs}")
        typer.echo(f"  Learning Rate: {config.tuning_learning_rate}")
        typer.echo(f"  Output Dir: {config.output_dir}")
        typer.echo("")

        typer.echo("API Keys:")
        typer.echo(f"  Purdue: {'Set' if config.purdue_api_key else 'Not set'}")
        typer.echo(f"  OpenAI: {'Set' if config.openai_api_key else 'Not set'}")
        typer.echo(f"  Anthropic: {'Set' if config.anthropic_api_key else 'Not set'}")
        typer.echo("")

        typer.echo("Note: Override settings via .env file or environment variables")
        typer.echo("Example: PROVIDER_TYPE=local, PROVIDER_NAME=ollama, MODEL_DEFAULT=qwen3:8b")

    except ImportError as e:
        typer.echo(f"Error: Could not import config module: {e}", err=True)
        raise typer.Exit(1)

    raise typer.Exit(0)

