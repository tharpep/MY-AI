"""Chat command - interactive AI chat"""
from typing import Optional

import typer

from ..utils import check_venv


def chat(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="AI provider to use (ollama, purdue)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
) -> None:
    """Interactive chat with the AI - type your questions and get responses"""
    if not check_venv():
        raise typer.Exit(1)

    try:
        from llm.gateway import AIGateway
        from core.config import get_config

        typer.echo("Initializing AI Gateway...")
        gateway = AIGateway()
        config = get_config()

        # Determine provider and model info
        provider_name = provider or config.provider_name
        model_name = model or config.model_name

        # Test connection
        typer.echo("Testing connection...")
        try:
            test_response = gateway.chat("Hello", provider=provider, model=model)
            typer.echo(f"Connection successful!")
        except Exception as e:
            typer.echo(f"Connection test failed: {e}", err=True)
            typer.echo("Please check your configuration and ensure the AI service is running.", err=True)
            raise typer.Exit(1)

        # Show connection info
        typer.echo("")
        typer.echo("=== AI Chat Session ===")
        typer.echo(f"Provider: {provider_name}")
        typer.echo(f"Model: {model_name}")
        typer.echo("Type 'quit', 'exit', or 'q' to end the session")
        typer.echo("")

        # Interactive chat loop
        while True:
            try:
                user_input = typer.prompt("You", default="").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    typer.echo("Ending chat session. Goodbye!")
                    break

                typer.echo("AI: ", nl=False)
                try:
                    response = gateway.chat(user_input, provider=provider, model=model)
                    typer.echo(response)
                except Exception as e:
                    typer.echo(f"Error: {e}", err=True)
                    typer.echo("Please try again or type 'quit' to exit.", err=True)

                typer.echo("")

            except KeyboardInterrupt:
                typer.echo("\nEnding chat session. Goodbye!")
                break
            except EOFError:
                typer.echo("\nEnding chat session. Goodbye!")
                break

        # Cleanup gateway if needed (currently no explicit cleanup required)
        # Gateway resources are automatically cleaned up by Python GC
        raise typer.Exit(0)

    except ImportError as e:
        typer.echo(f"Error: Could not import required modules: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

