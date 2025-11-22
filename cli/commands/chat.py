"""Chat command - interactive AI chat"""
from typing import Optional

import typer

from ..utils import check_venv


def chat(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="AI provider to use (anthropic/claude, ollama, purdue)"),
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
        # Default to Claude if API key is available, otherwise use config default
        if provider is None:
            # Check if Anthropic is available (preferred default)
            if "anthropic" in gateway.get_available_providers():
                provider = "anthropic"
                provider_name = "anthropic"
            else:
                # Fallback to config default (likely ollama)
                provider_name = config.provider_name
                provider = provider_name
        else:
            provider_name = provider
        
        # Get provider-specific default model if no model specified
        if model is None:
            model_name = config.get_model_for_provider(provider_name)
        else:
            model_name = model

        # Test connection
        typer.echo("Testing connection...")
        try:
            # Use model_name (provider-specific default) for test
            test_response = gateway.chat("Hello", provider=provider, model=model_name)
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
        typer.echo("Type 'clear' to reset conversation history")
        typer.echo("")

        # Maintain conversation history for stateful chat
        conversation_history = []

        # Interactive chat loop
        while True:
            try:
                user_input = typer.prompt("You", default="").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    typer.echo("Ending chat session. Goodbye!")
                    break

                if user_input.lower() == "clear":
                    conversation_history = []
                    typer.echo("Conversation history cleared.")
                    typer.echo("")
                    continue

                # Add user message to history
                conversation_history.append({"role": "user", "content": user_input})

                typer.echo("AI: ", nl=False)
                try:
                    # Get LLM prompt and format user message
                    from core.prompts import get_prompt
                    llm_template = get_prompt("llm")
                    formatted_message = f"{llm_template}\n\nUser: {user_input}"
                    
                    # Pass full conversation history for stateful chat
                    # Use model_name (provider-specific default) if no model override
                    response = gateway.chat(
                        message=formatted_message,
                        provider=provider,
                        model=model_name if model is None else model,
                        messages=conversation_history
                    )
                    typer.echo(response)
                    
                    # Add AI response to history
                    conversation_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    typer.echo(f"Error: {e}", err=True)
                    typer.echo("Please try again or type 'quit' to exit.", err=True)
                    # Remove the user message from history if there was an error
                    if conversation_history and conversation_history[-1]["role"] == "user":
                        conversation_history.pop()

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

