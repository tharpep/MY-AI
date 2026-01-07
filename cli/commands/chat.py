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

        # Initialize RAG if enabled (do this at startup to avoid delays during chat)
        rag_instance = None
        if config.chat_context_enabled and config.chat_library_enabled:
            try:
                typer.echo("Initializing RAG system...")
                typer.echo("Initializing RAG system...")
                from rag.rag_setup import get_rag
                rag_instance = get_rag()
                typer.echo("RAG system ready!")
            except Exception as e:
                typer.echo(f"Warning: RAG initialization failed: {e}", err=True)
                typer.echo("Continuing without RAG support.", err=True)
                rag_instance = None

        # Show connection info
        typer.echo("")
        typer.echo("=== AI Chat Session ===")
        typer.echo(f"Provider: {provider_name}")
        typer.echo(f"Model: {model_name}")
        typer.echo(f"RAG: {'Enabled' if rag_instance else 'Disabled'}")
        typer.echo("Type 'quit', 'exit', or 'q' to end the session")
        typer.echo("Type 'clear' to reset conversation history")
        typer.echo("")

        # Maintain conversation history for stateful chat
        conversation_history = []
        
        # Create ChatService once and reuse (for cache persistence)
        from core.services import ChatService
        chat_service = ChatService(config, context_engine=rag_instance)

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

                typer.echo("AI: ", nl=False)
                try:
                    import time
                    from datetime import datetime
                    
                    # Start total timing
                    total_start = time.time()
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    
                    if config.log_output:
                        typer.echo(f"[{timestamp}] Starting chat request...", err=True)
                    
                    # Use ChatService to prepare message with context
                    prep_start = time.time()
                    message_result = chat_service.prepare_chat_message(
                        user_message=user_input,
                        use_library=None,  # Use config default
                        use_journal=None,  # Use config default
                        library_top_k=None,  # Use config default
                        journal_top_k=None,  # Use config default
                        similarity_threshold=None,  # Use config default
                        system_prompt=None,  # Use default
                        context_prompt_template=None  # Use default
                    )
                    prep_time = (time.time() - prep_start) * 1000
                    
                    # Log RAG results if logging enabled
                    if config.log_output:
                        if message_result.library_results:
                            typer.echo(f"[Library: Retrieved {len(message_result.library_results)} docs]", err=True)
                        else:
                            typer.echo(f"[RAG: No docs found]", err=True)
                        typer.echo(f"[Message Prep: {prep_time:.1f}ms]", err=True)
                    
                    # Add formatted message (with RAG context clearly separated) to conversation history
                    # This preserves previous conversation while adding RAG to current message
                    conversation_history.append({"role": "user", "content": message_result.formatted_message})
                    
                    if config.log_output:
                        typer.echo(f"[Using: {provider_name}/{model_name}]", err=True)
                    
                    # Pass conversation history (which now includes RAG context in the user message)
                    # Gateway will use messages array if provided, message parameter is just for backward compatibility
                    llm_start = time.time()
                    response = gateway.chat(
                        message=message_result.formatted_message,  # Fallback if messages not supported
                        provider=provider,
                        model=model_name if model is None else model,
                        messages=conversation_history
                    )
                    llm_time = (time.time() - llm_start) * 1000
                    total_time = (time.time() - total_start) * 1000
                    
                    typer.echo(response)
                    
                    # Log timing breakdown
                    if config.log_output:
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        typer.echo(f"[{timestamp}] Performance Breakdown:", err=True)
                        typer.echo(f"  Message Prep: {prep_time:.1f}ms", err=True)
                        typer.echo(f"  LLM Generation: {llm_time:.1f}ms", err=True)
                        typer.echo(f"  Total: {total_time:.1f}ms", err=True)
                    
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

