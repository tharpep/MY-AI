"""CLI command implementations"""
import sys
import subprocess
from pathlib import Path
from typing import Optional

import typer

from .utils import get_python_cmd, check_venv, check_venv_health


def setup() -> None:
    """Setup virtual environment and install dependencies (with health check)"""
    typer.echo("=== Environment Setup ===")

    # Check if venv already exists
    venv_exists = Path("venv").exists()

    if venv_exists:
        typer.echo("Virtual environment already exists.")

        # Check if it's healthy
        typer.echo("Checking virtual environment health...")
        is_healthy = check_venv_health()

        if is_healthy:
            typer.echo("Virtual environment is healthy and ready to use!")
            raise typer.Exit(0)
        else:
            typer.echo("Virtual environment exists but has issues.")
            typer.echo("Reinstalling dependencies to fix...")
    else:
        typer.echo("Creating virtual environment...")
        result = subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        if result.returncode != 0:
            typer.echo("Failed to create virtual environment", err=True)
            raise typer.Exit(1)
        typer.echo("Virtual environment created")

    # Install/upgrade dependencies
    python_cmd = get_python_cmd()
    typer.echo("Installing dependencies...")

    # Upgrade pip first
    result = subprocess.run([python_cmd, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    if result.returncode != 0:
        typer.echo("Failed to upgrade pip", err=True)
        raise typer.Exit(1)

    # Install dependencies
    result = subprocess.run([python_cmd, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    if result.returncode != 0:
        typer.echo("Failed to install dependencies", err=True)
        raise typer.Exit(1)

    typer.echo("Dependencies installed successfully")

    # Final health check
    typer.echo("Performing final health check...")
    if check_venv_health():
        typer.echo("Setup completed successfully! Virtual environment is ready.")
        raise typer.Exit(0)
    else:
        typer.echo("Setup completed but virtual environment may have issues.")
        typer.echo("You can run 'my-ai setup' again to retry.")
        raise typer.Exit(1)


def test(
    all_tests: bool = typer.Option(False, "--all", "-a", help="Run all tests"),
    category: Optional[str] = typer.Argument(None, help="Test category to run (e.g., tests_api, tests_rag)"),
) -> None:
    """Run tests - specify category directly or use interactive selection"""
    if not check_venv():
        raise typer.Exit(1)

    python_cmd = get_python_cmd()

    # Check if user wants to run all tests
    if all_tests:
        typer.echo("Running all tests...")
        try:
            result = subprocess.run([python_cmd, "-m", "pytest", "tests/"], check=True)
            typer.echo("All tests passed!")
            raise typer.Exit(0)
        except subprocess.CalledProcessError as e:
            typer.echo(f"Tests failed with exit code {e.returncode}", err=True)
            raise typer.Exit(e.returncode)

    # If category specified, run that directly
    if category:
        test_path = Path("tests") / category
        if not test_path.exists():
            typer.echo(f"Test category '{category}' not found!", err=True)
            typer.echo("Available categories:", err=True)
            _list_test_categories()
            raise typer.Exit(1)

        typer.echo(f"Running: {category}")
        typer.echo("=" * 50)
        try:
            result = subprocess.run([python_cmd, "-m", "pytest", str(test_path), "-v", "-s"], check=True)
            typer.echo(f"\n{category} completed successfully!")
            raise typer.Exit(0)
        except subprocess.CalledProcessError as e:
            typer.echo(f"\nTest failed with exit code {e.returncode}", err=True)
            raise typer.Exit(e.returncode)

    # No category specified - show interactive selection
    _run_tests_interactive(python_cmd)


def _list_test_categories() -> None:
    """List available test categories"""
    tests_dir = Path("tests")
    if not tests_dir.exists():
        return

    test_folders = []
    for item in tests_dir.iterdir():
        if item.is_dir() and (item.name.startswith("test_") or item.name.startswith("tests_")):
            test_folders.append(item.name)

    for folder in sorted(test_folders):
        typer.echo(f"  - {folder}", err=True)


def _run_tests_interactive(python_cmd: str) -> None:
    """Interactive test selection menu"""
    tests_dir = Path("tests")
    if not tests_dir.exists():
        typer.echo("No tests directory found!", err=True)
        raise typer.Exit(1)

    # Find all test folders
    test_folders = []
    for item in tests_dir.iterdir():
        if item.is_dir() and (item.name.startswith("test_") or item.name.startswith("tests_")):
            test_folders.append(item.name)

    if not test_folders:
        typer.echo("No test folders found in tests/ directory!", err=True)
        raise typer.Exit(1)

    # Interactive test selection
    typer.echo("=== Test Selection ===")
    typer.echo("")

    # Display test options
    typer.echo("Available test folders:")
    for i, folder_name in enumerate(test_folders, 1):
        typer.echo(f"  {i}. {folder_name}")
    typer.echo("  0. Exit")
    typer.echo("")

    # Get user selection
    try:
        choice = typer.prompt("Enter test number", default="0").strip()

        if choice == "0":
            typer.echo("Exiting...")
            raise typer.Exit(0)

        choice_num = int(choice)
        if 1 <= choice_num <= len(test_folders):
            selected_folder = test_folders[choice_num - 1]
            typer.echo(f"\nRunning: {selected_folder}")
            typer.echo("=" * 50)

            # Run pytest on the selected folder with verbose output
            result = subprocess.run([python_cmd, "-m", "pytest", f"tests/{selected_folder}/", "-v", "-s"], check=True)

            typer.echo(f"\n{selected_folder} completed successfully!")
            raise typer.Exit(0)
        else:
            typer.echo(f"Invalid choice: {choice_num}", err=True)
            raise typer.Exit(1)

    except ValueError:
        typer.echo("Invalid input. Please enter a number.", err=True)
        raise typer.Exit(1)
    except KeyboardInterrupt:
        typer.echo("\nOperation cancelled.")
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        typer.echo(f"\nTest failed with exit code {e.returncode}", err=True)
        raise typer.Exit(e.returncode)


def demo(
    demo_type: Optional[str] = typer.Argument(None, help="Demo type: rag, llm, tuning, api"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Mode: automated (default), quick, full"),
) -> None:
    """Run demos - specify type and mode directly, or use interactive selection"""
    if not check_venv():
        raise typer.Exit(1)

    python_cmd = get_python_cmd()

    # If demo_type specified, run directly
    if demo_type:
        demo_type_lower = demo_type.lower()
        if demo_type_lower == "rag":
            _run_rag_demo(python_cmd, mode)
        elif demo_type_lower == "llm":
            _run_llm_demo(python_cmd, mode)
        elif demo_type_lower == "tuning":
            _run_tuning_demo(python_cmd, mode)
        elif demo_type_lower == "api":
            _run_api_demo(python_cmd)
        else:
            typer.echo(f"Unknown demo type: {demo_type}", err=True)
            typer.echo("Available: rag, llm, tuning, api", err=True)
            raise typer.Exit(1)
        return

    # No demo_type specified - show interactive selection
    _run_demo_interactive(python_cmd)


def _run_rag_demo(python_cmd: str, mode: Optional[str]) -> None:
    """Run RAG demo with specified mode"""
    if not mode:
        mode = "automated"  # Default

    if mode != "automated":
        typer.echo(f"Invalid mode: {mode}. Only 'automated' mode is supported.", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nRunning: RAG Demo (Automated)")
    typer.echo("=" * 50)
    subprocess.run([python_cmd, "-c", f"from rag.demo import run_rag_demo; run_rag_demo('automated')"], check=True)


def _run_llm_demo(python_cmd: str, mode: Optional[str]) -> None:
    """Run LLM demo with specified mode"""
    if not mode:
        mode = "automated"  # Default

    if mode != "automated":
        typer.echo(f"Invalid mode: {mode}. Only 'automated' mode is supported.", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nRunning: LLM Demo (Automated)")
    typer.echo("=" * 50)
    subprocess.run([python_cmd, "-c", f"from llm.demo import run_llm_demo; run_llm_demo('automated')"], check=True)


def _run_tuning_demo(python_cmd: str, mode: Optional[str]) -> None:
    """Run Tuning demo with specified mode"""
    if not mode:
        mode = "quick"  # Default

    if mode not in ["quick", "full"]:
        typer.echo(f"❌ Invalid mode: {mode}. Use 'quick' or 'full'", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nRunning: Tuning Demo ({mode.capitalize()})")
    typer.echo("=" * 50)
    subprocess.run([python_cmd, "-c", f"from tuning.demo import run_tuning_demo; run_tuning_demo('{mode}')"], check=True)


def _run_api_demo(python_cmd: str) -> None:
    """Run API demo (FastAPI server)"""
    typer.echo("\nRunning: API Demo (FastAPI Server)")
    typer.echo("=" * 50)
    typer.echo("Starting FastAPI server on http://localhost:8000")
    typer.echo("Press Ctrl+C to stop the server")
    typer.echo("")
    subprocess.run([python_cmd, "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"], check=True)


def _run_demo_interactive(python_cmd: str) -> None:
    """Interactive demo selection menu"""
    typer.echo("=== Demo Selection ===")
    typer.echo("")
    typer.echo("Available demos:")
    typer.echo("  1. RAG Demo")
    typer.echo("  2. LLM Demo (Direct AI chat)")
    typer.echo("  3. Tuning Demo")
    typer.echo("  4. API Demo (FastAPI server)")
    typer.echo("  0. Exit")
    typer.echo("")

    try:
        choice1 = typer.prompt("Enter demo number", default="0").strip()

        if choice1 == "0":
            typer.echo("Exiting...")
            raise typer.Exit(0)

        choice1_num = int(choice1)

        if choice1_num == 1:  # RAG Demo
            _run_rag_demo_interactive(python_cmd)
        elif choice1_num == 2:  # LLM Demo
            _run_llm_demo_interactive(python_cmd)
        elif choice1_num == 3:  # Tuning Demo
            _run_tuning_demo_interactive(python_cmd)
        elif choice1_num == 4:  # API Demo
            _run_api_demo(python_cmd)
        else:
            typer.echo("Invalid choice", err=True)
            raise typer.Exit(1)

        typer.echo("\nDemo completed successfully!")
        raise typer.Exit(0)

    except ValueError:
        typer.echo("Invalid input. Please enter a number.", err=True)
        raise typer.Exit(1)
    except KeyboardInterrupt:
        typer.echo("\nOperation cancelled.")
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        typer.echo(f"\nDemo failed with exit code {e.returncode}", err=True)
        raise typer.Exit(e.returncode)


def _run_rag_demo_interactive(python_cmd: str) -> None:
    """Run RAG demo - automated mode only"""
    _run_rag_demo(python_cmd, "automated")


def _run_llm_demo_interactive(python_cmd: str) -> None:
    """Run LLM demo - automated mode only"""
    _run_llm_demo(python_cmd, "automated")


def _run_tuning_demo_interactive(python_cmd: str) -> None:
    """Run Tuning demo - defaults to quick mode"""
    _run_tuning_demo(python_cmd, "quick")


def config() -> None:
    """Show current configuration settings"""
    if not check_venv():
        raise typer.Exit(1)

    typer.echo("=== Current Configuration ===")
    typer.echo("")

    try:
        from core.utils.config import get_rag_config, get_tuning_config

        rag_config = get_rag_config()
        tuning_config = get_tuning_config()

        typer.echo("Hardware Configuration:")
        typer.echo(f"  Platform: {'Laptop' if rag_config.use_laptop else 'PC'}")
        typer.echo(f"  Model: {rag_config.model_name}")
        typer.echo("")

        typer.echo("RAG Configuration:")
        typer.echo(f"  AI Provider: {'Ollama' if rag_config.use_ollama else 'Purdue API'}")
        typer.echo(f"  Storage: {'Persistent' if rag_config.use_persistent else 'In-memory'}")
        typer.echo(f"  Collection: {rag_config.collection_name}")
        typer.echo(f"  Top-K: {rag_config.top_k}")
        typer.echo("")

        typer.echo("Tuning Configuration:")
        typer.echo(f"  Model: {tuning_config.model_name}")
        typer.echo(f"  Batch Size: {tuning_config.batch_size}")
        typer.echo(f"  Epochs: {tuning_config.num_epochs}")
        typer.echo(f"  Output Dir: {tuning_config.output_dir}")
        typer.echo("")

        typer.echo("Note: You can override these settings by editing config.py")
        typer.echo("or setting environment variables (USE_LAPTOP, USE_OLLAMA, etc.)")

    except ImportError as e:
        typer.echo(f"Error: Could not import config module: {e}", err=True)
        raise typer.Exit(1)

    raise typer.Exit(0)


def chat(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="AI provider to use (ollama, purdue)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
) -> None:
    """Interactive chat with the AI - type your questions and get responses"""
    if not check_venv():
        raise typer.Exit(1)

    try:
        from llm.gateway import AIGateway
        from core.utils.config import get_rag_config

        typer.echo("Initializing AI Gateway...")
        gateway = AIGateway()
        config = get_rag_config()

        # Determine provider and model info
        provider_name = provider or ("ollama" if config.use_ollama else "purdue")
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



