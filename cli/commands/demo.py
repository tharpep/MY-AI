"""Demo command - run various demos"""
import subprocess
from typing import Optional

import typer

from ..utils import get_python_cmd, check_venv


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
        typer.echo(f"Invalid mode: {mode}. Use 'quick' or 'full'", err=True)
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

