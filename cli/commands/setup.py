"""Setup command - environment setup and dependency installation"""
import sys
import subprocess
from pathlib import Path

import typer

from ..utils import get_python_cmd, check_venv_health


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
        typer.echo("You can run 'myai setup' again to retry.")
        raise typer.Exit(1)

