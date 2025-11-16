"""CLI utility functions"""
import os
import subprocess
from pathlib import Path

import typer


def get_python_cmd() -> str:
    """Get the correct Python command for the platform"""
    if os.name == "nt":  # Windows
        return "venv\\Scripts\\python"
    else:  # Unix/Linux/Mac
        return "venv/bin/python"


def check_venv() -> bool:
    """Check if virtual environment exists"""
    venv_path = Path("venv")
    if not venv_path.exists():
        typer.echo("Virtual environment not found!", err=True)
        typer.echo("Run: my-ai setup first")
        return False
    return True


def check_venv_health() -> bool:
    """Check the health of the virtual environment"""
    if not check_venv():
        return False

    python_cmd = get_python_cmd()

    # Check if Python executable exists and works
    try:
        result = subprocess.run([python_cmd, "--version"], capture_output=True, text=True, check=True)
        typer.echo(f"Python version: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        typer.echo("Python executable not found or not working", err=True)
        return False

    # Check if pip is working
    try:
        result = subprocess.run([python_cmd, "-m", "pip", "--version"], capture_output=True, text=True, check=True)
        typer.echo(f"Pip version: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        typer.echo("Pip not working", err=True)
        return False

    # Check if key packages are installed
    key_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("sentence-transformers", "sentence_transformers"),
        ("qdrant-client", "qdrant_client"),
    ]
    missing_packages = []

    for package_name, import_name in key_packages:
        try:
            subprocess.run([python_cmd, "-c", f"import {import_name}"], capture_output=True, text=True, check=True)
            typer.echo(f"{package_name} is installed")
        except subprocess.CalledProcessError:
            typer.echo(f"{package_name} is missing", err=True)
            missing_packages.append(package_name)

    if missing_packages:
        typer.echo(f"\nMissing packages: {', '.join(missing_packages)}", err=True)
        return False

    typer.echo("Virtual environment is healthy!")
    return True

