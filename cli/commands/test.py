"""Test command - run test suites"""
import subprocess
from pathlib import Path
from typing import Optional

import typer

from ..utils import get_python_cmd, check_venv


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

