"""Setup Poetry command - Install and configure Poetry"""
import sys
import subprocess
import shutil
import platform

import typer

app = typer.Typer()


def check_poetry_installed() -> bool:
    """Check if Poetry is installed and available."""
    return shutil.which("poetry") is not None


def check_shell_plugin_installed() -> bool:
    """Check if Poetry shell plugin is installed."""
    try:
        result = subprocess.run(
            ["poetry", "self", "show", "poetry-plugin-shell"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def install_poetry_windows() -> bool:
    """Install Poetry on Windows using the official installer."""
    typer.echo("Installing Poetry on Windows...")
    typer.echo("\nRunning official Poetry installer via PowerShell...")
    
    try:
        # Use PowerShell to run the official installer
        ps_command = (
            "(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -"
        )
        result = subprocess.run(
            ["powershell", "-Command", ps_command],
            check=False,
        )
        
        if result.returncode == 0:
            typer.echo("\n✓ Poetry installed successfully!")
            typer.echo("\nPlease restart your terminal or add Poetry to PATH:")
            typer.echo('  $env:Path += ";$env:USERPROFILE\\.poetry\\bin"')
            return True
        else:
            typer.echo("\n✗ Automatic installation failed.")
            typer.echo("\nPlease install Poetry manually using PowerShell:")
            typer.echo("  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -")
            typer.echo("\nOr visit: https://python-poetry.org/docs/#installation")
            return False
    except Exception as e:
        typer.echo(f"\n✗ Error during installation: {e}")
        typer.echo("\nPlease install Poetry manually using PowerShell:")
        typer.echo("  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -")
        typer.echo("\nOr visit: https://python-poetry.org/docs/#installation")
        return False


def install_poetry_unix() -> bool:
    """Install Poetry on Unix-like systems using the official installer."""
    typer.echo("Installing Poetry on Unix-like system...")
    typer.echo("\nRunning official Poetry installer...")
    
    try:
        # Check if curl is available
        if not shutil.which("curl"):
            typer.echo("✗ curl is not available. Trying pip installation...")
            return install_poetry_pip()
        
        # Use curl to download and run installer
        result = subprocess.run(
            "curl -sSL https://install.python-poetry.org | python3 -",
            shell=True,
            check=False,
        )
        
        if result.returncode == 0:
            typer.echo("\n✓ Poetry installed successfully!")
            typer.echo("\nPlease restart your terminal or add Poetry to PATH:")
            typer.echo('  export PATH="$HOME/.local/bin:$PATH"')
            return True
        else:
            typer.echo("\n✗ Automatic installation failed.")
            typer.echo("\nPlease install Poetry manually:")
            typer.echo("  curl -sSL https://install.python-poetry.org | python3 -")
            typer.echo("\nOr visit: https://python-poetry.org/docs/#installation")
            return False
    except Exception as e:
        typer.echo(f"\n✗ Error during installation: {e}")
        typer.echo("\nPlease install Poetry manually:")
        typer.echo("  curl -sSL https://install.python-poetry.org | python3 -")
        typer.echo("\nOr visit: https://python-poetry.org/docs/#installation")
        return False


def install_poetry_pip() -> bool:
    """Install Poetry using pip (fallback method)."""
    typer.echo("Installing Poetry via pip...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user", "poetry"],
            check=False,
        )
        
        if result.returncode == 0:
            typer.echo("Poetry installed successfully via pip!")
            typer.echo("Note: Official installer is recommended for best experience.")
            return True
        else:
            return False
    except Exception as e:
        typer.echo(f"Error during pip installation: {e}")
        return False


def setup_poetry(
    install: bool = typer.Option(
        False,
        "--install",
        "-i",
        help="Attempt to install Poetry if not found",
    ),
    install_plugin: bool = typer.Option(
        True,
        "--install-plugin/--no-install-plugin",
        help="Install Poetry shell plugin (Poetry 2.0+ requirement)",
    ),
) -> None:
    """
    Setup Poetry for the project.
    
    Checks if Poetry is installed and optionally installs it.
    Also installs the shell plugin required for Poetry 2.0+.
    """
    typer.echo("=== Poetry Setup ===")
    
    # Check if Poetry is installed
    if check_poetry_installed():
        typer.echo("✓ Poetry is already installed")
        
        # Check version
        try:
            result = subprocess.run(
                ["poetry", "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            typer.echo(f"  {result.stdout.strip()}")
        except Exception:
            pass
    else:
        typer.echo("✗ Poetry is not installed")
        
        if install:
            typer.echo("\nAttempting to install Poetry...")
            system = platform.system().lower()
            
            if system == "windows":
                success = install_poetry_windows()
            else:
                # Try official installer first
                success = install_poetry_unix()
                if not success:
                    # Fallback to pip
                    typer.echo("Trying pip installation as fallback...")
                    success = install_poetry_pip()
            
            if success:
                typer.echo("\n✓ Poetry installation completed!")
                typer.echo("Please restart your terminal and run this command again.")
                raise typer.Exit(0)
            else:
                typer.echo("\n✗ Automatic installation failed.")
                typer.echo("Please install Poetry manually:")
                typer.echo("  Visit: https://python-poetry.org/docs/#installation")
                raise typer.Exit(1)
        else:
            typer.echo("\nTo install Poetry, run:")
            typer.echo("  myai setup-poetry --install")
            typer.echo("\nOr install manually:")
            typer.echo("  Visit: https://python-poetry.org/docs/#installation")
            raise typer.Exit(1)
    
    # Check and install shell plugin
    if install_plugin:
        typer.echo("\nChecking Poetry shell plugin...")
        
        if check_shell_plugin_installed():
            typer.echo("✓ Poetry shell plugin is already installed")
        else:
            typer.echo("✗ Poetry shell plugin is not installed")
            typer.echo("Installing Poetry shell plugin (required for Poetry 2.0+)...")
            
            try:
                result = subprocess.run(
                    ["poetry", "self", "add", "poetry-plugin-shell"],
                    check=True,
                )
                
                if result.returncode == 0:
                    typer.echo("✓ Poetry shell plugin installed successfully!")
                else:
                    typer.echo("✗ Failed to install shell plugin")
                    typer.echo("You can install it manually with:")
                    typer.echo("  poetry self add poetry-plugin-shell")
            except subprocess.CalledProcessError:
                typer.echo("✗ Failed to install shell plugin")
                typer.echo("You can install it manually with:")
                typer.echo("  poetry self add poetry-plugin-shell")
            except Exception as e:
                typer.echo(f"✗ Error installing shell plugin: {e}")
                typer.echo("You can install it manually with:")
                typer.echo("  poetry self add poetry-plugin-shell")
    
    typer.echo("\n✓ Poetry setup complete!")
    typer.echo("\nNext steps:")
    typer.echo("  1. Run: poetry install")
    typer.echo("  2. Run: poetry shell (or use poetry run prefix)")
    typer.echo("  3. Run: myai setup")


if __name__ == "__main__":
    setup_poetry()

