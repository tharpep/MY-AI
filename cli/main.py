"""Personal AI Assistant CLI - Main entry point"""
import typer

from .commands import setup, test, demo, config

app = typer.Typer(
    name="my-ai",
    help="Personal AI Assistant - Local-first AI with RAG and tools",
    add_completion=False,
)

# Register subcommands
app.command(name="setup")(setup)
app.command(name="test")(test)
app.command(name="demo")(demo)
app.command(name="config")(config)


def main() -> None:
    """Main CLI entry point"""
    app()


if __name__ == "__main__":
    main()

