"""Personal AI Assistant CLI - Main entry point"""
import typer

from .commands import setup, test, demo, config, chat, query, ingest

app = typer.Typer(
    name="myai",
    help="Personal AI Assistant - Local-first AI with RAG and tools",
    add_completion=True,
)

# Register subcommands
app.command(name="setup")(setup)  # Kept for backward compatibility, but poetry handles setup
app.command(name="test")(test)
app.command(name="demo")(demo)
app.command(name="config")(config)
app.command(name="chat")(chat)
app.command(name="query")(query)
app.command(name="ingest")(ingest)


def main() -> None:
    """Main CLI entry point"""
    app()


if __name__ == "__main__":
    main()

