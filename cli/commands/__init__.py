"""CLI command modules - exports all commands"""
from .setup import setup
from .test import test
from .demo import demo
from .config import config
from .chat import chat
from .query import query
from .ingest import ingest

__all__ = ["setup", "test", "demo", "config", "chat", "query", "ingest"]

