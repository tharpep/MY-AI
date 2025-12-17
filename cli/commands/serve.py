"""CLI command to start the full application stack"""
import subprocess
import signal
import sys
import time
import typer
import socket

from core.config import get_config


def _check_redis_running(host: str = "localhost", port: int = 6379) -> bool:
    """Check if Redis is running and accepting connections"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def _start_redis_container() -> bool:
    """Start Redis container if not running"""
    try:
        # Check if container exists but is stopped
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=myai-redis", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        
        if "Exited" in result.stdout:
            # Container exists but stopped, start it
            typer.echo("[Redis] Starting existing Redis container...")
            subprocess.run(["docker", "start", "myai-redis"], check=True)
        elif not result.stdout.strip():
            # Container doesn't exist, create it
            typer.echo("[Redis] Creating Redis container...")
            subprocess.run([
                "docker", "run", "-d",
                "--name", "myai-redis",
                "-p", "6379:6379",
                "redis:7-alpine"
            ], check=True)
        else:
            # Container is already running
            typer.echo("[Redis] Redis container already running")
            return True
        
        # Wait for Redis to be ready
        config = get_config()
        for i in range(10):
            if _check_redis_running(config.redis_host, config.redis_port):
                typer.echo("[Redis] Redis is ready")
                return True
            time.sleep(0.5)
        
        typer.echo("[Redis] Redis started but not responding", err=True)
        return False
        
    except subprocess.CalledProcessError as e:
        typer.echo(f"[Redis] Failed to start Redis: {e}", err=True)
        return False
    except FileNotFoundError:
        typer.echo("[Redis] Docker not found. Please install Docker or start Redis manually.", err=True)
        return False


def _check_qdrant_running(host: str = "localhost", port: int = 6333) -> bool:
    """Check if Qdrant is running and accepting connections"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def _start_qdrant_container() -> bool:
    """Start Qdrant container if not running"""
    try:
        # Check if container exists but is stopped
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=myai-qdrant", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        
        if "Exited" in result.stdout:
            # Container exists but stopped, start it
            typer.echo("[Qdrant] Starting existing Qdrant container...")
            subprocess.run(["docker", "start", "myai-qdrant"], check=True)
        elif not result.stdout.strip():
            # Container doesn't exist, create it
            typer.echo("[Qdrant] Creating Qdrant container...")
            # Get absolute path for volume mount
            import os
            data_path = os.path.abspath("./data/qdrant_db")
            os.makedirs(data_path, exist_ok=True)
            subprocess.run([
                "docker", "run", "-d",
                "--name", "myai-qdrant",
                "-p", "6333:6333",
                "-p", "6334:6334",
                "-v", f"{data_path}:/qdrant/storage",
                "qdrant/qdrant:latest"
            ], check=True)
        else:
            # Container is already running
            typer.echo("[Qdrant] Qdrant container already running")
            return True
        
        # Wait for Qdrant to be ready
        config = get_config()
        for i in range(20):  # Qdrant takes longer to start
            if _check_qdrant_running(config.qdrant_host, config.qdrant_port):
                typer.echo("[Qdrant] Qdrant is ready")
                return True
            time.sleep(0.5)
        
        typer.echo("[Qdrant] Qdrant started but not responding", err=True)
        return False
        
    except subprocess.CalledProcessError as e:
        typer.echo(f"[Qdrant] Failed to start Qdrant: {e}", err=True)
        return False
    except FileNotFoundError:
        typer.echo("[Qdrant] Docker not found. Please install Docker or start Qdrant manually.", err=True)
        return False


def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(True, "--reload/--no-reload", help="Enable auto-reload"),
    worker: bool = typer.Option(True, "--worker/--no-worker", help="Start Redis worker"),
    auto_redis: bool = typer.Option(True, "--auto-redis/--no-auto-redis", help="Auto-start Redis if not running"),
    auto_qdrant: bool = typer.Option(True, "--auto-qdrant/--no-auto-qdrant", help="Auto-start Qdrant if not running"),
) -> None:
    """
    Start the API server and all required services.
    
    Starts:
    - Qdrant (auto-starts container if not running)
    - Redis (auto-starts container if not running)
    - FastAPI server (uvicorn)
    - Redis worker (arq) if --worker flag is set
    """
    processes = []
    
    def cleanup(signum=None, frame=None):
        """Clean up all spawned processes"""
        typer.echo("\n[Shutdown] Stopping services...")
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        typer.echo("[Shutdown] All services stopped.")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        # Get config for host/port settings
        config = get_config()
        
        # Check/start Qdrant (vector database)
        if not _check_qdrant_running(config.qdrant_host, config.qdrant_port):
            if auto_qdrant:
                typer.echo("[Startup] Qdrant not running, attempting to start...")
                if not _start_qdrant_container():
                    typer.echo("[Startup] Qdrant failed to start. Will use in-memory storage.", err=True)
            else:
                typer.echo("[Startup] Qdrant not running. Will use in-memory storage.", err=True)
        else:
            typer.echo("[Startup] Qdrant is running")
        
        # Check/start Redis (job queue)
        if not _check_redis_running(config.redis_host, config.redis_port):
            if auto_redis:
                typer.echo("[Startup] Redis not running, attempting to start...")
                if not _start_redis_container():
                    typer.echo("[Startup] Cannot start without Redis. Use --no-worker to skip worker.", err=True)
                    if worker:
                        raise typer.Exit(1)
            else:
                typer.echo("[Startup] Redis not running. Worker will not start.", err=True)
                worker = False
        else:
            typer.echo("[Startup] Redis is running")
        
        # Start worker if requested and Redis is available
        if worker:
            typer.echo("[Startup] Starting Redis worker...")
            # Don't suppress output so we can see errors
            worker_proc = subprocess.Popen(
                [sys.executable, "-m", "arq", "rag.workers.WorkerSettings"],
                # Inherit stdout/stderr so worker output is visible
            )
            processes.append(worker_proc)
            
            # Give worker a moment to start and check if it crashed
            time.sleep(1)
            if worker_proc.poll() is not None:
                typer.echo(f"[Startup] Worker exited immediately with code: {worker_proc.returncode}", err=True)
            else:
                typer.echo(f"[Startup] Worker started (PID: {worker_proc.pid})")
        
        # Build uvicorn command - use sys.executable to ensure correct Python
        uvicorn_cmd = [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", host,
            "--port", str(port),
        ]
        if reload:
            uvicorn_cmd.append("--reload")
        
        typer.echo(f"[Startup] Starting API server on {host}:{port}...")
        api_proc = subprocess.Popen(uvicorn_cmd)
        processes.append(api_proc)
        
        typer.echo("[Startup] ✓ All services running. Press Ctrl+C to stop.\n")
        
        # Wait for API process (main process)
        api_proc.wait()
        
    except KeyboardInterrupt:
        cleanup()
    finally:
        cleanup()
