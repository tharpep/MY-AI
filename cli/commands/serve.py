"""CLI command to start the full application stack"""
import subprocess
import signal
import sys
import time
import typer
import socket


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
        for i in range(10):
            if _check_redis_running():
                typer.echo("[Redis] Redis is ready")
                return True
            time.sleep(0.5)
        
        typer.echo("[Redis] ⚠️  Redis started but not responding", err=True)
        return False
        
    except subprocess.CalledProcessError as e:
        typer.echo(f"[Redis] ❌ Failed to start Redis: {e}", err=True)
        return False
    except FileNotFoundError:
        typer.echo("[Redis] ❌ Docker not found. Please install Docker or start Redis manually.", err=True)
        return False


def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(True, "--reload/--no-reload", help="Enable auto-reload"),
    worker: bool = typer.Option(True, "--worker/--no-worker", help="Start Redis worker"),
    auto_redis: bool = typer.Option(True, "--auto-redis/--no-auto-redis", help="Auto-start Redis if not running"),
) -> None:
    """
    Start the API server and Redis worker.
    
    Starts:
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
        # Check/start Redis
        if not _check_redis_running():
            if auto_redis:
                typer.echo("[Startup] Redis not running, attempting to start...")
                if not _start_redis_container():
                    typer.echo("[Startup] ❌ Cannot start without Redis. Use --no-worker to skip worker.", err=True)
                    if worker:
                        raise typer.Exit(1)
            else:
                typer.echo("[Startup] ⚠️  Redis not running. Worker will not start.", err=True)
                worker = False
        else:
            typer.echo("[Startup] ✓ Redis is running")
        
        # Start worker if requested and Redis is available
        if worker:
            typer.echo("[Startup] Starting Redis worker...")
            worker_proc = subprocess.Popen(
                ["python", "-m", "arq", "rag.workers.WorkerSettings"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            processes.append(worker_proc)
            typer.echo(f"[Startup] ✓ Worker started (PID: {worker_proc.pid})")
        
        # Build uvicorn command
        uvicorn_cmd = [
            "python", "-m", "uvicorn",
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
