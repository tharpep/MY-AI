"""FastAPI startup - Personal AI Assistant API"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from llm.gateway import AIGateway

from .routes import health, llm, query, ingest, config, logs, memory, profile
from .db import init_database, log_request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global gateway instance
gateway: AIGateway = None

# Global RAG instance (initialized at startup if RAG is enabled)
rag_instance = None


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all API requests."""
    
    async def dispatch(self, request: Request, call_next):
        # Skip logging for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Get request ID from header or generate one
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            # Generate request ID if not provided
            import uuid
            request_id = f"req_{uuid.uuid4().hex[:12]}"
        
        # Start timing
        start_time = time.time()
        
        # Process request
        response = None
        status_code = 500
        error_type = None
        error_message = None
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            error_type = type(e).__name__
            error_message = str(e)
            raise
        finally:
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Log request (non-blocking)
            try:
                log_request(
                    request_id=request_id,
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=status_code,
                    response_time_ms=response_time_ms,
                    error_type=error_type,
                    error_message=error_message
                )
            except Exception as e:
                logger.error(f"Failed to log request: {e}")
        
        # Add request ID to response header
        if response:
            response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage FastAPI application lifecycle."""
    global gateway
    
    # Startup
    logger.info("Starting Personal AI Assistant API")
    
    # Initialize database
    try:
        init_database()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Continue anyway - logging is non-critical
    
    # Initialize gateway
    gateway = AIGateway()
    logger.info("AI Gateway initialized")
    
    # Initialize RAG if enabled (do this at startup to avoid delays during requests)
    global rag_instance
    try:
        from core.config import get_config
        config = get_config()
        if config.chat_context_enabled:
            logger.info("Initializing RAG system...")
            from rag.rag_setup import get_rag
            rag_instance = get_rag()
            logger.info("RAG system initialized and ready")
            
            # Start background worker for ingestion
            try:
                import asyncio
                from arq.worker import Worker
                from rag.workers import WorkerSettings
                
                # Run worker in background task
                logger.info("Starting background ingestion worker...")
                
                # Instantiate worker directly to avoid loop conflict
                # WorkerSettings doesn't support direct init in all versions, so we unpack
                worker = Worker(
                    functions=WorkerSettings.functions,
                    redis_settings=WorkerSettings.redis_settings,
                    max_jobs=WorkerSettings.max_jobs,
                    job_timeout=WorkerSettings.job_timeout
                )
                
                # Store worker in app state for cleanup
                app.state.worker = worker
                
                async def start_worker():
                    await worker.async_run()
                
                app.state.worker_task = asyncio.create_task(start_worker())
                logger.info("Background worker started")
            except Exception as e:
                logger.error(f"Failed to start background worker: {e}")
        else:
            logger.info("RAG is disabled in config")
    except Exception as e:
        logger.warning(f"RAG initialization failed, continuing without RAG: {e}")
        rag_instance = None
    
    # Initialize tool registry and register default tools
    try:
        from agents.tool_registry import get_registry
        from agents.tools.rag_answer import RAGAnswerTool
        
        registry = get_registry()
        
        # Register RAG answer tool (default tool)
        rag_tool = RAGAnswerTool()
        registry.register(rag_tool)
        
        # Set initial allowlist (v0 tools: rag_answer)
        registry.set_allowlist(["rag_answer"])
        
        logger.info("Tool registry initialized with RAG answer tool")
    except Exception as e:
        logger.error(f"Failed to initialize tool registry: {e}")
        # Continue anyway - tools are optional
    
    yield
    
    # Shutdown
    logger.info("Shutting down Personal AI Assistant API")
    
    # Stop background worker
    if hasattr(app.state, "worker") and app.state.worker:
        logger.info("Stopping background worker...")
        await app.state.worker.close()
        logger.info("Background worker stopped")
        
    if hasattr(app.state, "worker_task") and app.state.worker_task:
        try:
            await app.state.worker_task
        except asyncio.CancelledError:
            pass
    
    if gateway:
        try:
            await gateway.__aexit__(None, None, None)
        except Exception:
            pass  # Gateway might not have been entered


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Personal AI Assistant API",
        description="OpenAI-compatible API for personal AI assistant",
        version="0.1.0",
        lifespan=lifespan
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Include routes
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(llm.router, prefix="/v1", tags=["llm"])
    app.include_router(query.router, prefix="/v1", tags=["rag"])
    app.include_router(ingest.router, prefix="/v1", tags=["ingest"])
    app.include_router(config.router, tags=["config"])
    app.include_router(logs.router, tags=["logs"])
    app.include_router(memory.router, prefix="/v1", tags=["memory"])
    app.include_router(profile.router, prefix="/v1", tags=["profile"])
    
    return app


app = create_app()
