"""
LLM API routes - OpenAI-compatible endpoints

This module provides OpenAI-compatible endpoints for:
- Chat completions: /v1/chat/completions (main chat endpoint)
- Embeddings: /v1/embeddings (text embeddings)
- Models: /v1/models (list available models)

All endpoints include:
- Request ID generation for tracing
- Proper error handling with HTTP status codes
- Structured error responses
- Token usage estimation

Session Management:
- Messages are saved to SQLite when session_id is provided
- Session switch triggers auto-ingest of previous session
- Session names are auto-generated from first user message

TODO:
- Add /v1/chat/simple endpoint: Direct gateway.chat() call without RAG context,
  session management, or request logging. Simple request/response for basic AI calls.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from core.prompts import get_prompt

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level session tracking for auto-ingest on session switch
_last_session_id: Optional[str] = None


# ===== Request/Response Models =====

class ChatMessage(BaseModel):
    """OpenAI-style chat message."""
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: Optional[str] = Field(None, description="Model to use for completion")
    provider: Optional[str] = Field(None, description="AI provider to use (e.g., 'ollama', 'purdue'). Auto-selects if not specified")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p sampling")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate")
    use_library: Optional[bool] = Field(None, description="Enable Library (document) context retrieval")
    use_journal: Optional[bool] = Field(None, description="Enable Journal (chat history) context retrieval")
    library_top_k: Optional[int] = Field(None, ge=1, le=100, description="Top-k documents to retrieve from Library")
    journal_top_k: Optional[int] = Field(None, ge=1, le=50, description="Top-k entries to retrieve from Journal")
    session_id: Optional[str] = Field(None, description="Session ID for chat history tracking and Journal context")
    save_messages: Optional[bool] = Field(True, description="Save messages to session history (requires session_id)")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt (overrides default)")
    context_prompt_template: Optional[str] = Field(None, description="Custom context prompt template (overrides default)")


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""
    input: str = Field(..., description="Text to embed")
    model: Optional[str] = Field(None, description="Model to use for embeddings")


# ===== Session Management Helpers =====


def _handle_session_management(
    session_id: str,
    user_message: str,
    assistant_response: str,
    config: Any
) -> bool:
    """
    Handle session management after a successful chat completion.

    1. Check for session switch and auto-ingest previous session if needed
    2. Save user message to SQLite
    3. Save assistant response to SQLite
    4. Auto-set session name if this is the first assistant response

    Args:
        session_id: Current session ID
        user_message: The user's message content
        assistant_response: The assistant's response content
        config: App configuration

    Returns:
        True if messages were saved successfully
    """
    global _last_session_id

    from core.session_store import get_session_store

    session_store = get_session_store()

    # Step 1: Check for session switch and auto-ingest previous session
    if _last_session_id is not None and _last_session_id != session_id:
        _maybe_auto_ingest_session(_last_session_id)

    # Step 2: Ensure session exists and update last_activity
    session_store.upsert_session(session_id)

    # Step 3: Save user message
    session_store.add_message(session_id, "user", user_message)
    session_store.increment_message_count(session_id)

    # Step 4: Save assistant response
    session_store.add_message(session_id, "assistant", assistant_response)
    session_store.increment_message_count(session_id)

    # Step 6: Auto-set session name if not set (after first assistant response)
    _maybe_auto_name_session(session_id, session_store, config)

    # Step 7: Update tracking
    _last_session_id = session_id

    if config.log_output:
        logger.debug(f"Saved messages to session: {session_id}")

    return True


def _maybe_auto_ingest_session(session_id: str) -> None:
    """
    Auto-ingest a session if it has new messages since last ingest.

    Called when switching away from a session.

    Args:
        session_id: Session ID to potentially ingest
    """
    from core.session_store import get_session_store
    from rag.journal import JournalManager

    try:
        session_store = get_session_store()

        # Check if session needs ingestion
        if session_store.has_new_messages_since_ingest(session_id):
            logger.info(f"Auto-ingesting session on switch: {session_id}")
            journal_manager = JournalManager()
            result = journal_manager.ingest_session(session_id)

            if "error" in result:
                logger.warning(f"Auto-ingest failed for {session_id}: {result['error']}")
            else:
                logger.info(f"Auto-ingested session {session_id}: {result.get('chunks_created', 0)} chunks")

    except Exception as e:
        logger.error(f"Auto-ingest error for session {session_id}: {e}")


def _maybe_auto_name_session(session_id: str, session_store: Any, config: Any) -> None:
    """
    Auto-set session name from first user message if not already set.

    Called after first assistant response.

    Args:
        session_id: Session ID
        session_store: SessionStore instance
        config: App configuration
    """
    try:
        session = session_store.get_session(session_id)

        # Only auto-name if name is not set and we have messages
        if session and session.get("name") is None and session.get("message_count", 0) >= 2:
            first_message = session_store.get_first_user_message(session_id)

            if first_message:
                # Truncate to configured max length
                max_length = config.journal_title_max_length
                if len(first_message) > max_length:
                    truncated = first_message[:max_length].rstrip() + "..."
                else:
                    truncated = first_message

                session_store.set_session_name(session_id, truncated)
                logger.debug(f"Auto-named session {session_id}: {truncated}")

    except Exception as e:
        logger.error(f"Auto-name error for session {session_id}: {e}")


# ===== Chat Completions Endpoint =====


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Dict[str, Any]:
    """OpenAI-compatible chat completions endpoint."""
    import time
    from ..main import gateway
    from ..db import log_request
    
    # Generate request ID for tracing
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    start_time = time.time()
    
    try:
        # Validate messages
        if not request.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "Messages array cannot be empty",
                        "type": "invalid_request_error",
                        "code": "empty_messages"
                    },
                    "request_id": request_id
                }
            )
        
        # Validate message roles and content
        valid_roles = {"system", "user", "assistant"}
        for i, msg in enumerate(request.messages):
            if msg.role not in valid_roles:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": {
                            "message": f"Invalid role '{msg.role}' in message {i}. Must be one of: {', '.join(valid_roles)}",
                            "type": "validation_error",
                            "code": "invalid_role"
                        },
                        "request_id": request_id
                    }
                )
            if not msg.content or not msg.content.strip():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": {
                            "message": f"Message {i} content cannot be empty",
                            "type": "validation_error",
                            "code": "empty_content"
                        },
                        "request_id": request_id
                    }
                )
        
        # Convert Pydantic messages to dicts
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Validate provider if specified
        provider_used = None
        if request.provider:
            available_providers = gateway.get_available_providers()
            if request.provider not in available_providers:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": {
                            "message": f"Provider '{request.provider}' not available. Available providers: {', '.join(available_providers)}",
                            "type": "invalid_request_error",
                            "code": "invalid_provider"
                        },
                        "request_id": request_id
                    }
                )
            provider_used = request.provider
        
        # Prepare kwargs for optional parameters (for future use when gateway supports them)
        kwargs = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        
        # Get user message
        user_message = messages[-1]["content"]
        
        # Get config
        config = gateway.config
        
        # Timing breakdown
        prep_start = time.time()
        
        # Use ChatService to prepare message with RAG context
        from core.services import ChatService
        from rag.rag_setup import get_rag
        
        chat_service = ChatService(config, context_engine=get_rag())
        message_result = chat_service.prepare_chat_message(
            user_message=user_message,
            use_library=request.use_library,
            use_journal=request.use_journal,
            session_id=request.session_id,  # Pass session_id for potential other uses
            library_top_k=request.library_top_k,
            journal_top_k=request.journal_top_k,
            similarity_threshold=None,  # Use config default
            system_prompt=request.system_prompt,
            context_prompt_template=request.context_prompt_template
        )
        
        prep_time = (time.time() - prep_start) * 1000
        
        # Add system message if not present (should be first message)
        from core.prompt_manager import get_prompt_manager
        from core.profile_manager import get_profile_manager
        
        prompt_mgr = get_prompt_manager()
        profile_mgr = get_profile_manager()
        
        system_prompt = request.system_prompt or prompt_mgr.get_system_prompt()
        
        profile_context = profile_mgr.get_context_string()
        if profile_context:
            system_prompt = f"{system_prompt}\n\n{profile_context}"
        
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Update the last message in messages array with formatted message (RAG context + user message)
        # This preserves conversation history while adding RAG context to current user message
        messages[-1]["content"] = message_result.formatted_message
        library_results = message_result.library_results
        
        # Log model/provider usage if logging enabled
        if config.log_output:
            import logging
            from datetime import datetime
            logger = logging.getLogger(__name__)
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            model_to_use = request.model or gateway.config.model_name
            provider_to_use = provider_used or "auto-select"
            logger.info(f"[{timestamp}] Chat Request:")
            logger.info(f"  Provider: {provider_to_use}, Model: {model_to_use}")
            logger.info(f"  Library Context: {'Yes' if library_results else 'No'} ({len(library_results) if library_results else 0} docs)")
            logger.info(f"  Message Prep: {prep_time:.1f}ms")
        
        # Call gateway with messages array (which now includes RAG context in the user message)
        # Gateway will use messages array if provided, message parameter is just for backward compatibility
        llm_start = time.time()
        response = gateway.chat(
            message=message_result.formatted_message,  # Fallback if messages not supported
            provider=provider_used,
            model=request.model,
            messages=messages  # Pass messages array with RAG context included
        )
        llm_time = (time.time() - llm_start) * 1000

        # Log LLM timing if enabled
        if config.log_output:
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            logger.info(f"[{timestamp}] LLM Generation: {llm_time:.1f}ms")

        # =========================================================================
        # Session Management: Save messages, auto-ingest, auto-name
        # =========================================================================
        session_saved = False
        if request.session_id and request.save_messages:
            try:
                session_saved = _handle_session_management(
                    session_id=request.session_id,
                    user_message=user_message,
                    assistant_response=response,
                    config=config
                )
            except Exception as e:
                # Don't fail the request if session management fails
                logger.error(f"Session management error: {e}")
        
        # Determine which provider was actually used
        # Note: If provider_used is None, gateway auto-selected based on config/fallback
        # We'll use the first available provider as an approximation
        if provider_used is None:
            available_providers = gateway.get_available_providers()
            if available_providers:
                # Use config provider if available, otherwise first available
                config_provider = gateway.config.provider_name
                provider_used = config_provider if config_provider in available_providers else available_providers[0]
            else:
                provider_used = "auto"  # Fallback if no providers available (shouldn't happen)
        
        # Rough token estimation (word count * 1.3)
        prompt_text = " ".join([msg["content"] for msg in messages])
        prompt_tokens = int(len(prompt_text.split()) * 1.3)
        completion_tokens = int(len(response.split()) * 1.3)
        total_tokens = prompt_tokens + completion_tokens
        
        # Log total timing breakdown if enabled
        if config.log_output:
            import logging
            from datetime import datetime
            logger = logging.getLogger(__name__)
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            total_time = (time.time() - start_time) * 1000
            logger.info(f"[{timestamp}] Total Request Time: {total_time:.1f}ms")
            logger.info(f"  Breakdown: Prep {prep_time:.1f}ms + LLM {llm_time:.1f}ms = {total_time:.1f}ms")
        
        # Get model name (use requested model or default)
        model_used = request.model or gateway.config.model_name
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Log request to database
        try:
            log_request(
                request_id=request_id,
                endpoint="/v1/chat/completions",
                method="POST",
                status_code=200,
                provider=provider_used,
                model=model_used,
                response_time_ms=response_time_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        except Exception as e:
            # Don't fail the request if logging fails
            import logging
            logging.getLogger(__name__).error(f"Failed to log request: {e}")
        
        # Convert AIGateway response to OpenAI format
        # Include RAG metadata for dev/admin page
        result = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(__import__("time").time()),
            "model": model_used,
            "provider": provider_used,  # Include provider info
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "request_id": request_id
        }

        # Add session info if messages were saved
        if request.session_id and session_saved:
            result["session"] = {
                "session_id": request.session_id,
                "messages_saved": True
            }
        
        # Add RAG context metadata (for dev/admin debugging)
        if library_results or message_result.journal_results:
            result["rag_context"] = {
                "library": {
                    "enabled": bool(library_results),
                    "doc_count": len(library_results) if library_results else 0,
                    "documents": [
                        {
                            "text": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text,  # Truncate for display
                            "similarity": float(score),
                            "full_text": doc_text  # Include full text for detailed view
                        }
                        for doc_text, score in (library_results or [])
                    ] if library_results else [],
                    "context_text": message_result.library_context_text[:500] + "..." if message_result.library_context_text and len(message_result.library_context_text) > 500 else message_result.library_context_text
                },
                "journal": {
                    "enabled": bool(message_result.journal_results),
                    "entry_count": len(message_result.journal_results) if message_result.journal_results else 0,
                    "entries": message_result.journal_results or [],
                    "context_text": message_result.journal_context_text[:500] + "..." if message_result.journal_context_text and len(message_result.journal_context_text) > 500 else message_result.journal_context_text
                },
                "prep_time_ms": prep_time,
                "llm_time_ms": llm_time
            }
        
        return result
        
    except HTTPException as e:
        # Log error request
        response_time_ms = (time.time() - start_time) * 1000
        try:
            log_request(
                request_id=request_id,
                endpoint="/v1/chat/completions",
                method="POST",
                status_code=e.status_code,
                response_time_ms=response_time_ms,
                error_type="HTTPException",
                error_message=str(e.detail)
            )
        except Exception:
            pass  # Don't fail if logging fails
        # Re-raise HTTP exceptions (already properly formatted)
        raise
    except ValueError as e:
        # Validation errors
        response_time_ms = (time.time() - start_time) * 1000
        try:
            log_request(
                request_id=request_id,
                endpoint="/v1/chat/completions",
                method="POST",
                status_code=422,
                response_time_ms=response_time_ms,
                error_type="ValueError",
                error_message=str(e)
            )
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": {
                    "message": str(e),
                    "type": "validation_error",
                    "code": "invalid_parameter"
                },
                "request_id": request_id
            }
        )
    except ConnectionError as e:
        # Connection errors (provider unavailable)
        response_time_ms = (time.time() - start_time) * 1000
        try:
            log_request(
                request_id=request_id,
                endpoint="/v1/chat/completions",
                method="POST",
                status_code=503,
                response_time_ms=response_time_ms,
                error_type="ConnectionError",
                error_message=str(e)
            )
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "message": f"AI provider unavailable: {str(e)}",
                    "type": "service_unavailable",
                    "code": "provider_unavailable"
                },
                "request_id": request_id
            }
        )
    except TimeoutError as e:
        # Timeout errors
        response_time_ms = (time.time() - start_time) * 1000
        try:
            log_request(
                request_id=request_id,
                endpoint="/v1/chat/completions",
                method="POST",
                status_code=504,
                response_time_ms=response_time_ms,
                error_type="TimeoutError",
                error_message=str(e)
            )
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error": {
                    "message": f"Request timed out: {str(e)}",
                    "type": "timeout_error",
                    "code": "request_timeout"
                },
                "request_id": request_id
            }
        )
    except Exception as e:
        # Check if error is about provider availability
        error_msg = str(e)
        response_time_ms = (time.time() - start_time) * 1000
        
        # Log error
        try:
            status_code = 503 if ("not available" in error_msg.lower() or "no providers" in error_msg.lower()) else 500
            log_request(
                request_id=request_id,
                endpoint="/v1/chat/completions",
                method="POST",
                status_code=status_code,
                response_time_ms=response_time_ms,
                error_type=type(e).__name__,
                error_message=error_msg
            )
        except Exception:
            pass
        
        if "not available" in error_msg.lower() or "no providers" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "message": error_msg,
                        "type": "service_unavailable",
                        "code": "provider_unavailable"
                    },
                    "request_id": request_id
                }
            )
        # Generic server errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Chat completion failed: {error_msg}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


# ===== Embeddings Endpoint =====

@router.post("/embeddings")
async def create_embeddings(request: EmbeddingRequest) -> Dict[str, Any]:
    """OpenAI-compatible embeddings endpoint."""
    from ..main import gateway
    
    # Generate request ID for tracing
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        # Validate input
        if not request.input or not request.input.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "Input text cannot be empty",
                        "type": "invalid_request_error",
                        "code": "empty_input"
                    },
                    "request_id": request_id
                }
            )
        
        response = await gateway.embeddings(
            prompt=request.input,
            model=request.model
        )
        
        # Convert Ollama response to OpenAI format
        embedding = response.get("embedding", [])
        
        if not embedding:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "error": {
                        "message": "Embedding provider returned empty response",
                        "type": "bad_gateway",
                        "code": "empty_embedding"
                    },
                    "request_id": request_id
                }
            )
        
        # Rough token estimation (word count * 1.3)
        prompt_tokens = int(len(request.input.split()) * 1.3)
        
        return {
            "object": "list",
            "data": [{
                "object": "embedding",
                "index": 0,
                "embedding": embedding
            }],
            "model": request.model or gateway.config.model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens
            },
            "request_id": request_id
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (already properly formatted)
        raise
    except ConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "message": f"Embedding provider unavailable: {str(e)}",
                    "type": "service_unavailable",
                    "code": "provider_unavailable"
                },
                "request_id": request_id
            }
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error": {
                    "message": f"Request timed out: {str(e)}",
                    "type": "timeout_error",
                    "code": "request_timeout"
                },
                "request_id": request_id
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Embedding generation failed: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


# ===== Models Endpoint =====

@router.get("/models")
async def list_models() -> Dict[str, Any]:
    """List available models."""
    from ..main import gateway
    
    # Generate request ID for tracing
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        # Get available providers and their models
        providers = gateway.get_available_providers()
        models = []
        
        if not providers:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "message": "No AI providers available. Please configure at least one provider.",
                        "type": "service_unavailable",
                        "code": "no_providers"
                    },
                    "request_id": request_id
                }
            )
        
        for provider in providers:
            try:
                if provider == "ollama":
                    # Get Ollama models
                    ollama_client = gateway.providers["ollama"]
                    ollama_models = ollama_client.get_available_models()
                    models.extend([{"id": model, "provider": "ollama"} for model in ollama_models])
                elif provider == "purdue":
                    # Get Purdue models
                    purdue_client = gateway.providers["purdue"]
                    purdue_models = purdue_client.get_available_models()
                    models.extend([{"id": model, "provider": "purdue"} for model in purdue_models])
            except Exception as e:
                # Log provider error but continue with other providers
                import logging
                logging.warning(f"Failed to get models from provider {provider}: {e}")
                continue
        
        if not models:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "message": "No models available from any provider",
                        "type": "service_unavailable",
                        "code": "no_models"
                    },
                    "request_id": request_id
                }
            )
        
        return {
            "object": "list",
            "data": [
                {
                    "id": model["id"],
                    "object": "model",
                    "owned_by": model["provider"],
                    "permission": []
                }
                for model in models
            ],
            "request_id": request_id
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (already properly formatted)
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to list models: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )
