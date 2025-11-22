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
"""

import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter()


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
    use_rag: Optional[bool] = Field(None, description="Enable RAG context retrieval (overrides config default)")
    rag_top_k: Optional[int] = Field(None, ge=1, le=100, description="Top-k documents to retrieve for RAG (overrides config default)")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt (overrides default)")
    rag_prompt_template: Optional[str] = Field(None, description="Custom RAG prompt template (overrides default)")


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""
    input: str = Field(..., description="Text to embed")
    model: Optional[str] = Field(None, description="Model to use for embeddings")


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
        from ..main import rag_instance
        
        chat_service = ChatService(config, rag_instance=rag_instance)
        message_result = chat_service.prepare_chat_message(
            user_message=user_message,
            conversation_history=messages[:-1],  # All messages except the current one
            use_rag=request.use_rag,
            rag_top_k=request.rag_top_k,
            similarity_threshold=None,  # Use config default
            system_prompt=request.system_prompt,
            rag_prompt_template=request.rag_prompt_template
        )
        
        prep_time = (time.time() - prep_start) * 1000
        
        # Update the last message in messages array with formatted message
        # This preserves conversation history while adding RAG context to current message
        messages[-1]["content"] = message_result.formatted_message
        rag_results = message_result.rag_results
        
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
            logger.info(f"  RAG Context: {'Yes' if rag_results else 'No'} ({len(rag_results) if rag_results else 0} docs)")
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
            import logging
            from datetime import datetime
            logger = logging.getLogger(__name__)
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            logger.info(f"[{timestamp}] LLM Generation: {llm_time:.1f}ms")
        
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
        return {
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
