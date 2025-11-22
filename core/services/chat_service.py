"""
Chat Service - Shared business logic for chat functionality

This service encapsulates the common chat logic used by both CLI and API,
including RAG retrieval, prompt formatting, and message preparation.
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict

from core.config import AppConfig
from core.prompts import get_prompt, format_prompt

logger = logging.getLogger(__name__)


@dataclass
class ChatMessageResult:
    """Result of preparing a chat message with RAG context."""
    formatted_message: str
    rag_results: List[Tuple[str, float]]
    rag_context_text: Optional[str] = None


class ChatService:
    """
    Shared chat service for CLI and API.
    
    Handles RAG retrieval, prompt formatting, and message preparation
    in a consistent way across both interfaces.
    """
    
    # Class-level cache shared across all instances (for API multi-request scenarios)
    _class_cache: OrderedDict[str, List[Tuple[str, float]]] = OrderedDict()
    _max_cache_size = 20  # Keep last 20 queries
    
    def __init__(self, config: AppConfig, rag_instance=None):
        """
        Initialize chat service.
        
        Args:
            config: Application configuration
            rag_instance: Optional pre-initialized BasicRAG instance (for performance)
        """
        self.config = config
        self._rag_instance = rag_instance
    
    def prepare_chat_message(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        use_rag: Optional[bool] = None,
        rag_top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        system_prompt: Optional[str] = None,
        rag_prompt_template: Optional[str] = None
    ) -> ChatMessageResult:
        """
        Prepare a chat message with optional RAG context.
        
        This method:
        1. Retrieves RAG context if enabled
        2. Formats the user message with RAG context (if available)
        3. Returns the formatted message and RAG results
        
        Args:
            user_message: The user's message/query
            conversation_history: List of previous messages (role/content dicts)
            use_rag: Whether to use RAG (overrides config if provided)
            rag_top_k: Number of documents to retrieve (overrides config if provided)
            similarity_threshold: Minimum similarity score (overrides config if provided)
            system_prompt: Custom system prompt (overrides default if provided)
            rag_prompt_template: Custom RAG prompt template (overrides default if provided)
        
        Returns:
            ChatMessageResult with formatted message and RAG results
        """
        # Use provided values or fall back to config
        use_rag = use_rag if use_rag is not None else self.config.chat_rag_enabled
        rag_top_k = rag_top_k if rag_top_k is not None else self.config.chat_rag_top_k
        similarity_threshold = (
            similarity_threshold 
            if similarity_threshold is not None 
            else self.config.chat_rag_similarity_threshold
        )
        
        rag_results: List[Tuple[str, float]] = []
        rag_context_text: Optional[str] = None
        
        # Timing for RAG retrieval
        rag_start_time = time.time()
        
        # Retrieve RAG context if enabled
        if use_rag:
            rag_results, rag_context_text = self._retrieve_rag_context_hybrid(
                query=user_message,
                conversation_history=conversation_history,
                top_k=rag_top_k,
                similarity_threshold=similarity_threshold
            )
        
        rag_time = (time.time() - rag_start_time) * 1000  # Convert to ms
        
        # Timing for prompt formatting
        format_start_time = time.time()
        
        # Format the user message with RAG context if available
        formatted_message = self._format_user_message(
            user_message=user_message,
            rag_context_text=rag_context_text,
            system_prompt=system_prompt,
            rag_prompt_template=rag_prompt_template
        )
        
        format_time = (time.time() - format_start_time) * 1000  # Convert to ms
        
        # Log timing breakdown if enabled
        if self.config.log_output:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
            logger.info(f"[{timestamp}] Message Preparation:")
            if use_rag:
                logger.info(f"  RAG Retrieval: {rag_time:.1f}ms ({len(rag_results)} docs)")
            logger.info(f"  Prompt Formatting: {format_time:.1f}ms")
            logger.info(f"  Total Preparation: {rag_time + format_time:.1f}ms")
        
        return ChatMessageResult(
            formatted_message=formatted_message,
            rag_results=rag_results,
            rag_context_text=rag_context_text
        )
    
    def _retrieve_rag_context_hybrid(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        top_k: int,
        similarity_threshold: float
    ) -> Tuple[List[Tuple[str, float]], Optional[str]]:
        """
        Hybrid RAG retrieval with multiple fallback strategies.
        
        Strategy order:
        1. Check context cache (if enabled)
        2. Direct query retrieval
        3. Conversation-aware query (if minimal results and enabled)
        4. Lower threshold adaptive retrieval (if minimal results)
        
        Args:
            query: User query/message
            conversation_history: List of previous messages (role/content dicts)
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
        
        Returns:
            Tuple of (rag_results, rag_context_text)
        """
        retrieval_start = time.time()
        strategy_used = None
        
        # Strategy 1: Check context cache
        cache_check_start = time.time()
        if self.config.chat_rag_use_context_cache:
            cached_results = self._get_cached_context(query)
            cache_check_time = (time.time() - cache_check_start) * 1000
            if cached_results:
                if self.config.log_output:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    logger.info(f"[{timestamp}] RAG Strategy: Cache Hit ({cache_check_time:.1f}ms)")
                rag_context_text = self._format_context_text(cached_results)
                return cached_results, rag_context_text
        
        # Strategy 2: Direct query retrieval
        direct_start = time.time()
        rag_results = self._retrieve_rag_context_direct(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        direct_time = (time.time() - direct_start) * 1000
        strategy_used = "direct"
        
        min_results = self.config.chat_rag_min_results_threshold
        
        # Strategy 3: Conversation-aware query if minimal results
        if len(rag_results) < min_results and self.config.chat_rag_use_conversation_aware:
            conv_start = time.time()
            if self.config.log_output:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                logger.info(f"[{timestamp}] RAG Strategy: Direct ({direct_time:.1f}ms) → Minimal results ({len(rag_results)}), trying conversation-aware")
            
            conversation_query = self._build_conversation_aware_query(
                query=query,
                conversation_history=conversation_history
            )
            
            conversation_results = self._retrieve_rag_context_direct(
                query=conversation_query,
                top_k=top_k,
                similarity_threshold=similarity_threshold * 0.8  # Slightly lower threshold
            )
            conv_time = (time.time() - conv_start) * 1000
            
            # Use conversation results if better
            if len(conversation_results) > len(rag_results):
                rag_results = conversation_results
                strategy_used = "conversation-aware"
                if self.config.log_output:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    logger.info(f"[{timestamp}] RAG Strategy: Conversation-aware ({conv_time:.1f}ms) → Found {len(rag_results)} results")
        
        # Strategy 4: Lower threshold if still minimal results
        if len(rag_results) < min_results:
            lower_start = time.time()
            if self.config.log_output:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                logger.info(f"[{timestamp}] RAG Strategy: Still minimal ({len(rag_results)}), trying lower threshold")
            
            lower_threshold_results = self._retrieve_rag_context_direct(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold * 0.5  # Much lower threshold
            )
            lower_time = (time.time() - lower_start) * 1000
            
            # Use lower threshold results if better
            if len(lower_threshold_results) > len(rag_results):
                rag_results = lower_threshold_results
                strategy_used = "lower-threshold"
                if self.config.log_output:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    logger.info(f"[{timestamp}] RAG Strategy: Lower threshold ({lower_time:.1f}ms) → Found {len(rag_results)} results")
        
        total_retrieval_time = (time.time() - retrieval_start) * 1000
        if self.config.log_output and strategy_used:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            logger.info(f"[{timestamp}] RAG Retrieval Complete: {strategy_used} strategy, {total_retrieval_time:.1f}ms total")
        
        # Cache results for future use
        if self.config.chat_rag_use_context_cache:
            self._cache_context(query, rag_results)
        
        # Format context text
        rag_context_text = self._format_context_text(rag_results) if rag_results else None
        
        if self.config.log_output:
            if rag_results:
                logger.info(f"Chat RAG - Using {len(rag_results)} documents as context")
            else:
                logger.info("Chat RAG - No documents retrieved after all strategies")
        
        return rag_results, rag_context_text
    
    def _retrieve_rag_context_direct(
        self,
        query: str,
        top_k: int,
        similarity_threshold: float
    ) -> List[Tuple[str, float]]:
        """
        Direct RAG context retrieval (single strategy).
        
        Args:
            query: User query/message
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
        
        Returns:
            List of (document_text, similarity_score) tuples
        """
        try:
            # Use pre-initialized RAG instance if available, otherwise create new one
            if self._rag_instance is not None:
                rag = self._rag_instance
            else:
                from rag.rag_setup import BasicRAG
                rag = BasicRAG()
            
            # Retrieve RAG context
            rag_results = rag.get_context_for_chat(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            return rag_results
            
        except Exception as e:
            # Log error but continue without RAG
            logger.warning(
                f"RAG retrieval failed: {e}",
                exc_info=self.config.log_output
            )
            return []
    
    def _build_conversation_aware_query(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        max_history_turns: int = 3
    ) -> str:
        """
        Build a query that includes relevant conversation context.
        
        Args:
            query: Current user query
            conversation_history: List of previous messages
            max_history_turns: Maximum number of conversation turns to include
        
        Returns:
            Enhanced query string with conversation context
        """
        if not conversation_history:
            return query
        
        # Get recent conversation turns (last N user-assistant pairs)
        recent_turns = conversation_history[-max_history_turns * 2:]
        
        # Extract key information from recent conversation
        conversation_context = []
        for msg in recent_turns:
            role = msg.get("role", "")
            content = msg.get("content", "")
            # Skip system messages and very long messages
            if role in ["user", "assistant"] and len(content) < 500:
                conversation_context.append(f"{role}: {content[:200]}")
        
        if not conversation_context:
            return query
        
        # Build enhanced query
        context_summary = "\n".join(conversation_context[-4:])  # Last 4 messages max
        enhanced_query = f"{query}\n\nContext from recent conversation:\n{context_summary}"
        
        return enhanced_query
    
    def _get_cached_context(self, query: str) -> Optional[List[Tuple[str, float]]]:
        """
        Get cached context if query is similar to previous ones.
        
        Args:
            query: Current query (normalized for matching)
        
        Returns:
            Cached results if found, None otherwise
        """
        if not self._class_cache:
            return None
        
        # Normalize query for matching (lowercase, strip)
        normalized_query = query.lower().strip()
        query_keywords = set(normalized_query.split())
        
        # Check recent cache entries first (most likely to match)
        # Limit to last 5 entries for performance
        recent_entries = list(self._class_cache.items())[-5:]
        
        for cached_query, cached_results in reversed(recent_entries):
            cached_keywords = set(cached_query.lower().split())
            
            # Calculate Jaccard similarity (intersection over union)
            intersection = len(query_keywords & cached_keywords)
            union = len(query_keywords | cached_keywords)
            
            if union > 0:
                similarity = intersection / union
                # If >50% keyword overlap, reuse cached context
                if similarity > 0.5:
                    if self.config.log_output:
                        logger.info(f"Chat RAG - Cache hit (similarity: {similarity:.2f})")
                    # Move to end (most recently used)
                    self._class_cache.move_to_end(cached_query)
                    return cached_results
        
        return None
    
    def _cache_context(self, query: str, results: List[Tuple[str, float]]):
        """
        Cache retrieved context for future use.
        
        Args:
            query: Query string (will be normalized)
            results: Retrieved RAG results
        """
        # Normalize query for consistent caching
        normalized_query = query.lower().strip()
        
        # Keep cache size manageable
        if len(self._class_cache) >= self._max_cache_size:
            # Remove oldest entry (FIFO)
            self._class_cache.popitem(last=False)
        
        # Store results
        self._class_cache[normalized_query] = results
        
        # Move to end (most recently used)
        self._class_cache.move_to_end(normalized_query)
    
    def _format_context_text(self, rag_results: List[Tuple[str, float]]) -> str:
        """
        Format RAG results into context text.
        
        Args:
            rag_results: List of (document_text, similarity_score) tuples
        
        Returns:
            Formatted context text string
        """
        return "\n\n".join([doc for doc, _ in rag_results])
    
    def _format_user_message(
        self,
        user_message: str,
        rag_context_text: Optional[str],
        system_prompt: Optional[str] = None,
        rag_prompt_template: Optional[str] = None
    ) -> str:
        """
        Format user message with optional RAG context.
        
        Args:
            user_message: The user's message
            rag_context_text: RAG context text (if available)
            system_prompt: Custom system prompt (overrides default)
            rag_prompt_template: Custom RAG prompt template (overrides default)
        
        Returns:
            Formatted message string ready for LLM
        """
        if rag_context_text:
            # Format with RAG context clearly separated
            if rag_prompt_template:
                template = rag_prompt_template
            else:
                template = get_prompt("llm_with_rag")
            
            return format_prompt(
                template,
                rag_context=rag_context_text,
                user_message=user_message
            )
        else:
            # No RAG - use regular format
            if system_prompt:
                system = system_prompt
            else:
                system = get_prompt("llm")
            
            return f"{system}\n\nUser: {user_message}"

