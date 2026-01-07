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
    """Result of preparing a chat message with context."""
    formatted_message: str
    library_results: List[Tuple[str, float]]  # Library (document) results
    library_context_text: Optional[str] = None
    journal_results: List[Tuple[str, float]] = None  # Journal (chat history) results - now matches Library format
    journal_context_text: Optional[str] = None
    
    def __post_init__(self):
        if self.journal_results is None:
            self.journal_results = []


class ChatService:
    """
    Shared chat service for CLI and API.
    
    Handles RAG retrieval, prompt formatting, and message preparation
    in a consistent way across both interfaces.
    """
    
    # Class-level cache shared across all instances (for API multi-request scenarios)
    _class_cache: OrderedDict[str, List[Tuple[str, float]]] = OrderedDict()
    _max_cache_size = 20  # Keep last 20 queries
    
    def __init__(self, config: AppConfig, rag_instance=None, context_engine=None):
        """
        Initialize chat service.
        
        Args:
            config: Application configuration
            rag_instance: Deprecated, use context_engine instead
            context_engine: Optional pre-initialized ContextEngine instance (for performance)
        """
        self.config = config
        # Support both old (rag_instance) and new (context_engine) parameter names
        self._context_engine = context_engine or rag_instance
        self._journal = None  # Lazy-loaded from context_engine
    
    def prepare_chat_message(
        self,
        user_message: str,
        use_library: Optional[bool] = None,
        use_journal: Optional[bool] = None,
        session_id: Optional[str] = None,
        library_top_k: Optional[int] = None,
        journal_top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        system_prompt: Optional[str] = None,
        context_prompt_template: Optional[str] = None
    ) -> ChatMessageResult:
        """
        Prepare a chat message with optional context from Library and Journal.
        
        This method:
        1. Retrieves Library (document) context if enabled
        2. Retrieves Journal (chat history) context if enabled
        3. Merges and formats context for the LLM
        4. Returns the formatted message and retrieval results
        
        Args:
            user_message: The user's message/query
            use_library: Whether to use Library (document) retrieval
            use_journal: Whether to use Journal (chat history) retrieval
            session_id: Session ID for Journal filtering (optional)
            library_top_k: Number of documents to retrieve from Library
            journal_top_k: Number of entries to retrieve from Journal
            similarity_threshold: Minimum similarity score for Library
            system_prompt: Custom system prompt (overrides default if provided)
            context_prompt_template: Custom context prompt template (overrides default)
        
        Returns:
            ChatMessageResult with formatted message and retrieval results
        """
        # Master switch check
        if not self.config.chat_context_enabled:
            # Context disabled - return un-augmented message
            formatted = self._format_user_message(
                user_message=user_message,
                library_context_text=None,
                system_prompt=system_prompt,
                rag_prompt_template=context_prompt_template
            )
            return ChatMessageResult(
                formatted_message=formatted,
                library_results=[],
                library_context_text=None,
                journal_results=[],
                journal_context_text=None
            )
        
        # Use provided values or fall back to config
        use_library = use_library if use_library is not None else self.config.chat_library_enabled
        use_journal = use_journal if use_journal is not None else self.config.chat_journal_enabled
        library_top_k = library_top_k if library_top_k is not None else self.config.chat_library_top_k
        journal_top_k = journal_top_k if journal_top_k is not None else self.config.chat_journal_top_k
        similarity_threshold = (
            similarity_threshold 
            if similarity_threshold is not None 
            else self.config.chat_library_similarity_threshold
        )
        
        library_results: List[Tuple[str, float]] = []
        library_context_text: Optional[str] = None
        journal_results: List[Dict] = []
        journal_context_text: Optional[str] = None
        
        # Timing for Library retrieval
        library_start_time = time.time()
        
        # Retrieve Library context if enabled
        if use_library:
            library_results, library_context_text = self._retrieve_library_context(
                query=user_message,
                top_k=library_top_k,
                similarity_threshold=similarity_threshold
            )
        
        library_time = (time.time() - library_start_time) * 1000
        
        # Timing for Journal retrieval
        journal_start_time = time.time()
        
        # Retrieve Journal context if enabled
        if use_journal:
            journal_results, journal_context_text = self._retrieve_journal_context(
                query=user_message,
                session_id=session_id,
                limit=journal_top_k
            )
        
        journal_time = (time.time() - journal_start_time) * 1000
        
        # Timing for prompt formatting
        format_start_time = time.time()
        
        # Merge context from both sources
        merged_context = self._merge_context(
            library_context=library_context_text,
            journal_context=journal_context_text
        )
        
        # Format the user message with merged context if available
        formatted_message = self._format_user_message(
            user_message=user_message,
            library_context_text=merged_context,
            system_prompt=system_prompt,
            rag_prompt_template=context_prompt_template
        )
        
        format_time = (time.time() - format_start_time) * 1000
        
        # Log timing breakdown if enabled
        if self.config.log_output:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            logger.info(f"[{timestamp}] Message Preparation:")
            if use_library:
                logger.info(f"  Library Retrieval: {library_time:.1f}ms ({len(library_results)} docs)")
            if use_journal:
                logger.info(f"  Journal Retrieval: {journal_time:.1f}ms ({len(journal_results)} entries)")
            logger.info(f"  Prompt Formatting: {format_time:.1f}ms")
            logger.info(f"  Total Preparation: {library_time + journal_time + format_time:.1f}ms")
        
        return ChatMessageResult(
            formatted_message=formatted_message,
            library_results=library_results,
            library_context_text=library_context_text,
            journal_results=journal_results,
            journal_context_text=journal_context_text
        )
    
    def _retrieve_library_context(
        self,
        query: str,
        top_k: int,
        similarity_threshold: float
    ) -> Tuple[List[Tuple[str, float]], Optional[str]]:
        """
        Retrieve context from Library (document knowledge base).
        
        Strategies:
        1. Check context cache (if enabled)
        2. Direct vector search
        
        Args:
            query: User query/message
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
        
        Returns:
            Tuple of (library_results, library_context_text)
        """
        # Strategy 1: Check context cache
        if self.config.chat_library_use_cache:
            cached_results = self._get_cached_context(query)
            if cached_results:
                if self.config.log_output:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    logger.info(f"[{timestamp}] Library: Cache Hit")
                context_text = self._format_context_text(cached_results)
                return cached_results, context_text
        
        # Strategy 2: Direct query retrieval
        results = self._retrieve_library_direct(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Cache results for future use
        if self.config.chat_library_use_cache and results:
            self._cache_context(query, results)
        
        # Format context text
        context_text = self._format_context_text(results) if results else None
        
        if self.config.log_output:
            if results:
                logger.info(f"Library: Retrieved {len(results)} documents")
            else:
                logger.info("Library: No documents found")
        
        return results, context_text
    
    def _retrieve_library_direct(
        self,
        query: str,
        top_k: int,
        similarity_threshold: float
    ) -> List[Tuple[str, float]]:
        """
        Direct Library retrieval via vector search.
        
        Args:
            query: User query/message
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
        
        Returns:
            List of (document_text, similarity_score) tuples
        """
        try:
            # Use pre-initialized context engine if available
            if self._context_engine is not None:
                engine = self._context_engine
            else:
                from rag.rag_setup import get_rag
                self._context_engine = get_rag()
                engine = self._context_engine
            
            # Retrieve Library context
            results = engine.get_context_for_chat(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            return results
            
        except Exception as e:
            # Log error but continue without RAG
            logger.warning(
                f"RAG retrieval failed: {e}",
                exc_info=self.config.log_output
            )
            return []
    
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
    
    def _format_context_text(self, results: List[Tuple[str, float]]) -> str:
        """
        Format retrieval results into context text.
        
        Args:
            results: List of (document_text, similarity_score) tuples
        
        Returns:
            Formatted context text string
        """
        return "\n\n".join([doc for doc, _ in results])
    
    def _format_user_message(
        self,
        user_message: str,
        library_context_text: Optional[str],
        system_prompt: Optional[str] = None,
        rag_prompt_template: Optional[str] = None
    ) -> str:
        """
        Format user message with optional RAG context.
    
        
        Args:
            user_message: The user's message
            library_context_text: Merged context text (Library + Journal)
            system_prompt: Unused (kept for backward compatibility)
            rag_prompt_template: Custom context prompt template (overrides default)
        
        Returns:
            Formatted user message with RAG context prepended if available
        """
        if library_context_text:
            # Format with context clearly separated
            if rag_prompt_template:
                # Use custom template
                return format_prompt(
                    rag_prompt_template,
                    rag_context=library_context_text,
                    user_message=user_message
                )
            else:
                # Use default: make user question prominent, context as reference
                return f"""<CONTEXT_FOR_REFERENCE>
The following information is provided as reference context ONLY. It may or may not be relevant to answering the user's question below.

{library_context_text}
</CONTEXT_FOR_REFERENCE>

======================================
USER'S ACTUAL QUESTION (ANSWER THIS):
======================================
{user_message}"""
        else:
            # No RAG context - return plain user message
            return user_message
    
    def _retrieve_journal_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 5
    ) -> Tuple[List[Tuple[str, float]], Optional[str]]:
        """
        Retrieve relevant chat history from Journal.
        
        Now uses the same interface as Library retrieval for consistency.
        
        Args:
            query: User query for semantic search
            session_id: Optional session filter
            limit: Maximum entries to retrieve
            
        Returns:
            Tuple of (journal_results, formatted_context_text)
            where journal_results is List[(text, score)]
        """
        try:
            # Get Journal from ContextEngine
            if self._context_engine is None:
                from rag.rag_setup import get_rag
                self._context_engine = get_rag()
            
            journal = self._context_engine.journal
            if journal is None:
                return [], None
            
            # Use unified interface (same as Library)
            # Uses config default similarity threshold
            similarity_threshold = self.config.chat_library_similarity_threshold
            
            results = journal.get_context_for_chat(
                query=query,
                top_k=limit,
                similarity_threshold=similarity_threshold,
                session_id=None  # Always search all sessions, not just current
            )
            
            if not results:
                return [], None
            
            # Format context text (same pattern as Library)
            context_text = self._format_context_text(results)
            
            return results, context_text
            
        except Exception as e:
            logger.warning(f"Journal retrieval failed: {e}")
            return [], None
    
    def _merge_context(
        self,
        library_context: Optional[str],
        journal_context: Optional[str]
    ) -> Optional[str]:
        """
        Merge Library and Journal context into a single context string.
        
        Args:
            library_context: Context from Library (document) retrieval
            journal_context: Context from Journal (chat history) retrieval
            
        Returns:
            Merged context string, or None if both are empty
        """
        parts = []
        
        if library_context:
            parts.append("[KNOWLEDGE BASE - Documents from your personal library]\n" + library_context)
        
        if journal_context:
            parts.append("[PAST CONVERSATIONS - Previous chat history that may be relevant]\n" + journal_context)
        
        if not parts:
            return None
        
        return "\n\n".join(parts)

