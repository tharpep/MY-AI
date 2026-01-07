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
    library_results: List[Tuple[str, float]]
    library_context_text: Optional[str] = None
    journal_results: List[Tuple[str, float]] = None
    journal_context_text: Optional[str] = None
    
    def __post_init__(self):
        if self.journal_results is None:
            self.journal_results = []


class ChatService:
    """Shared chat service for CLI and API."""
    
    _class_cache: OrderedDict[str, List[Tuple[str, float]]] = OrderedDict()
    _max_cache_size = 20
    
    def __init__(self, config: AppConfig, rag_instance=None, context_engine=None):
        """Initialize chat service."""
        self.config = config
        self._context_engine = context_engine or rag_instance
        self._journal = None
    
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
        """Prepare a chat message with optional context from Library and Journal."""
        if not self.config.chat_context_enabled:
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
        
        library_start_time = time.time()
        
        if use_library:
            library_results, library_context_text = self._retrieve_library_context(
                query=user_message,
                top_k=library_top_k,
                similarity_threshold=similarity_threshold
            )
        
        library_time = (time.time() - library_start_time) * 1000
        
        journal_start_time = time.time()
        
        if use_journal:
            journal_results, journal_context_text = self._retrieve_journal_context(
                query=user_message,
                session_id=session_id,
                limit=journal_top_k
            )
        
        journal_time = (time.time() - journal_start_time) * 1000
        
        format_start_time = time.time()
        
        merged_context = self._merge_context(
            library_context=library_context_text,
            journal_context=journal_context_text
        )
        
        formatted_message = self._format_user_message(
            user_message=user_message,
            library_context_text=merged_context,
            system_prompt=system_prompt,
            rag_prompt_template=context_prompt_template
        )
        
        format_time = (time.time() - format_start_time) * 1000
        
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
        if self.config.chat_library_use_cache:
            cached_results = self._get_cached_context(query)
            if cached_results:
                if self.config.log_output:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    logger.info(f"[{timestamp}] Library: Cache Hit")
                context_text = self._format_context_text(cached_results)
                return cached_results, context_text
        
        results = self._retrieve_library_direct(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        if self.config.chat_library_use_cache and results:
            self._cache_context(query, results)
        
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
        try:
            if self._context_engine is not None:
                engine = self._context_engine
            else:
                from rag.rag_setup import get_rag
                self._context_engine = get_rag()
                engine = self._context_engine
            
            results = engine.get_context_for_chat(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            return results
            
        except Exception as e:
            logger.warning(
                f"RAG retrieval failed: {e}",
                exc_info=self.config.log_output
            )
            return []
    
    def _get_cached_context(self, query: str) -> Optional[List[Tuple[str, float]]]:
        if not self._class_cache:
            return None
        
        normalized_query = query.lower().strip()
        query_keywords = set(normalized_query.split())
        
        recent_entries = list(self._class_cache.items())[-5:]
        
        for cached_query, cached_results in reversed(recent_entries):
            cached_keywords = set(cached_query.lower().split())
            
            intersection = len(query_keywords & cached_keywords)
            union = len(query_keywords | cached_keywords)
            
            if union > 0:
                similarity = intersection / union
                if similarity > 0.5:
                    if self.config.log_output:
                        logger.info(f"Chat RAG - Cache hit (similarity: {similarity:.2f})")
                    self._class_cache.move_to_end(cached_query)
                    return cached_results
        
        return None
    
    def _cache_context(self, query: str, results: List[Tuple[str, float]]):
        normalized_query = query.lower().strip()
        
        if len(self._class_cache) >= self._max_cache_size:
            self._class_cache.popitem(last=False)
        
        self._class_cache[normalized_query] = results
        
        self._class_cache.move_to_end(normalized_query)
    
    def _format_context_text(self, results: List[Tuple[str, float]]) -> str:
        return "\n\n".join([doc for doc, _ in results])
    
    def _format_user_message(
        self,
        user_message: str,
        library_context_text: Optional[str],
        system_prompt: Optional[str] = None,
        rag_prompt_template: Optional[str] = None
    ) -> str:
        if library_context_text:
            if rag_prompt_template:
                return format_prompt(
                    rag_prompt_template,
                    rag_context=library_context_text,
                    user_message=user_message
                )
            else:
                return f"""<CONTEXT_FOR_REFERENCE>
The following information is provided as reference context ONLY. It may or may not be relevant to answering the user's question below.

{library_context_text}
</CONTEXT_FOR_REFERENCE>

======================================
USER'S ACTUAL QUESTION (ANSWER THIS):
======================================
{user_message}"""
        else:
            return user_message
    
    def _retrieve_journal_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 5
    ) -> Tuple[List[Tuple[str, float]], Optional[str]]:
        try:
            if self._context_engine is None:
                from rag.rag_setup import get_rag
                self._context_engine = get_rag()
            
            journal = self._context_engine.journal
            if journal is None:
                return [], None
            
            similarity_threshold = self.config.chat_library_similarity_threshold
            
            results = journal.get_context_for_chat(
                query=query,
                top_k=limit,
                similarity_threshold=similarity_threshold,
                session_id=None
            )
            
            if not results:
                return [], None
            
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
        parts = []
        
        if library_context:
            parts.append("[KNOWLEDGE BASE - Documents from your personal library]\n" + library_context)
        
        if journal_context:
            parts.append("[PAST CONVERSATIONS - Previous chat history that may be relevant]\n" + journal_context)
        
        if not parts:
            return None
        
        return "\n\n".join(parts)

