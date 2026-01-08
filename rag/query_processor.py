"""Query processor for LLM-based query expansion."""

from typing import Optional


QUERY_EXPANSION_PROMPT = """Rewrite this search query to be more specific and detailed for document retrieval. 
Add relevant synonyms and related terms. Output ONLY the expanded query, nothing else.

Original query: {query}

Expanded query:"""


class QueryProcessor:
    """Processes and expands queries before retrieval."""
    
    def __init__(self, gateway=None, model: str = None):
        """Initialize the query processor."""
        self._gateway = gateway
        self.model = model
    
    @property
    def gateway(self):
        if self._gateway is None:
            from llm.gateway import AIGateway
            self._gateway = AIGateway()
        return self._gateway
    
    def expand(self, query: str, model: str = None) -> str:
        """Expand a query using LLM to add relevant terms."""
        if len(query.strip()) < 5:
            return query
        
        use_model = model or self.model
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        
        try:
            expanded = self.gateway.chat(prompt, model=use_model)
            expanded = expanded.strip()
            
            if expanded and len(expanded) > len(query):
                return expanded
            return query
        except Exception:
            return query
