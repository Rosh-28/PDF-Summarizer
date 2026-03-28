"""Enhanced generation module with citations and validation."""
from typing import Dict, List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from backend.config import rag_config
from backend.logging_config import api_logger


# Enhanced prompt template with better instructions
RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions based on provided context.

Instructions:
1. Answer ONLY based on the provided context
2. If the context doesn't contain information to answer the question, say "I don't have enough information to answer this question"
3. Be concise and clear
4. Cite specific sections when relevant

Context:
{context}

---

Question: {question}

Answer:"""


class ResponseGenerator:
    """Enhanced response generation with citations."""
    
    def __init__(self, model: str = None, temperature: float = None):
        self.model = model or rag_config.model_config.llm_model
        self.temperature = temperature or rag_config.model_config.temperature
        self._llm = None
        self.prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    @property
    def llm(self):
        """Lazy load LLM on first access."""
        if self._llm is None:
            self._llm = OllamaLLM(model=self.model, temperature=self.temperature)
            api_logger.info(f"Initialized LLM connection with model: {self.model}")
        return self._llm
    
    def generate(
        self,
        query: str,
        context_docs: List[Tuple[Document, float]],
        include_citations: bool = None
    ) -> Dict:
        """
        Generate response with optional citations.
        
        Args:
            query: User query
            context_docs: List of (Document, score) tuples
            include_citations: Include citations in response
            
        Returns:
            Dict with response, citations, and metadata
        """
        include_citations = include_citations or rag_config.retrieval_config.use_citation_tracking
        
        # Build context
        context_text = self._build_context(context_docs)
        
        # Generate response
        try:
            prompt = self.prompt_template.format(
                context=context_text,
                question=query
            )
            
            api_logger.info(f"Generating response for query: {query[:50]}...")
            response_text = self.llm.invoke(prompt)
            api_logger.info("Response generated successfully")
            
            # Extract citations
            citations = self._extract_citations(context_docs) if include_citations else []
            
            return {
                "response": response_text,
                "citations": citations,
                "num_sources": len(context_docs),
                "query": query
            }
            
        except Exception as e:
            api_logger.error(f"Response generation failed: {e}")
            raise
    
    def _build_context(self, context_docs: List[Tuple[Document, float]]) -> str:
        """Build context string from documents."""
        context_parts = []
        
        for i, (doc, score) in enumerate(context_docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_str = f" (Page {page})" if page else ""
            
            context_part = f"[Source {i}: {source}{page_str}]\n{doc.page_content}"
            context_parts.append(context_part)
        
        return "\n\n---\n\n".join(context_parts)
    
    def _extract_citations(self, context_docs: List[Tuple[Document, float]]) -> List[Dict]:
        """Extract citation information from retrieved documents."""
        citations = []
        
        for i, (doc, score) in enumerate(context_docs, 1):
            citation = {
                "source_number": i,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", None),
                "relevance_score": round(score, 3),
                "chunk_id": doc.metadata.get("id", None)
            }
            citations.append(citation)
        
        return citations
    
    def validate_response(self, response: str, query: str) -> Dict:
        """
        Validate response quality.
        
        Returns:
            Dict with validation metrics
        """
        validation = {
            "has_content": len(response.strip()) > 0,
            "min_length_met": len(response) > 20,
            "max_length_ok": len(response) < 5000,
            "contains_query_terms": any(
                term.lower() in response.lower() 
                for term in query.split() 
                if len(term) > 3
            )
        }
        
        validation["is_valid"] = all(validation.values())
        return validation
