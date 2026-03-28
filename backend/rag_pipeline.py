"""Complete RAG pipeline orchestration."""
from typing import Dict, Optional
from backend.retrieval import EnhancedRetriever
from backend.generation import ResponseGenerator
from backend.config import rag_config
from backend.logging_config import logger


class RAGPipeline:
    """Main RAG pipeline that orchestrates retrieval and generation."""
    
    def __init__(self):
        self.retriever = EnhancedRetriever()
        self.generator = ResponseGenerator()
        logger.info("RAG Pipeline initialized")
    
    def query(
        self,
        query_text: str,
        k: int = None,
        include_citations: bool = True,
        validate: bool = True
    ) -> Dict:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query_text: User query
            k: Number of documents to retrieve
            include_citations: Include source citations
            validate: Validate response quality
            
        Returns:
            Dict with response, sources, and metadata
        """
        logger.info(f"Processing query: {query_text[:100]}...")
        
        try:
            # Step 1: Retrieve documents
            context_docs = self.retriever.retrieve(query_text, k=k)
            
            if not context_docs:
                logger.warning("No relevant documents found")
                return {
                    "response": "I couldn't find relevant information to answer your question.",
                    "citations": [],
                    "num_sources": 0,
                    "query": query_text,
                    "success": False,
                    "error": "No relevant documents found"
                }
            
            # Step 2: Generate response
            result = self.generator.generate(
                query_text,
                context_docs,
                include_citations=include_citations
            )
            
            # Step 3: Validate response
            if validate:
                validation = self.generator.validate_response(
                    result["response"],
                    query_text
                )
                result["validation"] = validation
            
            result["success"] = True
            logger.info(f"Query processed successfully. Sources: {result['num_sources']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "response": f"An error occurred while processing your query: {str(e)}",
                "success": False,
                "error": str(e),
                "query": query_text
            }
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "retriever_stats": self.retriever.get_retrieval_stats(),
            "model": rag_config.model_config.llm_model,
            "embedding_model": rag_config.model_config.embedding_model,
            "config": {
                "chunk_size": rag_config.chunk_config.size,
                "retrieval_k": rag_config.retrieval_config.k,
                "use_reranking": rag_config.retrieval_config.use_reranking,
                "use_query_expansion": rag_config.use_query_expansion
            }
        }


# Global pipeline instance
rag_pipeline = RAGPipeline()
