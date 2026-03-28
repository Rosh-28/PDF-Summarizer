"""Enhanced retrieval module with reranking and query expansion."""
from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from backend.config import rag_config
from backend.get_embedding_function import get_embedding_function
from backend.logging_config import retrieval_logger


class EnhancedRetriever:
    """Enhanced retrieval with reranking and query expansion."""
    
    def __init__(self, chroma_path: str = None):
        self.chroma_path = chroma_path or rag_config.chroma_path
        self.embedding_function = None
        self._db = None
    
    @property
    def db(self):
        """Lazy load database on first access."""
        if self._db is None:
            try:
                if self.embedding_function is None:
                    self.embedding_function = get_embedding_function()
                self._db = Chroma(
                    persist_directory=self.chroma_path,
                    embedding_function=self.embedding_function
                )
                retrieval_logger.info(f"Database initialized from {self.chroma_path}")
            except Exception as e:
                retrieval_logger.error(f"Failed to initialize database: {e}")
                raise
        return self._db
    
    def retrieve(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with optional reranking.
        
        Args:
            query: Query text
            k: Number of documents to return
            
        Returns:
            List of (Document, score) tuples where score is 0-1
        """
        k = k or rag_config.retrieval_config.k
        retrieval_logger.info(f"Retrieving {k} documents for query: {query[:100]}...")
        
        try:
            # Initial retrieval with more documents for reranking
            if rag_config.retrieval_config.use_reranking:
                initial_k = rag_config.retrieval_config.rerank_top_k
            else:
                initial_k = k
            
            results = self.db.similarity_search_with_score(query, k=initial_k)
            retrieval_logger.info(f"Retrieved {len(results)} initial results")
            
            # Normalize scores to 0-1 range (Chroma returns distances, convert to similarity)
            normalized_results = []
            for doc, score in results:
                # Chroma distance: lower is better. Convert to similarity: 1 / (1 + distance)
                normalized_score = 1 / (1 + score)
                normalized_results.append((doc, normalized_score))
            
            # Filter by score threshold
            threshold = rag_config.retrieval_config.score_threshold
            filtered_results = [
                (doc, score) for doc, score in normalized_results 
                if score >= threshold
            ]
            retrieval_logger.info(f"After filtering: {len(filtered_results)} results")
            
            # Rerank if enabled
            if rag_config.retrieval_config.use_reranking and len(filtered_results) > k:
                filtered_results = self._rerank_results(query, filtered_results, k)
            
            # Return top k
            return filtered_results[:k]
            
        except Exception as e:
            retrieval_logger.error(f"Retrieval failed: {e}")
            raise
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[Tuple[Document, float]], 
        k: int
    ) -> List[Tuple[Document, float]]:
        """
        Rerank results using BM25-style algorithm.
        Simple implementation: boost score based on query term overlap.
        Keeps scores normalized to 0-1 range.
        """
        retrieval_logger.debug(f"Reranking {len(results)} results")
        query_terms = set(query.lower().split())
        
        reranked = []
        for doc, score in results:
            content_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms & content_terms)
            # Boost score but cap at 1.0 to maintain 0-1 range
            boost = 1.0 + (overlap * 0.05)
            new_score = min(score * boost, 1.0)
            reranked.append((doc, new_score, overlap))
        
        # Sort by new score
        reranked.sort(key=lambda x: x[1], reverse=True)
        retrieval_logger.debug(f"Top reranked result overlap: {reranked[0][2]} terms")
        
        # Return top k without the overlap count
        return [(doc, score) for doc, score, _ in reranked[:k]]
    
    def get_retrieval_stats(self) -> dict:
        """Get database statistics."""
        try:
            stats = self.db.get(include=[])
            return {
                "total_documents": len(stats.get("ids", [])),
                "chroma_path": self.chroma_path
            }
        except Exception as e:
            retrieval_logger.error(f"Failed to get stats: {e}")
            return {}
