"""Configuration settings for RAG pipeline."""
import os
from dataclasses import dataclass, field


@dataclass
class ChunkConfig:
    """Document chunking configuration."""
    size: int = 800
    overlap: int = 80
    min_size: int = 200  # Minimum chunk size to avoid fragments


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    k: int = 5  # Number of documents to retrieve
    score_threshold: float = 0.5  # Minimum similarity score (0-1 range)
    use_reranking: bool = True
    rerank_top_k: int = 10  # Number of docs to rerank before selecting top k


@dataclass
class ModelConfig:
    """LLM configuration."""
    llm_model: str = "mistral"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass
class RAGConfig:
    """Main RAG configuration."""
    chroma_path: str = os.getenv("CHROMA_PATH", "data/chroma")
    data_path: str = os.getenv("DATA_PATH", "PDFs")
    chunk_config: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    
    # Feature flags
    use_query_expansion: bool = True
    use_citation_tracking: bool = True
    use_logging: bool = True
    verbose: bool = False


# Global config instance
rag_config = RAGConfig()
