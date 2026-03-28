"""Enhanced document processing with better chunking and metadata."""
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.config import rag_config
from backend.logging_config import processing_logger


class DocumentProcessor:
    """Enhanced document processing with metadata enrichment."""
    
    def __init__(self, chunk_config=None):
        self.chunk_config = chunk_config or rag_config.chunk_config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_config.size,
            chunk_overlap=self.chunk_config.overlap,
            length_function=len,
            is_separator_regex=False,
        )
        processing_logger.info(
            f"DocumentProcessor initialized with chunk_size={self.chunk_config.size}, "
            f"overlap={self.chunk_config.overlap}"
        )
    
    def load_documents(self, data_path: str = None) -> List[Document]:
        """
        Load PDF documents from directory.
        
        Args:
            data_path: Path to PDFs directory
            
        Returns:
            List of loaded documents with metadata
        """
        data_path = data_path or rag_config.data_path
        data_dir = Path(data_path)
        
        if not data_dir.exists():
            processing_logger.error(f"Data path does not exist: {data_dir.resolve()}")
            return []
        
        pdf_files = sorted(data_dir.glob("*.pdf"))
        processing_logger.info(f"Found {len(pdf_files)} PDF files")
        
        all_docs = []
        for pdf_file in pdf_files:
            try:
                docs = self._load_pdf(pdf_file)
                all_docs.extend(docs)
                processing_logger.info(f"Loaded {len(docs)} docs from {pdf_file.name}")
            except Exception as e:
                processing_logger.error(f"Failed to load {pdf_file.name}: {e}")
        
        processing_logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs
    
    def _load_pdf(self, pdf_path: Path) -> List[Document]:
        """Load single PDF and add metadata."""
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        
        if not docs:
            processing_logger.warning(f"No text extracted from: {pdf_path.name}")
            return []
        
        # Enrich metadata
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
            doc.metadata["source_path"] = str(pdf_path)
            if "page" not in doc.metadata:
                doc.metadata["page"] = 0
        
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks with metadata preservation.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks with preserved metadata
        """
        if not documents:
            processing_logger.warning("No documents to split")
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        processing_logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        
        return chunks
    
    def calculate_chunk_ids(self, chunks: List[Document]) -> List[Document]:
        """
        Calculate unique IDs for chunks based on source and content.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Chunks with added 'id' metadata
        """
        last_page_id = None
        current_chunk_index = 0
        
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)
            current_page_id = f"{source}:{page}"
            
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
            
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            chunk.metadata["id"] = chunk_id
            last_page_id = current_page_id
        
        processing_logger.info(f"Assigned IDs to all {len(chunks)} chunks")
        return chunks
    
    def filter_chunks(self, chunks: List[Document], min_length: int = None) -> List[Document]:
        """
        Filter out chunks that are too small.
        
        Args:
            chunks: List of document chunks
            min_length: Minimum chunk length
            
        Returns:
            Filtered chunks
        """
        min_length = min_length or self.chunk_config.min_size
        original_count = len(chunks)
        
        filtered_chunks = [
            chunk for chunk in chunks
            if len(chunk.page_content.strip()) >= min_length
        ]
        
        removed = original_count - len(filtered_chunks)
        if removed > 0:
            processing_logger.info(f"Filtered out {removed} chunks smaller than {min_length} chars")
        
        return filtered_chunks
