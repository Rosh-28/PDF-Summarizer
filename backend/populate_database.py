"""Enhanced document loading script with improved processing."""
import argparse
import os
import shutil
from langchain_core.documents import Document
from langchain_chroma import Chroma
from backend.get_embedding_function import get_embedding_function
from backend.processing import DocumentProcessor
from backend.config import rag_config
from backend.logging_config import processing_logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    
    if args.reset:
        processing_logger.info("Clearing Database")
        clear_database()

    # Create (or update) the data store.
    processor = DocumentProcessor()
    documents = processor.load_documents()
    
    if len(documents) == 0:
        processing_logger.warning("No documents were loaded. Check DATA_PATH and PDF files.")
        return

    chunks = processor.split_documents(documents)
    if len(chunks) == 0:
        processing_logger.warning("No chunks created from documents. PDFs may be scanned images or empty.")
        return
    
    # Filter small chunks and calculate IDs
    chunks = processor.filter_chunks(chunks)
    chunks = processor.calculate_chunk_ids(chunks)

    add_to_chroma(chunks)



def add_to_chroma(chunks: list[Document]):
    """Add chunks to Chroma database, skipping duplicates."""
    db = Chroma(
        persist_directory=rag_config.chroma_path,
        embedding_function=get_embedding_function()
    )

    # Get existing IDs
    existing_ids_result = db.get(include=["metadatas"])
    existing_ids = set()
    for metadata in existing_ids_result.get("metadatas", []):
        if isinstance(metadata, dict) and "id" in metadata:
            existing_ids.add(metadata["id"])

    processing_logger.info(f"Number of existing documents in DB: {len(existing_ids)}")

    # Find new chunks
    new_chunks = []
    for chunk in chunks:
        cid = chunk.metadata.get("id")
        if cid is None:
            processing_logger.warning("Chunk has no id metadata; skipping")
            continue
        if cid not in existing_ids:
            new_chunks.append(chunk)

    # Add new chunks
    if len(new_chunks):
        processing_logger.info(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        processing_logger.info("Documents added successfully")
    else:
        processing_logger.info("No new documents to add")


def clear_database():
    """Clear the Chroma database."""
    if os.path.exists(rag_config.chroma_path):
        shutil.rmtree(rag_config.chroma_path)
        processing_logger.info(f"Database cleared: {rag_config.chroma_path}")


if __name__ == "__main__":
    main()
