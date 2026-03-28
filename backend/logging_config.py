"""Logging configuration for RAG pipeline."""
import logging
import sys
from datetime import datetime


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Setup logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


# Create loggers
logger = setup_logger("rag_pipeline")
retrieval_logger = setup_logger("retrieval")
processing_logger = setup_logger("processing")
api_logger = setup_logger("api")
