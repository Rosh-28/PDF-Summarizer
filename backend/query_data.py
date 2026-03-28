"""Enhanced query interface with RAG pipeline."""
import argparse
import json
from backend.rag_pipeline import rag_pipeline
from backend.logging_config import logger


def main():
    """Main entry point for CLI queries."""
    parser = argparse.ArgumentParser(description="Query RAG pipeline")
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    args = parser.parse_args()
    
    result = query_rag(args.query_text, k=args.k)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_formatted_result(result)


def query_rag(query_text: str, k: int = 5) -> dict:
    """
    Query the RAG pipeline and return formatted response.
    
    Args:
        query_text: User query
        k: Number of documents to retrieve
        
    Returns:
        Dictionary with response and metadata
    """
    logger.info(f"Processing query: {query_text}")
    result = rag_pipeline.query(query_text, k=k, include_citations=True, validate=True)
    return result


def print_formatted_result(result: dict):
    """Print result in a user-friendly format."""
    print("\n" + "="*60)
    print("RESPONSE")
    print("="*60)
    print(result.get("response", "No response"))
    
    if result.get("citations"):
        print("\n" + "="*60)
        print("SOURCES")
        print("="*60)
        for citation in result.get("citations", []):
            print(f"\n[{citation['source_number']}] {citation['source']}")
            if citation.get("page") is not None:
                print(f"    Page: {citation['page']}")
            print(f"    Relevance: {citation['relevance_score']}")
    
    if result.get("validation"):
        print("\n" + "="*60)
        print("RESPONSE QUALITY")
        print("="*60)
        validation = result.get("validation", {})
        print(f"Valid: {'✓' if validation.get('is_valid') else '✗'}")
        if not validation.get("is_valid"):
            print("Issues:")
            for key, value in validation.items():
                if key != "is_valid" and not value:
                    print(f"  - {key}")
    
    print("\n")


if __name__ == "__main__":
    main()