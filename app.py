"""Flask web application for PDF RAG system."""
from flask import Flask, request, render_template, jsonify
import os
import traceback
from pathlib import Path
from werkzeug.utils import secure_filename
from backend.rag_pipeline import rag_pipeline
from backend.config import rag_config
from backend.logging_config import api_logger
from backend.processing import DocumentProcessor
from backend.get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')

# Configuration
UPLOAD_FOLDER = rag_config.data_path or 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    api_logger.info(f"Created upload folder: {UPLOAD_FOLDER}")


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def index_pdf_file(filepath):
    """
    Index a single PDF file to the Chroma database.
    
    Args:
        filepath: Path to the PDF file to index
        
    Returns:
        dict: Status of indexing with success and message
    """
    try:
        processor = DocumentProcessor()
        
        # Convert to Path object if string
        pdf_path = Path(filepath) if isinstance(filepath, str) else filepath
        
        # Load the single PDF
        api_logger.info(f"Loading PDF: {filepath}")
        documents = processor._load_pdf(pdf_path)
        
        if not documents:
            return {"success": False, "message": "No text extracted from PDF (may be scanned image)"}
        
        # Split into chunks
        api_logger.info(f"Splitting {len(documents)} documents into chunks")
        chunks = processor.split_documents(documents)
        
        if not chunks:
            return {"success": False, "message": "No chunks created from PDF"}
        
        # Filter and calculate IDs
        chunks = processor.filter_chunks(chunks)
        chunks = processor.calculate_chunk_ids(chunks)
        
        if not chunks:
            return {"success": False, "message": "PDF filtered out (too small)"}
        
        # Add to database
        db = Chroma(
            persist_directory=rag_config.chroma_path,
            embedding_function=get_embedding_function()
        )
        
        # Get existing IDs to check for duplicates
        existing_ids_result = db.get(include=["metadatas"])
        existing_ids = set()
        for metadata in existing_ids_result.get("metadatas", []):
            if isinstance(metadata, dict) and "id" in metadata:
                existing_ids.add(metadata["id"])
        
        # Add new chunks
        new_chunks = [chunk for chunk in chunks if chunk.metadata.get("id") not in existing_ids]
        
        if new_chunks:
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            api_logger.info(f"Indexed {len(new_chunks)} new chunks from PDF")
            return {"success": True, "indexed_chunks": len(new_chunks)}
        else:
            return {"success": True, "indexed_chunks": 0, "message": "PDF already indexed"}
            
    except Exception as e:
        api_logger.error(f"Error indexing PDF {filepath}: {e}\n{traceback.format_exc()}")
        return {"success": False, "message": f"Indexing error: {str(e)}"}


@app.route('/')
def index():
    """Serve main page."""
    try:
        return render_template('index.html')
    except Exception as e:
        api_logger.error(f"Error serving index: {e}")
        return "Error loading page", 500


@app.route('/api/query', methods=['POST'])
def query_api():
    """
    Query the RAG pipeline.
    
    Request JSON:
    {
        "query": "Your question",
        "k": 5  (optional)
    }
    
    Response JSON:
    {
        "success": true,
        "response": "Answer",
        "citations": [...],
        "num_sources": 5
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            api_logger.warning("Query request without query field")
            return jsonify({"success": False, "error": "Query field required"}), 400
        
        query_text = data.get('query', '').strip()
        if not query_text:
            return jsonify({"success": False, "error": "Query cannot be empty"}), 400
        
        k = data.get('k', 5)
        api_logger.info(f"Query received: {query_text[:100]}... (k={k})")
        
        # process query through RAG pipeline
        result = rag_pipeline.query(query_text, k=k, include_citations=True, validate=True)
        
        return jsonify(result), 200
        
    except Exception as e:
        api_logger.error(f"Query processing error: {e}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": f"Query processing failed: {str(e)}"
        }), 500


@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """
    Upload and automatically index PDF file to the system.
    
    Response JSON:
    {
        "success": true,
        "filename": "document.pdf",
        "indexed_chunks": 45,
        "message": "PDF uploaded and indexed successfully"
    }
    """
    try:
        if 'file' not in request.files:
            api_logger.warning("Upload request without file")
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Only PDF files allowed"}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        api_logger.info(f"File uploaded: {filename}")
        
        # Automatically index the PDF
        index_result = index_pdf_file(filepath)
        
        if index_result["success"]:
            indexed_chunks = index_result.get("indexed_chunks", 0)
            message = f"PDF uploaded and indexed successfully ({indexed_chunks} chunks)"
            if indexed_chunks == 0:
                message = "PDF uploaded (already indexed or no new content)"
            
            return jsonify({
                "success": True,
                "filename": filename,
                "indexed_chunks": indexed_chunks,
                "message": message
            }), 200
        else:
            return jsonify({
                "success": False,
                "filename": filename,
                "error": index_result.get("message", "Indexing failed")
            }), 200
        
    except Exception as e:
        api_logger.error(f"Upload error: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get RAG pipeline statistics."""
    try:
        stats = rag_pipeline.get_stats()
        return jsonify({
            "success": True,
            "stats": stats
        }), 200
    except Exception as e:
        api_logger.error(f"Stats error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    api_logger.error(f"Internal error: {e}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    api_logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
