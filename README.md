"# DocRAG - PDF Document Intelligence System

A modern, professional Retrieval-Augmented Generation (RAG) system that enables intelligent querying of PDF documents with AI-powered semantic search, automatic source citation, and response validation.

## 🎯 What is DocRAG?

DocRAG is a complete solution for building an intelligent document Q&A system. Upload your PDFs, and use natural language to search and query your documents. The system automatically:

- **Indexes** documents with semantic embeddings
- **Retrieves** relevant content using advanced search
- **Generates** accurate answers with source citations
- **Validates** response quality automatically

Perfect for knowledge bases, internal documentation, research libraries, and corporate document management.

---

## ✨ Key Features

### Smart Document Retrieval
- Vector-based semantic search (not just keyword matching)
- Automatic relevance scoring and filtering
- Result reranking for better accuracy
- Configurable retrieval sensitivity

### Intelligent Response Generation
- LLM-powered context-aware answers
- Automatic source citations with page numbers
- Quality metrics and validation
- Structured response format

### Automatic Processing
- PDF parsing and intelligent chunking
- Automatic indexing on upload (no manual steps)
- Duplicate detection and deduplication
- Database statistics and monitoring

### Professional Interface
- Modern, clean web UI
- Tab-based navigation (Query / Upload / Stats)
- Real-time upload and indexing
- Response formatting with citations

### Enterprise Ready
- RESTful API for integration
- Comprehensive logging and monitoring
- Configuration management
- Error handling and validation

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+
- Ollama (for local LLM: `ollama pull mistral`)
- PDFs to analyze

### 2. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment (optional)
export DATA_PATH="./PDFs"
export CHROMA_PATH="./data/chroma"
```

### 3. Start the System

```bash
# Run the application
python app.py

# Open in browser
http://localhost:5000
```

### 4. Upload and Query
1. Go to **Upload** tab → Add your PDFs
2. PDFs are automatically indexed
3. Go to **Query** tab → Ask questions
4. View answers with sources and quality metrics

---

## 📚 How It Works

```
┌─────────────────────────────────────────────┐
│         User Query or PDF Upload            │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│         Document Processing                 │
│  • PDF loading & text extraction            │
│  • Chunking (800 chars, 80 char overlap)   │
│  • Metadata enrichment                      │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│         Embedding & Storage                 │
│  • Semantic embeddings (nomic-embed-text)   │
│  • Chroma vector database                   │
│  • Automatic deduplication                  │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│         Query Processing                    │
│  • Semantic similarity search               │
│  • Score filtering & reranking              │
│  • Result ranking by relevance              │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│         Response Generation                 │
│  • LLM contextual generation (Mistral)      │
│  • Citation extraction                      │
│  • Quality validation                       │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│    Formatted Answer with Sources            │
│  • Main response                            │
│  • Source citations with page #             │
│  • Quality metrics                          │
└─────────────────────────────────────────────┘
```

---

## 🎨 Web Interface

### Query Tab
- Ask questions about your documents
- Adjust number of sources to retrieve (1-20)
- View answers with automatic citations
- Quality metrics show response validity
- Copy button for quick sharing

### Upload Tab
- Drag & drop PDF files
- Shows automatic indexing status
- Displays number of chunks indexed
- Real-time processing feedback

### Stats Tab
- View indexed document count
- See configured LLM and embedding models
- Check document chunk size settings
- Refresh to update statistics

---

## 🔌 API Endpoints

### Query Document
```bash
POST /api/query
Content-Type: application/json

{
  "query": "What does the document say about X?",
  "k": 5
}

Response:
{
  "success": true,
  "response": "The document states...",
  "citations": [
    {
      "source_number": 1,
      "source": "document.pdf",
      "page": 3,
      "relevance_score": 0.92
    }
  ],
  "validation": {
    "is_valid": true,
    "min_length_met": true,
    "contains_query_terms": true
  }
}
```

### Upload PDF
```bash
POST /api/upload
Content-Type: multipart/form-data

[PDF file in multipart body]

Response:
{
  "success": true,
  "filename": "document.pdf",
  "indexed_chunks": 45,
  "message": "PDF uploaded and indexed successfully"
}
```

### Get Statistics
```bash
GET /api/stats

Response:
{
  "success": true,
  "stats": {
    "retriever_stats": {
      "total_documents": 1250
    },
    "model": "mistral",
    "embedding_model": "nomic-embed-text",
    "config": {
      "chunk_size": 800,
      "retrieval_k": 5
    }
  }
}
```

---

## ⚙️ Configuration

Edit `backend/config.py` to customize behavior:

```python
# Document Processing
chunk_size = 800              # Characters per chunk
chunk_overlap = 80            # Overlap between chunks
min_size = 200                # Minimum chunk size

# Retrieval
retrieval_k = 5               # Documents to retrieve
score_threshold = 0.5         # Minimum similarity (0-1)
use_reranking = True          # Enable result reranking
rerank_top_k = 10             # Documents to consider for reranking

# Models
llm_model = "mistral"         # LLM to use
embedding_model = "nomic-embed-text"  # Embedding model
temperature = 0.7             # Generation randomness
max_tokens = 512              # Maximum response length

# Features
use_query_expansion = True    # Additional query variations
use_citation_tracking = True  # Track and cite sources
use_logging = True            # Enable logging
verbose = False               # Verbose logging
```

---

## 📁 Project Structure

```
PDF Summarizer/
├── app.py                          # Flask web server
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── frontend/
│   ├── static/
│   │   └── style.css              # Modern UI styling
│   └── templates/
│       └── index.html             # Web interface
│
├── backend/
│   ├── config.py                  # Configuration settings
│   ├── logging_config.py          # Logging setup
│   ├── rag_pipeline.py            # Main orchestrator
│   ├── retrieval.py               # Document retrieval
│   ├── generation.py              # Response generation
│   ├── processing.py              # PDF processing
│   ├── populate_database.py       # Indexing script
│   ├── query_data.py              # CLI interface
│   ├── get_embedding_function.py  # Embedding setup
│   └── __pycache__/
│
├── PDFs/                          # Input PDF directory
├── data/
│   └── chroma/                    # Vector database
└── uploads/                       # Uploaded files
```

---

## 🔧 Common Tasks

### Index New PDFs
Place PDFs in the `PDFs/` folder, then either:
- **Automatic:** Upload via web UI (auto-indexes)
- **Manual:** Run `python -m backend.populate_database`

### Query from Command Line
```bash
python -m backend.query_data "Your question here"
python -m backend.query_data "Your question here" --json
```

### Reset Database
```bash
python -m backend.populate_database --reset
```

### View Logs
Logs are printed to console with timestamps and log levels.

### Adjust Retrieval Sensitivity
Edit `backend/config.py`:
- Increase `retrieval_k` for more results
- Lower `score_threshold` for less strict filtering
- Disable `use_reranking` for faster queries

---

## 🐛 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "No relevant documents found" | Empty database or low similarity | Upload PDFs, lower `score_threshold` |
| 500 error on stats | Configuration issue | Check `backend/config.py` syntax |
| Slow responses | Too many documents | Reduce `retrieval_k`, disable reranking |
| Poor answer quality | Bad retrieval | Add more PDFs, increase `k` parameter |
| Upload fails | Large file | Max 50MB, try smaller PDF |
| Can't connect to server | Port in use | Change port in `app.py` or kill process |

---

## 📊 Tech Stack

**Frontend:**
- HTML5 + CSS3 (modern, responsive)
- Vanilla JavaScript (no dependencies)

**Backend:**
- Python 3.8+
- Flask (web server)
- LangChain (LLM/embedding integration)
- Chroma (vector database)
- Ollama (local LLM backend)

**AI/ML:**
- Mistral (default LLM)
- nomic-embed-text (embeddings)
- BM25-style reranking

**Infrastructure:**
- Structured logging
- Configuration management
- Error handling & validation

---

## 📝 License & Attribution

This project is built with:
- LangChain framework
- Chroma vector database
- Ollama for local inference
- Mistral AI models

---

## 🤝 Support & Documentation

For detailed information, see:
- **[START_HERE.md](START_HERE.md)** - Project overview
- **[BACKEND_DOCUMENTATION.md](BACKEND_DOCUMENTATION.md)** - Technical details
- Inline code comments for implementation details

---

## 🎯 Use Cases

✅ **Corporate Knowledge Base** - Query internal documents  
✅ **Legal Document Review** - Search contracts and agreements  
✅ **Research & Academia** - Find citations and references  
✅ **Customer Support** - Build FAQ from documentation  
✅ **Medical Records** - Intelligent patient information retrieval  
✅ **Financial Analysis** - Document and report analysis

---

**DocRAG** - Making documents intelligent. 🚀
2. Use WSGI server (Gunicorn)
3. Enable authentication
4. Use persistent storage
5. Monitor logs

See [BACKEND_DOCUMENTATION.md](BACKEND_DOCUMENTATION.md) for details.

---

## 📞 Help & Support

- **Quick reference** → [QUICK_START.md](QUICK_START.md)
- **Technical details** → [BACKEND_DOCUMENTATION.md](BACKEND_DOCUMENTATION.md)
- **Understanding it** → [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
- **What changed** → [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)
- **Complete list** → [COMPLETION_CHECKLIST.md](COMPLETION_CHECKLIST.md)

---

## 📦 Dependencies

```
langchain
langchain-chroma
langchain-ollama
langchain-community
flask
```

Install with:
```bash
pip install langchain langchain-chroma langchain-ollama langchain-community flask
```

---

## 🎉 Ready to Go!

**Start using your enhanced RAG system:**

```bash
python app.py
# http://localhost:5000
```

For detailed guidance, read **[START_HERE.md](START_HERE.md)** first!

---

**Version**: 2.0 - Enhanced RAG Pipeline  
**Status**: ✅ Ready for production  
**Last Updated**: March 28, 2024" 
