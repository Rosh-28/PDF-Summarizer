import argparse
import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from backend.get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "PDFs/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    print(f"[INFO] Documents loaded: {len(documents)}")
    if len(documents) == 0:
        print("[WARN] No documents were loaded. Check DATA_PATH and PDF files.")
        return

    chunks = split_documents(documents)
    print(f"[INFO] Total chunks created: {len(chunks)}")
    if len(chunks) == 0:
        print("[WARN] No chunks created from documents. PDFs may be scanned images or empty.")
        return

    # show sample document metadata for debugging
    #print_sample_info(documents, chunks)

    add_to_chroma(chunks)


def load_documents():
    data_dir = Path(DATA_PATH)
    if not data_dir.exists():
        print(f"[ERROR] DATA_PATH does not exist: {data_dir.resolve()}")
        return []

    files = sorted(data_dir.glob("*.pdf"))
    print(f"[INFO] PDF files found: {[f.name for f in files]}")
    all_docs = []
    for pdf_file in files:
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            docs = loader.load()
            if not docs:
                print(f"[WARN] No text extracted from: {pdf_file.name}")
            else:
                # add source metadata if missing
                for d in docs:
                    if "source" not in d.metadata:
                        d.metadata["source"] = str(pdf_file.name)
                all_docs.extend(docs)
                print(f"[INFO] Loaded {len(docs)} docs from {pdf_file.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load {pdf_file.name}: {e}")
    return all_docs


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    chunk_with_ids = calculate_chunk_ids(chunks)

    # Get existing metadatas safely
    existing_ids_result = db.get(include=["metadatas", "documents"])
    existing_ids = set()
    for metadata in existing_ids_result.get("metadatas", []):
        if isinstance(metadata, dict) and "id" in metadata:
            existing_ids.add(metadata["id"])

    print(f"Number of existing documents in DB:  {len(existing_ids)}")

    # Debug: show first 5 existing ids
    if len(existing_ids) > 0:
        print("Sample existing IDs:", list(existing_ids)[:5])

    new_chunks = []
    for chunk in chunk_with_ids:
        cid = chunk.metadata.get("id")
        if cid is None:
            print("[WARN] chunk has no id metadata; skipping chunk")
            continue
        if cid not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source_raw = chunk.metadata.get("source", "unknown")
        # normalize to basename to avoid absolute path mismatches
        source = os.path.basename(source_raw)
        page = chunk.metadata.get("page", "0")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


# def print_sample_info(documents, chunks):
#     print("--- SAMPLE DOCUMENT METADATA ---")
#     for i, d in enumerate(documents[:3]):
#         print(f"Doc {i}: source={d.metadata.get('source')}, length={len(d.page_content)}")
#     print("--- SAMPLE CHUNKS ---")
#     for i, c in enumerate(chunks[:5]):
#         print(f"Chunk {i}: id={c.metadata.get('id')}, source={c.metadata.get('source')}, len={len(c.page_content)}")


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
