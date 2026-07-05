import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# Paths
KB_DIR = Path(__file__).parent
DOC_DIR = KB_DIR / "documents"
CHROMA_DIR = KB_DIR / "chroma_store"

def main():
    # Eagerly load the sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load all generated JSON documents
    doc_files = list(DOC_DIR.glob("kb_doc_*.json"))
    if not doc_files:
        raise FileNotFoundError(f"No document files found in {DOC_DIR}. Please run generate_documents.py first.")
    
    print(f"Found {len(doc_files)} documents to ingest.")
    
    ids = []
    documents = []
    metadatas = []
    
    for doc_file in doc_files:
        with open(doc_file, "r", encoding="utf-8") as f:
            doc_data = json.load(f)
            
        ids.append(doc_data["id"])
        documents.append(doc_data["content"])
        metadatas.append({
            "id": doc_data["id"],
            "title": doc_data["title"],
            "category": doc_data["category"],
            "product_type": doc_data["product_type"],
            "region": doc_data["region"]
        })
        
    # Generate embeddings in batch
    print("Generating sentence embeddings in batch...")
    raw_embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=False)
    embeddings = [emb.tolist() for emb in raw_embeddings]
    
    # Set up Persistent ChromaDB client
    print(f"Initializing persistent ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Enforce idempotency: clear and recreate collection
    collection_name = "nexbank_policy_kb"
    try:
        client.delete_collection(name=collection_name)
        print(f"Cleared existing collection '{collection_name}' for idempotency.")
    except Exception:
        # Collection didn't exist, which is fine
        pass
        
    collection = client.create_collection(name=collection_name)
    
    # Ingest in batches to prevent hitting any database buffers
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        end_idx = min(i + batch_size, len(ids))
        print(f"Adding batch {i // batch_size + 1} ({i} to {end_idx})...")
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=metadatas[i:end_idx],
            documents=documents[i:end_idx]
        )
        
    print(f"Ingestion successful! Loaded {len(ids)} documents into ChromaDB.")

if __name__ == "__main__":
    main()
