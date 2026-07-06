import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

# Output paths
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
GAP_REPORT_PATH = LOGS_DIR / "kb_gap_report.json"


def flag_kb_gaps(log_path: Path, csat_threshold: int = 2) -> List[Dict[str, Any]]:
    """Clusters low-CSAT user queries to find gaps in the knowledge base.
    
    Uses sentence embeddings similarity to group similar queries. Clusters of size 5+ 
    are outputted to kb_gap_report.json for human review.
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Interaction logs file not found at: {log_path}")

    low_csat_records = []
    
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                csat = record.get("csat_score")
                if csat is not None and csat <= csat_threshold:
                    low_csat_records.append(record)
            except Exception as e:
                print(f"Error parsing log line: {e}")

    if not low_csat_records:
        print("No low-CSAT records found.")
        # Write empty report
        with open(GAP_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return []

    # Extract messages and embed them using the same Model as the KB
    messages = [r["customer_message"] for r in low_csat_records]
    
    print(f"Embedding {len(messages)} low-CSAT messages for KB gap clustering...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    raw_embeddings = model.encode(messages, show_progress_bar=False, convert_to_numpy=True)
    
    # Normalize vectors for fast cosine similarity via dot product
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    embeddings = raw_embeddings / (norms + 1e-9)

    # Perform basic leader-based clustering
    clusters = []  # list of lists of indices
    similarity_threshold = 0.60

    for i, emb in enumerate(embeddings):
        placed = False
        for cluster in clusters:
            rep_idx = cluster[0]
            # Since vectors are normalized, dot product is cosine similarity
            sim = np.dot(emb, embeddings[rep_idx])
            if sim >= similarity_threshold:
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    # Filter clusters with 5+ occurrences
    gaps = []
    for cluster in clusters:
        if len(cluster) >= 5:
            rep_record = low_csat_records[cluster[0]]
            conversation_ids = [low_csat_records[idx]["conversation_id"] for idx in cluster]
            
            gaps.append({
                "pattern_summary": rep_record["customer_message"],
                "example_conversation_ids": conversation_ids,
                "count": len(cluster)
            })

    # Save to report path
    with open(GAP_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(gaps, f, indent=2)
        
    print(f"Identified {len(gaps)} knowledge base gaps with 5+ occurrences. Report written to {GAP_REPORT_PATH}.")
    return gaps


if __name__ == "__main__":
    log_file = Path(__file__).parent / "logs" / "interactions.jsonl"
    flag_kb_gaps(log_file)
