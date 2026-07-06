import json
from pathlib import Path
from typing import Dict, List
from nlu.taxonomy import load_taxonomy

# Log output directory config
OUTPUT_DIR = Path(__file__).parent / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPDATES_PATH = OUTPUT_DIR / "proposed_taxonomy_updates.json"


def mine_corrections(log_path: Path) -> Dict[str, List[str]]:
    """Identifies recurring supervisor intent corrections in the logs.
    
    Proposes new training examples for intents with 3+ corrections, 
    staged to proposed_taxonomy_updates.json.
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Interaction logs file not found at: {log_path}")

    # Load existing taxonomy to check duplicates
    taxonomy = load_taxonomy()
    
    # Map intent names to their existing examples (lowercased & stripped for deduplication)
    existing_examples = {}
    for domain in taxonomy.domains:
        for intent_obj in domain.intents:
            existing_examples[intent_obj.intent] = {
                utt.lower().strip() for utt in intent_obj.example_utterances
            }

    # Gather candidate corrections
    candidates_by_intent: Dict[str, List[str]] = {}
    
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                correction = record.get("supervisor_correction")
                if correction is not None:
                    corrected_intent = correction.get("corrected_intent")
                    customer_msg = record.get("customer_message")
                    
                    if corrected_intent and customer_msg:
                        # Normalize string
                        clean_msg = customer_msg.strip()
                        norm_msg = clean_msg.lower()
                        
                        # Verify intent exists in taxonomy
                        if corrected_intent not in existing_examples:
                            continue
                            
                        # Deduplicate against existing training utterances
                        if norm_msg in existing_examples[corrected_intent]:
                            continue
                            
                        # Add to candidates list
                        if corrected_intent not in candidates_by_intent:
                            candidates_by_intent[corrected_intent] = []
                        
                        # Prevent duplicates inside the candidates list itself
                        if clean_msg not in candidates_by_intent[corrected_intent]:
                            candidates_by_intent[corrected_intent].append(clean_msg)
            except Exception as e:
                # Log reading error but continue parsing other lines
                print(f"Error parsing log line: {e}")

    # Filter to only keep intents with 3+ corrections clustered
    proposed_updates = {}
    for intent, list_of_msgs in candidates_by_intent.items():
        if len(list_of_msgs) >= 3:
            proposed_updates[intent] = list_of_msgs

    # Save to staging file
    updates_wrapper = {"intents": proposed_updates}
    with open(UPDATES_PATH, "w", encoding="utf-8") as f:
        json.dump(updates_wrapper, f, indent=2)
        
    print(f"Staged {sum(len(v) for v in proposed_updates.values())} proposed training examples for review across {len(proposed_updates)} intents.")
    
    return proposed_updates


if __name__ == "__main__":
    log_file = Path(__file__).parent / "logs" / "interactions.jsonl"
    mine_corrections(log_file)
