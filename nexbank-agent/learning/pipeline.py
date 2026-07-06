import os
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

# Import miners
from learning.mine_corrections import mine_corrections
from learning.flag_kb_gaps import flag_kb_gaps

# Path setups
KB_DIR = Path(__file__).parent.parent
TAXONOMY_PATH = KB_DIR / "nlu" / "intent_taxonomy.json"
CACHE_PATH = KB_DIR / "nlu" / "embeddings_cache.pkl"
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOGS_DIR / "interactions.jsonl"
HELD_OUT_TEST_PATH = LOGS_DIR / "held_out_test_set.json"
RUN_REPORT_PATH = LOGS_DIR / "pipeline_run_report.json"
BACKUP_DIR = Path(__file__).parent / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def merge_updates(taxonomy_data: dict, updates: dict) -> dict:
    """Merges new training examples from updates dictionary into the taxonomy dictionary."""
    updated = json.loads(json.dumps(taxonomy_data))  # Deep copy
    intent_updates = updates.get("intents", {})
    
    for domain in updated.get("domains", []):
        for intent_obj in domain.get("intents", []):
            intent_name = intent_obj.get("intent")
            if intent_name in intent_updates:
                new_utterances = intent_updates[intent_name]
                for new_utt in new_utterances:
                    if new_utt not in intent_obj.get("example_utterances", []):
                        intent_obj["example_utterances"].append(new_utt)
    return updated


def evaluate_nlu_accuracy() -> float:
    """Evaluates NLU classification accuracy against the frozen held-out test set."""
    # Import locally to use current state of intent_taxonomy.json
    from nlu import taxonomy
    from nlu.classify import classify
    
    # Invalidate taxonomy cache to force reload
    taxonomy._taxonomy_instance = None
    
    # Load test set
    if not HELD_OUT_TEST_PATH.exists():
        raise FileNotFoundError(f"Held-out test set not found at: {HELD_OUT_TEST_PATH}")
        
    with open(HELD_OUT_TEST_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
        
    correct = 0
    total = len(test_cases)
    
    for case in test_cases:
        res = classify(case["text"])
        if res.intent == case["expected_intent"]:
            correct += 1
            
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def run_nightly_improvement_job() -> Dict[str, Any]:
    """Runs the full continuous learning job.
    
    Mines corrections, flags gaps, creates a candidate taxonomy, validates it, 
    and commits/promotes if all safety gates pass.
    """
    print("=" * 60)
    print("         STARTING NIGHTLY IMPROVEMENT PIPELINE JOB")
    print("=" * 60)
    
    # 1. Run Correction Mining and KB Gap Flagging
    print("Step 1: Running correction mining and KB gap detection...")
    proposed_updates = mine_corrections(LOG_PATH)
    flag_kb_gaps(LOG_PATH)
    
    # Check if there are updates to evaluate
    if not proposed_updates:
        summary = {
            "status": "skipped",
            "reason": "No proposed intent updates found in logs.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        with open(RUN_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("Skipped: No corrections mined. Current taxonomy is optimal.")
        return summary

    # Load current active taxonomy file
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        current_taxonomy = json.load(f)

    # 2. Compute Baseline Accuracy
    print("Step 2: Evaluating NLU baseline accuracy on held-out test set...")
    # Force delete embeddings cache to start fresh
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
        
    baseline_accuracy = evaluate_nlu_accuracy()
    print(f"  Baseline Accuracy: {baseline_accuracy * 100:.2f}%")

    # 3. Build Staged Candidate Taxonomy
    print("Step 3: Building candidate taxonomy...")
    candidate_taxonomy = merge_updates(current_taxonomy, proposed_updates)
    candidate_file = LOGS_DIR / "candidate_taxonomy.json"
    with open(candidate_file, "w", encoding="utf-8") as f:
        json.dump(candidate_taxonomy, f, indent=2)

    # 4. Apply Candidate Taxonomy to Temporary File Space for Validation Gates
    print("Step 4: Swapping taxonomy to Candidate and running validation gates...")
    # Create backup copy of original taxonomy
    temp_backup = LOGS_DIR / "temp_backup_intent_taxonomy.json"
    shutil.copyfile(TAXONOMY_PATH, temp_backup)
    
    # Overwrite the production taxonomy with the candidate
    shutil.copyfile(candidate_file, TAXONOMY_PATH)
    
    # Delete embeddings cache to force regeneration under the candidate taxonomy
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()

    # GATE check results variables
    guardrails_passed = False
    candidate_accuracy = 0.0
    regression_value = 0.0
    failure_reasons = []

    try:
        # A. Classification Accuracy Gate
        candidate_accuracy = evaluate_nlu_accuracy()
        regression_value = candidate_accuracy - baseline_accuracy
        print(f"  Candidate Accuracy: {candidate_accuracy * 100:.2f}% (Change: {regression_value * 100:+.2f}%)")
        
        # We allow a maximum regression threshold of -1.0%
        if regression_value < -0.01:
            failure_reasons.append(
                f"NLU classification accuracy regressed by {regression_value * 100:.2f}% "
                f"(Threshold: -1.0%)"
            )

        # B. Safety Guardrail Gate
        print("  Running adversarial guardrail test suite in clean process...")
        pytest_exe = str(KB_DIR / ".venv" / "Scripts" / "pytest.exe")
        adversarial_test_file = str(KB_DIR / "tests" / "test_guardrails_adversarial.py")
        
        # Run pytest as a clean subprocess to avoid module cache pollution
        run_res = subprocess.run([pytest_exe, "-q", adversarial_test_file], capture_output=True, text=True)
        guardrails_passed = (run_res.returncode == 0)
        
        if not guardrails_passed:
            failure_reasons.append("Safety Guardrail Adversarial test suite failed under the candidate taxonomy.")
            print("  [Gate Fail] Safety Guardrails regression detected!")
        else:
            print("  [Gate Pass] Safety Guardrails adversarial checks passed successfully.")

    except Exception as e:
        failure_reasons.append(f"Validation execution crashed: {e}")

    # Determine gate pass status
    gates_passed = (len(failure_reasons) == 0)

    # 5. Commit/Promote or Rollback
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_delta": regression_value,
        "guardrails_passed": guardrails_passed,
        "proposed_updates": proposed_updates,
        "gates_passed": gates_passed,
        "status": "promoted" if gates_passed else "rejected",
        "failure_reasons": failure_reasons
    }

    if gates_passed:
        print("\n" + "=" * 60)
        print("   SUCCESS: ALL GATES PASSED. PROMOTING CANDIDATE TAXONOMY.")
        print("=" * 60)
        
        # Save timestamped backup of the previous production version
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"intent_taxonomy_backup_{timestamp_str}.json"
        shutil.copyfile(temp_backup, backup_path)
        print(f"Saved historical backup to {backup_path.name}")
        
        # Delete temp backup since candidate is promoted
        if temp_backup.exists():
            temp_backup.unlink()
            
        # Re-clear in-memory instances
        from nlu import taxonomy
        taxonomy._taxonomy_instance = None
    else:
        print("\n" + "=" * 60)
        print("   REJECTED: SAFETY OR REGRESSION GATES FAILED. ROLLING BACK.")
        print("=" * 60)
        for reason in failure_reasons:
            print(f"  - {reason}")
            
        # Rollback: restore backup
        shutil.copyfile(temp_backup, TAXONOMY_PATH)
        if temp_backup.exists():
            temp_backup.unlink()
            
        # Delete invalid candidate embeddings cache
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
            
        # Re-clear taxonomy cache to point to original
        from nlu import taxonomy
        taxonomy._taxonomy_instance = None

    # Write summary run report
    with open(RUN_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Pipeline run report saved to {RUN_REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_nightly_improvement_job()
