import json
from pathlib import Path
import pytest

from learning.mine_corrections import mine_corrections, UPDATES_PATH
from learning.flag_kb_gaps import flag_kb_gaps, GAP_REPORT_PATH
from learning.pipeline import run_nightly_improvement_job, RUN_REPORT_PATH

# Log directories
KB_DIR = Path(__file__).parent.parent
LOG_FILE = KB_DIR / "learning" / "logs" / "interactions.jsonl"


def test_correction_mining():
    """Verifies that supervisor corrections are correctly mined from the logs."""
    assert LOG_FILE.exists(), "Interactions log file does not exist. Run generator first."
    
    # Run corrections mining
    proposed = mine_corrections(LOG_FILE)
    
    # We expect 'dispute_unauthorized' to have proposed examples
    assert "dispute_unauthorized" in proposed
    assert len(proposed["dispute_unauthorized"]) >= 3
    assert UPDATES_PATH.exists()
    
    # Read staging file
    with open(UPDATES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "dispute_unauthorized" in data["intents"]


def test_kb_gap_flagging():
    """Verifies that low-CSAT queries are clustered and gaps are reported."""
    assert LOG_FILE.exists()
    
    # Run gap detection
    gaps = flag_kb_gaps(LOG_FILE)
    
    assert len(gaps) > 0
    assert any("reward points" in gap["pattern_summary"] for gap in gaps)
    assert GAP_REPORT_PATH.exists()


def test_promotion_pipeline():
    """Confirms that the nightly promotion job evaluates and promotes safely."""
    # Run the nightly improvement job
    report = run_nightly_improvement_job()
    
    # Since we use clean synthetic data, it should pass all validation checks and promote
    assert report["gates_passed"] is True
    assert report["status"] == "promoted"
    assert report["accuracy_delta"] >= -0.01
    assert report["guardrails_passed"] is True
    assert RUN_REPORT_PATH.exists()
