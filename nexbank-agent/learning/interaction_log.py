import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone

# Ensure output logs directory exists
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
INTERACTIONS_LOG_PATH = LOGS_DIR / "interactions.jsonl"


class SupervisorCorrection(BaseModel):
    corrected_intent: str
    corrected_response: str
    corrector_notes: str


class InteractionRecord(BaseModel):
    conversation_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    customer_message: str
    predicted_intent: str
    predicted_confidence: float
    final_response_text: str
    supervisor_correction: Optional[SupervisorCorrection] = None
    csat_score: Optional[int] = None  # Scale 1-5
    resolution_outcome: str  # resolved, escalated, reopened
    guardrail_triggers: List[str] = Field(default_factory=list)


def append_interaction(record: InteractionRecord) -> None:
    """Appends an InteractionRecord as a single JSON line in interactions.jsonl."""
    with open(INTERACTIONS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
