import re
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Setup logger for JSON log lines
logger = logging.getLogger("guardrails.checker")
logger.setLevel(logging.WARNING)

# Ensure console logging format is clean (unformatted since we print custom JSON lines)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)


class Rule(BaseModel):
    rule_id: str
    description: str
    trigger_patterns: List[str] = Field(default_factory=list)
    trigger_intents: List[str] = Field(default_factory=list)
    action: str
    severity: str
    reason_code: str


class RulesCollection(BaseModel):
    pre_check_rules: List[Rule]
    post_check_rules: List[Rule]


class PreCheckResult(BaseModel):
    triggered_rules: List[str]
    action: str
    reason_codes: List[str]
    allow_llm_free_response: bool


class PostCheckResult(BaseModel):
    triggered_rules: List[str]
    action: str
    modified_response: Optional[str]
    reason_codes: List[str]


# Load rules.yaml on import (fail loudly if schema is invalid)
RULES_YAML_PATH = Path(__file__).parent / "rules.yaml"

try:
    with open(RULES_YAML_PATH, "r", encoding="utf-8") as f:
        rules_data = yaml.safe_load(f)
    _rules = RulesCollection.model_validate(rules_data)
except Exception as e:
    raise RuntimeError(f"Failed to load or validate safety guardrails configuration: {e}") from e


def _log_guardrail_trigger(conversation_id: str, rule: Rule, action_taken: str) -> None:
    """Logs the guardrail event as a structured JSON line."""
    log_event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
        "rule_id": rule.rule_id,
        "reason_code": rule.reason_code,
        "severity": rule.severity,
        "action_taken": action_taken
    }
    logger.warning(json.dumps(log_event))


def pre_check(message: str, detected_intent: Optional[str] = None, conversation_id: str = "default_session") -> PreCheckResult:
    """Evaluates safety checks BEFORE the LLM makes any free-text decision.
    
    Checks regex patterns against the customer's raw message and evaluates the NLU-detected intent.
    """
    triggered_rules = []
    reason_codes = []
    matched_rules: List[Rule] = []

    for rule in _rules.pre_check_rules:
        # Check intent triggers
        intent_matched = detected_intent is not None and detected_intent in rule.trigger_intents

        # Check regex pattern triggers
        pattern_matched = False
        for pattern in rule.trigger_patterns:
            try:
                if re.search(pattern, message):
                    pattern_matched = True
                    break
            except re.error as e:
                # Log invalid patterns but continue
                logging.error(f"Invalid regex pattern in rule {rule.rule_id}: {pattern}. Error: {e}")

        if intent_matched or pattern_matched:
            triggered_rules.append(rule.rule_id)
            reason_codes.append(rule.reason_code)
            matched_rules.append(rule)
            
            # Log the trigger
            _log_guardrail_trigger(conversation_id, rule, rule.action)

    if not triggered_rules:
        return PreCheckResult(
            triggered_rules=[],
            action="allow",
            reason_codes=[],
            allow_llm_free_response=True
        )

    # Action priority: route_to_human (3) > require_stepped_up_auth (2) > allow_with_logging (1)
    action_priority = {"route_to_human": 3, "require_stepped_up_auth": 2, "allow_with_logging": 1}
    resolved_action = max(
        (rule.action for rule in matched_rules),
        key=lambda act: action_priority.get(act, 0)
    )

    # allow_llm_free_response is false if we must block/reroute the flow
    # (i.e. require stepped up auth or human routing)
    allow_llm_free_response = resolved_action == "allow_with_logging"

    return PreCheckResult(
        triggered_rules=triggered_rules,
        action=resolved_action,
        reason_codes=reason_codes,
        allow_llm_free_response=allow_llm_free_response
    )


def _contains_val_in_data(val: str, known_data: Any) -> bool:
    """Helper to check if a string representation of a value exists in known data structure."""
    if isinstance(known_data, dict):
        return any(_contains_val_in_data(val, v) for v in known_data.values())
    elif isinstance(known_data, list):
        return any(_contains_val_in_data(val, item) for item in known_data)
    else:
        # Standardize strings, clean whitespaces and punctuation to match accurately
        clean_val = re.sub(r'[-\s]', '', str(val))
        clean_known = re.sub(r'[-\s]', '', str(known_data))
        return clean_val == clean_known and len(clean_val) > 0


def post_check(response_text: str, conversation_id: str, known_customer_data: dict) -> PostCheckResult:
    """Evaluates safety checks on the LLM's generated response BEFORE showing it to the customer.
    
    Verifies financial recommendations, disclaimer triggers, and flags potential third-party PII leaks.
    """
    triggered_rules = []
    reason_codes = []
    matched_rules: List[Rule] = []

    for rule in _rules.post_check_rules:
        pattern_matched = False
        
        # Heuristic PII leak verification:
        # Match accounts or cards shapes in the text, then check if they exist in known user profile
        if rule.rule_id == "pii_leak_pattern":
            # Limitation Note: This matches numbers shaped like credit cards or account IDs.
            # It does not constitute perfect Named Entity Recognition (NER), but serves as a regex heuristic.
            for pattern in rule.trigger_patterns:
                for match in re.finditer(pattern, response_text):
                    matched_val = match.group(0)
                    # If this number does NOT exist in the known customer records, flag as PII leak!
                    if not _contains_val_in_data(matched_val, known_customer_data):
                        pattern_matched = True
                        break
                if pattern_matched:
                    break
        else:
            # Standard pattern checking
            for pattern in rule.trigger_patterns:
                try:
                    if re.search(pattern, response_text):
                        pattern_matched = True
                        break
                except re.error as e:
                    logging.error(f"Invalid regex in post-rule {rule.rule_id}: {pattern}. Error: {e}")

        if pattern_matched:
            triggered_rules.append(rule.rule_id)
            reason_codes.append(rule.reason_code)
            matched_rules.append(rule)
            _log_guardrail_trigger(conversation_id, rule, rule.action)

    if not triggered_rules:
        return PostCheckResult(
            triggered_rules=[],
            action="allow",
            modified_response=response_text,
            reason_codes=[]
        )

    # Action priority: block_hard (3) > block_and_regenerate (2) > append_disclaimer (1)
    action_priority = {"block_hard": 3, "block_and_regenerate": 2, "append_disclaimer": 1}
    resolved_action = max(
        (rule.action for rule in matched_rules),
        key=lambda act: action_priority.get(act, 0)
    )

    modified_response = response_text
    if resolved_action in ["block_hard", "block_and_regenerate"]:
        modified_response = None
    elif resolved_action == "append_disclaimer":
        disclaimer = (
            " Disclaimer: NexBank does not provide personalized tax or legal advice. "
            "Please consult a qualified tax professional or legal advisor for your specific situation."
        )
        # Only append if not already present
        if disclaimer.strip() not in response_text:
            modified_response = response_text.rstrip() + disclaimer

    return PostCheckResult(
        triggered_rules=triggered_rules,
        action=resolved_action,
        modified_response=modified_response,
        reason_codes=reason_codes
    )
