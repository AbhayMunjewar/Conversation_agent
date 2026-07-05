import pytest
import sys
from nlu.taxonomy import load_taxonomy, list_intents, get_intent

def test_taxonomy_loads_without_error():
    """Verify that taxonomy can be successfully parsed and validated against the Pydantic schema."""
    taxonomy = load_taxonomy()
    assert taxonomy is not None
    assert taxonomy.bank == "NexBank"
    assert len(taxonomy.domains) > 0

def test_total_intents_count():
    """Confirm there are 25+ total intents in the taxonomy."""
    intents = list_intents()
    assert len(intents) >= 25, f"Expected 25+ intents, but got {len(intents)}"

def test_every_intent_has_min_examples():
    """Confirm every intent in the taxonomy has at least 5 example utterances."""
    taxonomy = load_taxonomy()
    for domain in taxonomy.domains:
        for intent in domain.intents:
            assert len(intent.example_utterances) >= 5, (
                f"Intent '{intent.intent}' in domain '{domain.domain}' has only "
                f"{len(intent.example_utterances)} examples, expected at least 5."
            )

def test_sensitive_guardrails_summary():
    """Identify and flag intents with sensitive guardrail notes for manual review.
    
    Keywords to match: 'escalat', 'never', 'always', 'step-up', or 'auth'.
    """
    sensitive_keywords = ["escalat", "never", "always", "step-up", "auth"]
    taxonomy = load_taxonomy()
    flagged_intents = []

    for domain in taxonomy.domains:
        for intent in domain.intents:
            note_lower = intent.guardrail_notes.lower()
            if any(keyword in note_lower for keyword in sensitive_keywords):
                flagged_intents.append((domain.domain, intent.intent, intent.guardrail_notes))

    # Print summary block
    print("\n" + "=" * 80)
    print("SENSITIVE INTENT GUARDRAILS SUMMARY FOR MANUAL REVIEW")
    print("=" * 80)
    for domain_name, intent_name, note in flagged_intents:
        print(f"Domain: {domain_name:<15} | Intent: {intent_name:<30}")
        print(f"  Guardrail Note: {note}")
        print("-" * 80)
    print(f"Total flagged sensitive intents: {len(flagged_intents)}")
    print("=" * 80)

    # Always succeeds, but prints output during pytest runs (run with -s flag to view)
    assert len(flagged_intents) > 0
