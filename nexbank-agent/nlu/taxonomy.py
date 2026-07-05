import json
from pathlib import Path
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field, field_validator

# Pydantic v2 models for the intent taxonomy schema
class Intent(BaseModel):
    intent: str
    description: str
    required_slots: List[str]
    guardrail_notes: str
    example_utterances: List[str]

    @field_validator("description")
    @classmethod
    def validate_non_empty_description(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("description must be a non-empty string")
        return v

    @field_validator("example_utterances")
    @classmethod
    def validate_example_utterances_count(cls, v: List[str]) -> List[str]:
        if len(v) < 5:
            raise ValueError(f"must have at least 5 example utterances, got {len(v)}")
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("example utterance must be a non-empty string")
        return v


class Domain(BaseModel):
    domain: str
    description: str
    intents: List[Intent]

    @field_validator("domain", "description")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("domain and description fields must not be empty")
        return v


class Taxonomy(BaseModel):
    taxonomy_version: str
    bank: str
    notes: str
    domains: List[Domain]


# Path to the JSON taxonomy configuration
TAXONOMY_JSON_PATH = Path(__file__).parent / "intent_taxonomy.json"

# Globally cached/validated Taxonomy instance
_taxonomy_instance: Optional[Taxonomy] = None


def load_taxonomy(path: Path = TAXONOMY_JSON_PATH) -> Taxonomy:
    """Loads and strictly validates the intent taxonomy JSON file against the Pydantic schema."""
    global _taxonomy_instance
    if _taxonomy_instance is not None and path == TAXONOMY_JSON_PATH:
        return _taxonomy_instance

    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found at: {path.absolute()}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON format in taxonomy file: {e}") from e

    # This validates the complete hierarchical schema, failing loudly if invalid
    taxonomy = Taxonomy.model_validate(data)
    
    if path == TAXONOMY_JSON_PATH:
        _taxonomy_instance = taxonomy
    return taxonomy


# Initial eager validation on import
try:
    load_taxonomy()
except Exception as e:
    # Fail loudly on import if taxonomy schema validation fails
    raise RuntimeError(f"Failed to initialize intent taxonomy layer: {e}") from e


def list_domains() -> List[str]:
    """Returns a list of all domain names in the taxonomy."""
    taxonomy = load_taxonomy()
    return [d.domain for d in taxonomy.domains]


def list_intents(domain: Optional[str] = None) -> List[str]:
    """Returns a list of intent names. Optionally filtered by domain name."""
    taxonomy = load_taxonomy()
    if domain is not None:
        matched_domain = next((d for d in taxonomy.domains if d.domain == domain), None)
        if not matched_domain:
            raise KeyError(f"Domain '{domain}' not found in taxonomy.")
        return [i.intent for i in matched_domain.intents]
    
    # Return all intents
    all_intents = []
    for d in taxonomy.domains:
        all_intents.extend([i.intent for i in d.intents])
    return all_intents


def get_intent(name: str) -> Intent:
    """Retrieves an Intent object by its name. Raises KeyError if not found."""
    taxonomy = load_taxonomy()
    for d in taxonomy.domains:
        for i in d.intents:
            if i.intent == name:
                return i
    raise KeyError(f"Intent '{name}' not found in taxonomy.")


def get_required_slots(intent: str) -> List[str]:
    """Retrieves the list of required slots for the specified intent name."""
    intent_obj = get_intent(intent)
    return intent_obj.required_slots


def flatten_training_examples() -> List[Tuple[str, str, str]]:
    """Flattens all training utterances from the taxonomy.
    
    Returns:
        List of tuples: (utterance, domain, intent)
    """
    taxonomy = load_taxonomy()
    examples = []
    for d in taxonomy.domains:
        for i in d.intents:
            for utterance in i.example_utterances:
                examples.append((utterance, d.domain, i.intent))
    return examples
