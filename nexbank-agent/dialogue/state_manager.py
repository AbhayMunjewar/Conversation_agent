"""
%%{init: {'theme': 'dark'}}%%
stateDiagram-v2
    [*] --> greeting
    greeting --> intent_detected : Intent classified
    intent_detected --> slot_filling : Slots missing
    intent_detected --> slot_complete : All slots present
    slot_filling --> slot_filling : More slots missing
    slot_filling --> slot_complete : Final slot filled
    slot_complete --> ready_for_action : Handoff prepared
    ready_for_action --> resolved : Action execution (future block)
    
    greeting --> escalated : escalation_triggered (future block)
    intent_detected --> escalated : escalation_triggered (future block)
    slot_filling --> escalated : escalation_triggered (future block)
    slot_complete --> escalated : escalation_triggered (future block)
    ready_for_action --> escalated : escalation_triggered (future block)
    resolved --> escalated : escalation_triggered (future block)
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# Import NLU modules from Block 1 and 2
from nlu.taxonomy import get_required_slots
from nlu.classify import classify, ClassificationResult

logger = logging.getLogger(__name__)


class DialogueStatus(str, Enum):
    GREETING = "greeting"
    INTENT_DETECTED = "intent_detected"
    SLOT_FILLING = "slot_filling"
    SLOT_COMPLETE = "slot_complete"
    READY_FOR_ACTION = "ready_for_action"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class Turn(BaseModel):
    role: str  # "customer" | "bot"
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationState(BaseModel):
    conversation_id: str
    turn_history: List[Turn] = Field(default_factory=list)
    active_intent: Optional[str] = None
    active_domain: Optional[str] = None
    filled_slots: Dict[str, Any] = Field(default_factory=dict)
    missing_slots: List[str] = Field(default_factory=list)
    escalation_flags: List[str] = Field(default_factory=list)
    status: DialogueStatus = DialogueStatus.GREETING
    low_confidence_strikes: int = 0


class DialogueResponse(BaseModel):
    message_to_customer: str
    status: DialogueStatus
    active_intent: Optional[str] = None
    filled_slots: Dict[str, Any] = Field(default_factory=dict)
    missing_slots: List[str] = Field(default_factory=list)
    debug_trace: Dict[str, Any] = Field(default_factory=dict)


# Session Store Layer (Abstract Base Class for future swapability, e.g. Redis / Postgres)
class BaseSessionStore(ABC):
    @abstractmethod
    def get_or_create(self, conversation_id: str) -> ConversationState:
        """Retrieves an existing ConversationState or returns a blank new state."""
        pass

    @abstractmethod
    def save(self, state: ConversationState) -> None:
        """Saves the conversation state."""
        pass


class InMemorySessionStore(BaseSessionStore):
    """In-memory implementation of BaseSessionStore.
    
    NOTE: For production deployments, this should be replaced with a persistent,
    distributed storage backend such as Redis or PostgreSQL.
    """
    def __init__(self) -> None:
        self._store: Dict[str, ConversationState] = {}

    def get_or_create(self, conversation_id: str) -> ConversationState:
        if conversation_id not in self._store:
            self._store[conversation_id] = ConversationState(conversation_id=conversation_id)
        # Return a copy to simulate database read/write isolation
        return self._store[conversation_id].model_copy(deep=True)

    def save(self, state: ConversationState) -> None:
        self._store[state.conversation_id] = state.model_copy(deep=True)


class DialogueStateManager:
    def __init__(self, session_store: Optional[BaseSessionStore] = None) -> None:
        self.session_store = session_store or InMemorySessionStore()
        
        # Clarifying templates for slots
        self.slot_templates = {
            "amount": "Could you tell me the approximate amount involved?",
            "transaction_id": "Do you have the transaction reference number or the approximate date and merchant name?",
            "dispute_id_or_transaction_id": "Could you share the dispute ticket number or transaction reference ID?",
            "date": "What was the date of this transaction?",
            "card_type": "Are we blocking a debit card or a credit card?",
            "account_type": "Is this for a savings, salary, or current account?",
            "document_type": "Which document type are you updating (e.g. PAN, Aadhaar, address proof)?",
            "new_value": "Please provide the new details or value you wish to update.",
            "reason": "Could you tell me the reason for this request?"
        }

    def _generate_clarifying_question(self, slot_name: str) -> str:
        """Generates clarifying question using template or falls back to generic text."""
        return self.slot_templates.get(
            slot_name, 
            f"Could you share more details about {slot_name}?"
        )

    def process_turn(self, conversation_id: str, message: str) -> DialogueResponse:
        # a. Load state
        state = self.session_store.get_or_create(conversation_id)

        # b. Append user turn
        state.turn_history.append(Turn(role="customer", message=message))

        # c. Classify incoming message
        nlu_result: ClassificationResult = classify(message)

        # Track low-confidence classification strikes
        is_low_confidence = nlu_result.confidence < 0.6
        if is_low_confidence:
            state.low_confidence_strikes += 1
        else:
            state.low_confidence_strikes = 0

        # d. INTENT SWITCH HANDLING
        intent_switched = False
        if state.active_intent is not None:
            # Different intent detected with confidence > 0.6 triggers intent context switch
            if (nlu_result.intent is not None 
                    and nlu_result.intent != state.active_intent 
                    and nlu_result.confidence > 0.6):
                logger.info(
                    f"Context switch detected in session {conversation_id}: "
                    f"changing intent from '{state.active_intent}' to '{nlu_result.intent}'."
                )
                state.active_intent = nlu_result.intent
                state.active_domain = nlu_result.domain
                state.filled_slots = {}
                state.missing_slots = []
                state.status = DialogueStatus.INTENT_DETECTED
                intent_switched = True
        else:
            # First intent detection
            if nlu_result.intent is not None:
                state.active_intent = nlu_result.intent
                state.active_domain = nlu_result.domain
                state.status = DialogueStatus.INTENT_DETECTED

        # e. SLOT FILLING
        response_msg = ""
        if state.active_intent is not None:
            try:
                required = get_required_slots(state.active_intent)
            except KeyError:
                required = []

            # If not switching intent, or if we just switched to a new intent, merge entities
            # Skip merging if confidence is too low and we DID NOT switch (i.e. keep slot-filling going)
            # but still allow matching entities from user's slot response
            for slot_key, slot_val in nlu_result.entities.items():
                if slot_key in required:
                    state.filled_slots[slot_key] = slot_val

            # Recompute missing slots
            state.missing_slots = [s for s in required if s not in state.filled_slots]

            if state.missing_slots:
                state.status = DialogueStatus.SLOT_FILLING
                # Ask clarifying question for first missing slot
                response_msg = self._generate_clarifying_question(state.missing_slots[0])
            else:
                state.status = DialogueStatus.SLOT_COMPLETE
                # Transition to ready for action
                state.status = DialogueStatus.READY_FOR_ACTION
                
                # Check for metadata/system intents
                if state.active_intent == "general_greeting":
                    response_msg = "Hello! How can I help you today?"
                    state.status = DialogueStatus.RESOLVED
                elif state.active_intent == "general_farewell":
                    response_msg = "Thank you for banking with NexBank. Goodbye!"
                    state.status = DialogueStatus.RESOLVED
                elif state.active_intent == "escalation_request":
                    response_msg = "I am transferring you to a live customer support agent now."
                    state.status = DialogueStatus.ESCALATED
                else:
                    response_msg = (
                        f"I have gathered all the necessary details for your {state.active_intent} request. "
                        f"Details: {state.filled_slots}. Processing now..."
                    )
        else:
            # Fallback if no intent could be parsed
            response_msg = "I'm sorry, I didn't quite catch that. Could you please specify how I can help you today?"
            state.status = DialogueStatus.GREETING

        # Append bot turn
        state.turn_history.append(Turn(role="bot", message=response_msg))

        # Save and return
        self.session_store.save(state)

        return DialogueResponse(
            message_to_customer=response_msg,
            status=state.status,
            active_intent=state.active_intent,
            filled_slots=state.filled_slots,
            missing_slots=state.missing_slots,
            debug_trace={"classification": nlu_result.model_dump()}
        )
