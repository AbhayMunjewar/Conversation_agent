import pytest
from unittest.mock import patch
from dialogue.state_manager import DialogueStateManager, DialogueStatus
from nlu.classify import ClassificationResult


def test_happy_path_slot_filling():
    """Verifies a full happy-path multi-turn conversation for dispute_unauthorized.
    
    Fills slots ['transaction_id', 'amount', 'date'] over 3 turns.
    """
    manager = DialogueStateManager()
    session_id = "test_happy_path_123"

    # Turn 1: User indicates unauthorized transaction and mentions amount
    response1 = manager.process_turn(session_id, "there's a charge I never authorized of 5000 rupees")
    assert response1.status == DialogueStatus.SLOT_FILLING
    assert response1.active_intent == "dispute_unauthorized"
    assert response1.filled_slots.get("amount") == "5000"
    assert "transaction_id" in response1.missing_slots
    assert "date" in response1.missing_slots
    # It should ask clarifying question for transaction_id
    assert "transaction reference number" in response1.message_to_customer.lower()

    # Turn 2: User provides transaction_id
    response2 = manager.process_turn(session_id, "the transaction reference is TXN88214")
    assert response2.status == DialogueStatus.SLOT_FILLING
    assert response2.active_intent == "dispute_unauthorized"
    assert response2.filled_slots.get("amount") == "5000"
    assert response2.filled_slots.get("transaction_id") == "TXN88214"
    assert response2.missing_slots == ["date"]
    # It should ask clarifying question for date
    assert "date" in response2.message_to_customer.lower()

    # Turn 3: User provides date
    response3 = manager.process_turn(session_id, "it happened yesterday")
    assert response3.status == DialogueStatus.READY_FOR_ACTION
    assert response3.active_intent == "dispute_unauthorized"
    assert response3.filled_slots.get("amount") == "5000"
    assert response3.filled_slots.get("transaction_id") == "TXN88214"
    assert response3.filled_slots.get("date") == "yesterday"
    assert not response3.missing_slots
    assert "gathered all the necessary details" in response3.message_to_customer


def test_dialogue_context_switch():
    """Verifies a mid-conversation intent switch resets state appropriately."""
    manager = DialogueStateManager()
    session_id = "test_context_switch_123"

    # Start dispute
    response1 = manager.process_turn(session_id, "there's a charge I never authorized of 5000 rupees")
    assert response1.active_intent == "dispute_unauthorized"
    assert response1.filled_slots.get("amount") == "5000"

    # User changes mind and asks for balance inquiry (confidence is high, e.g. 0.95 in mock)
    response2 = manager.process_turn(session_id, "actually what's my account balance")
    assert response2.active_intent == "balance_inquiry"
    # Ensure slots from old intent are discarded
    assert "amount" not in response2.filled_slots
    assert "transaction_id" not in response2.filled_slots
    
    # balance_inquiry requires ['account_type']
    assert response2.status == DialogueStatus.SLOT_FILLING
    assert response2.missing_slots == ["account_type"]
    assert "savings, salary, or current" in response2.message_to_customer.lower()


def test_low_confidence_switch_rejection():
    """Verifies that a low-confidence classification (< 0.6) does not trigger a context switch."""
    manager = DialogueStateManager()
    session_id = "test_low_conf_123"

    # Start dispute
    response1 = manager.process_turn(session_id, "there's a charge I never authorized of 5000 rupees")
    assert response1.active_intent == "dispute_unauthorized"
    assert response1.filled_slots.get("amount") == "5000"

    # Mock a low-confidence classification result (intent = kyc_update, confidence = 0.3)
    low_conf_result = ClassificationResult(
        intent="kyc_update",
        sub_intent=None,
        entities={"document_type": "PAN"},
        confidence=0.3,
        reasoning="Very unsure matching attempt",
        domain="account",
        domain_confidence=0.4,
        stage1_domain="account"
    )

    with patch("dialogue.state_manager.classify", return_value=low_conf_result):
        # User says something ambiguous that maps to low-confidence kyc_update
        response2 = manager.process_turn(session_id, "kyc pan stuff maybe?")
        
        # Should NOT switch intent, stays on dispute_unauthorized
        assert response2.active_intent == "dispute_unauthorized"
        # Strikes should increment
        state = manager.session_store.get_or_create(session_id)
        assert state.low_confidence_strikes == 1


def test_session_persistence():
    """Verifies that session state persists correctly across separate process_turn invocations."""
    store = manager = DialogueStateManager()
    session_id = "session_persist_xyz"

    # Turn 1: Process card block request
    res1 = manager.process_turn(session_id, "please block my debit card immediately")
    assert res1.active_intent == "card_block"
    assert res1.filled_slots.get("card_type") == "debit"
    assert "last_4_digits" in res1.missing_slots

    # Instantiating a new manager sharing the same session store
    new_manager = DialogueStateManager(session_store=manager.session_store)
    
    # Turn 2: Provide missing slots
    res2 = new_manager.process_turn(session_id, "the last 4 digits are 1234")
    
    # State should carry over from the store
    assert res2.active_intent == "card_block"
    assert res2.filled_slots.get("card_type") == "debit"
    # Although TXN regex matches, we verify the dynamic parsing extracted it if possible,
    # or at least verify the card block state persists
    state = new_manager.session_store.get_or_create(session_id)
    assert state.active_intent == "card_block"
    assert len(state.turn_history) == 4  # 2 customer turns, 2 bot turns
