import random
from pathlib import Path
from datetime import datetime, timezone, timedelta

from learning.interaction_log import InteractionRecord, SupervisorCorrection, append_interaction, INTERACTIONS_LOG_PATH

# NOTE: This synthetic generator exists strictly to populate logs for demonstration 
# and testing purposes in the absence of real production data.

CORRECTED_UTTERANCES = [
    "unauthorized transaction on my debit card of 2000 rs",
    "stolen card charged 5000 without my otp",
    "some unknown merchant charged me 1500 rupees",
    "who debited 1200 rupees from my savings account",
    "report fraud transaction on credit card",
    "money stolen from my account via online transfer",
    "i did not make this 3000 rs purchase on card",
    "fraudulent charges on my credit statement",
    "unauthorized swipe at merchant store yesterday",
    "investigate unrecognized debit of 800 rupees",
    "someone used my card details online without permission",
    "fraudulent debit on my savings statement",
    "i see an unauthorized billing of 2500 rs",
    "please file dispute for unauthorized charge of 4000",
    "unrecognized transaction on my salary account"
]

KB_GAP_UTTERANCES = [
    "how do I redeem my credit card reward points",
    "reward points catalog details",
    "what is the value of reward points in rupees",
    "do my credit card reward points expire",
    "how to convert reward points to cash",
    "points redemption process",
    "how many rewards points do I get per transaction",
    "redeeming reward points for vouchers",
    "rewards balance update timeline",
    "points balance check on mobile app",
    "maximum reward points limit per month",
    "earning rates for reward points",
    "can I transfer reward points to airline miles",
    "rewards club membership details",
    "failed to redeem reward points on web portal",
    "minimum points required for redemption",
    "fees for redeeming reward points",
    "reward points not credited for purchase",
    "cashback vs reward points choice",
    "validity period of reward points"
]

STANDARD_INTENTS = [
    ("balance_inquiry", "What is my account balance?", "Your current balance is Rs. 15,420."),
    ("statement_request", "Send my statement for last month.", "We have sent your monthly statement to your registered email."),
    ("card_block", "I lost my debit card, please block it.", "Your debit card has been successfully blocked. A replacement has been ordered."),
    ("kyc_update", "How do I update my KYC?", "Please upload your Aadhaar/PAN documents via our secure profile verification portal."),
    ("general_greeting", "Hello there, good morning.", "Hello! Welcome to NexBank. How can I help you today?"),
    ("general_farewell", "Thank you, goodbye.", "Thank you for banking with NexBank. Have a great day!"),
    ("interest_rate_query", "What is the FD interest rate?", "Our current 1-year Fixed Deposit interest rate is 6.5% per annum.")
]


def generate_logs():
    # If the file already exists, clear it for a clean, deterministic synthetic run
    if INTERACTIONS_LOG_PATH.exists():
        INTERACTIONS_LOG_PATH.unlink()

    base_time = datetime.now(timezone.utc) - timedelta(days=5)

    print(f"Generating 300 synthetic interaction records in {INTERACTIONS_LOG_PATH}...")

    # 1. Generate 15 Supervisor Corrections (weakness simulation)
    for i, msg in enumerate(CORRECTED_UTTERANCES):
        timestamp = (base_time + timedelta(hours=i * 2)).isoformat()
        record = InteractionRecord(
            conversation_id=f"conv_corr_{i:03d}",
            timestamp=timestamp,
            customer_message=msg,
            predicted_intent="balance_inquiry",  # System weakness: misclassifying disputes
            predicted_confidence=0.42,
            final_response_text="Your balance is Rs. 10,000.",
            supervisor_correction=SupervisorCorrection(
                corrected_intent="dispute_unauthorized",
                corrected_response="I will file a dispute for this unrecognized charge of Rs. 5,000.",
                corrector_notes="System misclassified unauthorized dispute as balance inquiry."
            ),
            csat_score=2,
            resolution_outcome="resolved",
            guardrail_triggers=[]
        )
        append_interaction(record)

    # 2. Generate 20 Low-CSAT KB Gaps
    for i, msg in enumerate(KB_GAP_UTTERANCES):
        timestamp = (base_time + timedelta(days=1, hours=i)).isoformat()
        record = InteractionRecord(
            conversation_id=f"conv_gap_{i:03d}",
            timestamp=timestamp,
            customer_message=msg,
            predicted_intent="recommend_card",
            predicted_confidence=0.75,
            final_response_text="Sorry, I cannot find any policy guidelines about reward points redemption in the NexBank policy database.",
            supervisor_correction=None,
            csat_score=1,  # Critical low CSAT
            resolution_outcome="escalated",
            guardrail_triggers=[]
        )
        append_interaction(record)

    # 3. Generate 265 standard interactions
    for i in range(265):
        timestamp = (base_time + timedelta(days=2, minutes=i * 15)).isoformat()
        intent, msg, resp = random.choice(STANDARD_INTENTS)
        
        # Random parameters
        confidence = round(random.uniform(0.65, 0.99), 2)
        csat = random.choice([4, 5, 5, 3])  # mostly high CSAT
        outcome = "resolved" if csat >= 3 else "escalated"
        
        triggers = []
        if intent == "card_block":
            triggers = ["card_block_action"]
            
        record = InteractionRecord(
            conversation_id=f"conv_std_{i:03d}",
            timestamp=timestamp,
            customer_message=msg,
            predicted_intent=intent,
            predicted_confidence=confidence,
            final_response_text=resp,
            supervisor_correction=None,
            csat_score=csat,
            resolution_outcome=outcome,
            guardrail_triggers=triggers
        )
        append_interaction(record)

    print("Synthetic logs generation completed successfully.")


if __name__ == "__main__":
    generate_logs()
