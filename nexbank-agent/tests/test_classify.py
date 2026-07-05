import pytest
from nlu.classify import classify

# Global test stats tracker
STATS = {
    "total": 0,
    "correct_domain": 0,
    "correct_intent": 0
}


@pytest.fixture(scope="module", autouse=True)
def print_accuracy_summary():
    """Module-level fixture to print the accuracy summary at the end of the test run."""
    yield
    total = STATS["total"]
    if total > 0:
        domain_acc = (STATS["correct_domain"] / total) * 100
        intent_acc = (STATS["correct_intent"] / total) * 100
        print("\n" + "=" * 60)
        print("             NLU CLASSIFIER ACCURACY SUMMARY")
        print("=" * 60)
        print(f"Total Test Cases Run: {total}")
        print(f"Domain Accuracy:      {domain_acc:.2f}% ({STATS['correct_domain']}/{total})")
        print(f"Intent Accuracy:      {intent_acc:.2f}% ({STATS['correct_intent']}/{total})")
        print("=" * 60)


# 15 test cases pulled directly from nlu/intent_taxonomy.json example_utterances
STANDARD_TEST_CASES = [
    # (message, expected_domain, expected_intent)
    # Account domain
    ("what's my account balance", "account", "balance_inquiry"),
    ("send me last 3 months statement", "account", "statement_request"),
    ("I need to update my KYC", "account", "kyc_update"),
    ("please block my debit card immediately", "account", "card_block"),
    ("I want to change my registered address", "account", "address_change"),
    ("I want to close my account", "account", "account_closure"),
    # Transaction domain
    ("there's a charge I never authorized", "transaction", "dispute_unauthorized"),
    ("I got charged twice for one purchase", "transaction", "dispute_duplicate"),
    ("money debited but not credited to receiver", "transaction", "dispute_failed_credit"),
    ("what's the status of my refund", "transaction", "refund_status"),
    # Product domain
    ("which savings account is best for me", "product", "recommend_savings"),
    ("what home loan options do you have", "product", "recommend_loan"),
    # Complaint domain
    ("your customer service is terrible", "complaint", "service_complaint"),
    # Advisory domain
    ("what is the current savings interest rate", "advisory", "interest_rate_query"),
    # Meta domain
    ("connect me to a human agent", "meta", "escalation_request")
]


@pytest.mark.parametrize("message,expected_domain,expected_intent", STANDARD_TEST_CASES)
def test_standard_cases(message: str, expected_domain: str, expected_intent: str):
    """Verifies taxonomy examples map to correct domain and intent."""
    STATS["total"] += 1
    result = classify(message)

    is_domain_correct = (result.domain == expected_domain)
    is_intent_correct = (result.intent == expected_intent)

    if is_domain_correct:
        STATS["correct_domain"] += 1
    if is_intent_correct:
        STATS["correct_intent"] += 1

    assert is_domain_correct, f"Expected domain '{expected_domain}', got '{result.domain}' for: {message}"
    assert is_intent_correct, f"Expected intent '{expected_intent}', got '{result.intent}' for: {message}"


# 3 ambiguous/mixed-intent messages with allowed primary intents
AMBIGUOUS_TEST_CASES = [
    (
        "someone stole my card and made a fraud transaction of 5000 rupees",
        ["card_block", "dispute_unauthorized", "fraud_report"],
        "User is reporting a stolen card (card_block) and an unauthorized transaction (dispute_unauthorized/fraud_report)."
    ),
    (
        "I want to close my savings account and transfer the remaining balance",
        ["account_closure", "balance_inquiry", "statement_request"],
        "User is requesting account closure (account_closure) combined with balance queries."
    ),
    (
        "what are the current fd interest rates and can I get a loan against it",
        ["interest_rate_query", "recommend_fixed_deposit", "recommend_loan"],
        "User inquires about interest rates/FD options and borrowing against them."
    )
]


@pytest.mark.parametrize("message,allowed_intents,reason_note", AMBIGUOUS_TEST_CASES)
def test_ambiguous_cases(message: str, allowed_intents: list, reason_note: str):
    """Verifies that ambiguous queries map to one of the expected/reasonable candidate intents."""
    STATS["total"] += 1
    result = classify(message)

    is_intent_correct = (result.intent in allowed_intents)
    
    # For ambiguous inputs, if the intent matched an expected candidate, count domain and intent as correct
    if is_intent_correct:
        STATS["correct_intent"] += 1
        STATS["correct_domain"] += 1

    print(f"\n[Ambiguous Case] Message: '{message}'")
    print(f"  Reason Note:       {reason_note}")
    print(f"  Classified Intent: {result.intent} (Confidence: {result.confidence:.2f})")
    print(f"  Classified Domain: {result.domain}")
    print(f"  Reasoning:         {result.reasoning}")

    assert is_intent_correct, (
        f"Ambiguous query '{message}' classified as '{result.intent}', "
        f"which is not in the allowed list: {allowed_intents}"
    )
