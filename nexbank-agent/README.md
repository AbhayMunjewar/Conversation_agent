# NexBank Agentic Customer Service System

This repository contains the production-grade agentic customer service system for NexBank. It is structured modularly to separate concerns across key conversational, business, and guardrail layers.

## Architecture Directory Layout

- **`nlu/`**: Natural Language Understanding. Manages the hierarchical intent taxonomy, parsing, validation, and intent/slot classification models.
- **`dialogue/`**: Dialogue management. Manages conversation context, state transitions, slot-filling flows, and response generation rules.
- **`kb/`**: Knowledge Base. Provides access to banking product guidelines, terms, conditions, and FAQs via retrieval-augmented interfaces.
- **`guardrails/`**: Safety and alignment layer. Evaluates user input and agent responses for compliance, security risks, sensitive fields, and output validation.
- **`escalation/`**: Human-in-the-loop transition. Routes high-risk queries, dissatisfied customers, or direct escalation requests to live support agents.
- **`learning/`**: Continuous evaluation. Gathers system telemetry, customer feedback, and classification logs for post-hoc analysis and training.
- **`api/`**: Integration interface. Serves endpoints for mobile apps, web interfaces, and third-party messaging integrations.
- **`tests/`**: Automated verification test suites.
