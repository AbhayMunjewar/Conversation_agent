# SmartAssist AI: NexBank Agentic Customer Service System

SmartAssist AI is a production-grade, self-improving agentic customer service system designed for **NexBank**—a digital neo-bank serving 2.7 million active customers in India. It is engineered to handle up to 18,000 daily customer interactions across account inquiries, transactions, disputes, recommendations, and complaint resolutions.

The core architecture separates NLU, dialogue orchestration, policy lookup, safety enforcement, and continuous optimization into dedicated modular layers.

---

## 🏛 Directory Architecture

The repository is structured as a multi-layered conversational pipeline:

*   **[`api/`](file:///d:/conversational_system/nexbank-agent/api)**: Integrated FastAPI server exposing chat and system health endpoints, alongside the single-page premium UI console dashboard.
*   **[`nlu/`](file:///d:/conversational_system/nexbank-agent/nlu)**: Natural Language Understanding. Implements two-stage intent routing (Stage 1 embedding-based domain routing, Stage 2 structured intent classification and slot-filling with retry mechanisms).
*   **[`dialogue/`](file:///d:/conversational_system/nexbank-agent/dialogue)**: Dialogue manager. Controls state transitions, tracks slot completion, and logs telemetry.
*   **[`kb/`](file:///d:/conversational_system/nexbank-agent/kb)**: Knowledge Base. Retrieves grounded banking policies from a persistent vector store using embedded similarity and in-memory LRU caching.
*   **[`guardrails/`](file:///d:/conversational_system/nexbank-agent/guardrails)**: Safety and compliance layer. Deterministic, non-LLM pre-check and post-check rules to catch restricted queries, PII leakage, or unauthorized actions.
*   **[`escalation/`](file:///d:/conversational_system/nexbank-agent/escalation)**: Human-in-the-loop transition framework. Evaluates 11 distinct triggers to safely route high-risk, low-confidence, or sensitive situations to the live support Agent Queue.
*   **[`learning/`](file:///d:/conversational_system/nexbank-agent/learning)**: Continuous self-improvement pipeline. Captures customer telemetry and supervisor feedback to safely optimize prompts and intent taxonomies, subject to guardrail regression testing.
*   **[`tests/`](file:///d:/conversational_system/nexbank-agent/tests)**: Complete verification suite including 84 unit and integration tests.

---

## 💻 Technology Stack

*   **Web Framework & API Core**: [FastAPI](https://fastapi.tiangolo.com/) + Pydantic (data parsing & schema validation) + [Uvicorn](https://www.uvicorn.org/) (ASGI server).
*   **Vector Database & Embeddings**: [ChromaDB](https://www.trychroma.com/) (persistent storage) + `SentenceTransformers` (`all-MiniLM-L6-v2` local embeddings model).
*   **Conversational Reasoning LLM**: [Anthropic Claude 3.5 Sonnet](https://www.anthropic.com/claude) (via structured tool-calling schemas).
*   **Deterministic Safety Gateways**: Regular expressions & strict token checkers.
*   **Diagnostic UI Dashboard**: Single-page vanilla HTML5 / CSS3 / ES6 Javascript client served directly via FastAPI static mounts.
*   **Package Management**: `poetry` / virtualenv environments.

---

## 🚨 Human-in-the-Loop Handoff (11 Triggers)

The system continuously audits dialogue states and escalates to a human agent when any of the following 11 triggers are met:

1.  **Explicit User Handoff Request**: Direct commands (e.g., *"connect to human"*).
2.  **Pre-Check Safety Human Redirect**: Triggered by critical actions requiring human approval (e.g., account closure requests).
3.  **Post-Check Safety Hard Block**: Activated when answers contain critical security issues (e.g., PII leak attempt).
4.  **Consecutive Compliance Blocks**: Two consecutive safety overrides.
5.  **High-Value Financial Dispute**: Any disputed transaction amount exceeding ₹50,000.
6.  **NLU Confidence Strike Limit**: Three consecutive classification turns falling below the 60% confidence threshold.
7.  **Swearing, Abuse or Ombudsman Threats**: Detection of legal warnings or severe negative sentiment.
8.  **Repetitive Slot Failure**: When the bot fails to resolve the same required slot three times in a row.
9.  **Excessive Context Switching**: When a customer switches intents $\ge 3$ times in a single session.
10. **Critical Security Takeover**: Immediate handoff for high-risk flags (e.g., card fraud reports).
11. **Reopened Grievance / Low Historical CSAT**: Transferring clients who have poor CSAT histories ($\le 2/5$).

---

## 🚀 Setup & Execution

### 1. Installation
Ensure python `3.11` is installed. Clone the repository and install dependencies in your virtual environment:

```bash
cd nexbank-agent
.venv\Scripts\activate
pip install -r requirements.txt  # Or install dependencies via poetry
```

### 2. Configure Environment Keys
Add your Anthropic API Key to resolve live responses:
```bash
# Windows Command Prompt
set ANTHROPIC_API_KEY=your_api_key_here

# Windows PowerShell
$env:ANTHROPIC_API_KEY="your_api_key_here"
```
*Note: If no API key is present, the app gracefully falls back to mock classification matrices for offline testing.*

### 3. Run the Core Web Server
Launch the FastAPI uvicorn daemon:
```bash
.venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
You can access the **SmartAssist Sandbox UI Dashboard Console** by opening `http://127.0.0.1:8000/` in your browser.

---

## 🧪 Verification & Testing

### Running Unit Tests
To run the automated test suite of 84 unit and system regression gates:
```bash
.venv\Scripts\pytest
```

### Running the End-to-End UI Integration Tester
To run the offline dialogue simulation suite mapping all 12 key system capabilities:
```bash
.venv\Scripts\python scratch/test_ui_functions.py
```
