# ⚕️ MediCode-Agentic-AI: Agentic RAG for Autonomous Revenue Cycle Management (RCM)

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA_Accelerated-ee4c2c?logo=pytorch&logoColor=white)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph-orange)
![FastMCP](https://img.shields.io/badge/Integration-FastMCP-brightgreen)
![Stripe](https://img.shields.io/badge/Payments-Stripe_API-635bff?logo=stripe&logoColor=white)

MediCode-Agentic-AI is an end-to-end **Agentic Retrieval-Augmented Generation (RAG)** pipeline designed to automate the healthcare "Medical-to-Money" workflow. It translates unstructured clinical physician notes into standardized billing codes, adjudicates them against strict deterministic rules, and securely triggers financial payouts.



## 💡 The Core Idea & Enterprise Utility
In the healthcare enterprise, claim denials cost billions annually due to manual coding errors and vague documentation. **MediCode-Agentic-AI acts as an autonomous medical biller.** By coupling a localized Large Language Model (LLM) with mathematical vector similarity search and hard-coded insurance logic, this system processes clinical documentation at high throughput while guaranteeing compliance and preventing automated financial errors.

### 🌟 The Best-Selling Point: Why this isn't just another "AI Wrapper"
Most mass-produced AI projects simply pass user text to the OpenAI API and print the response. MediCode-Agentic-AI is built for **enterprise security, deterministic safety, and actionability:**
1. **Zero Data Leakage (HIPAA-Aligned Concept):** Uses a local Llama 3.2 model via Ollama. Patient data never leaves the local machine.
2. **GPU-Accelerated RAG:** Vector embeddings are generated locally using SentenceTransformers, explicitly routed to **NVIDIA CUDA cores** for hardware-accelerated semantic search via FAISS.
3. **Deterministic Guardrails:** AI hallucinations are dangerous in finance. A custom Python Rule Engine acts as a gatekeeper, cross-referencing AI outputs for medical necessity (e.g., rejecting an ankle procedure for a throat diagnosis) and strict vector confidence thresholds (>0.85).
4. **Agentic Action:** The system doesn't just output text; it executes a state machine (LangGraph) that concludes by triggering a real-world financial transaction via the Stripe API.

## 🏗️ Architecture & Tech Stack

* **Frontend UI:** Streamlit (Features a Systems Architect dashboard with raw JSON state inspection and database visualization).
* **Orchestration:** LangGraph (Manages the `ClaimState` through the pipeline).
* **Intelligence:** Llama 3.2 (3B Parameters) running locally via Ollama.
* **Vector Database:** FAISS (Facebook AI Similarity Search) + `faiss-cpu` or `faiss-gpu`.
* **Tooling Protocol:** FastMCP (Model Context Protocol) securely isolates the database from the LLM.
* **Database:** SQLite + SQLAlchemy (Maintains a permanent audit trail of all AI decisions).
* **Payment Gateway:** Stripe API (Creates automated `PaymentIntent` transactions for approved claims).

## 🧑‍⚖️ Human-in-Loop (HIL) Safety
Complete automation is a liability without oversight. If the deterministic rule engine detects low FAISS confidence or a missing procedure code, the LangGraph agent halts the pipeline and flags the claim as `SUSPICIOUS` in the SQLite database. 

The Streamlit UI includes an Auditor Dashboard where a Human-in-the-Loop can review the AI's reasoning, provide manual justification, and explicitly override the system to either **Reject** or **Approve & Pay** the claim.

---

## 🚀 How to Run the Project Locally

To run this architecture on your local machine, ensure you have Python 3.10+ installed and an NVIDIA GPU (optional, but recommended for CUDA acceleration). *Note: The `.env` file containing the Stripe API key and `.gitignore` are pre-configured locally.*

### 1. Install Dependencies
Clone the repository and install the locked requirements in a virtual environment:
```bash
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Database & Vector Indexes
The system uses pre-seeded SQLite databases and FAISS indexes. Ensure `medical.db`, `icd10.index`, and `cpt.index` are present in the `backend/data/...` directory.

### 3. Start the FastMCP Librarian Server
The semantic search relies on an MCP tool server running independently. Open a terminal and run:
```bash
python -m backend.mcp.server
```
(Wait until the console prints Librarian is awake and indicates whether it loaded the model on CPU or CUDA).

### 4. Launch the Architect Dashboard
Open a second terminal, ensure your virtual environment is active, and start the Streamlit UI:
```bash
python -m streamlit run frontend/app.py
```
This will automatically open the interactive dashboard in your default web browser (typically at http://localhost:8501). Paste a clinical note into the sidebar and watch the Agentic pipeline execute in real-time!

## 🧪 Test Cases to Try in the Dashboard
To see the Agentic RAG system handle different scenarios, copy and paste these exact clinical notes into the Streamlit sidebar:

### 1. The "Happy Path" (Expected Result: ✅ APPROVED)
This note is clear, matches our vector database perfectly, and satisfies the medical necessity rules.
```bash
"Patient presents with a severe sore throat and difficulty swallowing.
Performed a rapid strep test in the clinic today."
```

AI Action: Extracts "severe sore throat" and "rapid strep test".

Vector Match: `High confidence` (>0.85) for J02.9 and 87880.

Rule Engine: Passes. Triggers a successful Stripe transaction.

### 2. The "Vague/Low Confidence" Path (Expected Result: ⚠️ SUSPICIOUS)
This note uses non-clinical language. The AI will find a match, but the math score will fall below the safety threshold, triggering the Human-in-the-Loop.
```bash
"The patient came in saying they feel super weird and tired all the time."
```
AI Action: Extracts "feel super weird and tired".

Vector Match: Matches R53.81 (Other malaise) but with a `low confidence` score (e.g., ~0.65).

Rule Engine: Flags as `R1_LOW_CONFIDENCE`. Halts payment. Requires you to manually review and approve/reject at the bottom of the dashboard.

### 3. The "Missing Data" Path (Expected Result: ❌ REJECTED)
This note describes a diagnosis but completely fails to mention any procedure being performed.
```bash
"Patient came in complaining of a sprained ankle after playing basketball. Advised rest and ice."
```
AI Action: Extracts the diagnosis (Ankle sprain) but outputs "None" for the procedure.

Vector Match: Finds S93.4 (Sprain of ankle) but fails to find a CPT code.

Rule Engine: Flags as `R0_MISSING_DATA` because an insurance claim cannot be billed without a valid procedure code. Claim is hard-rejected.

### 4. The "Fraud/Mismatch" Path (Expected Result: ❌ REJECTED)
This note is an example of medical billing fraud (upcoding or mismatched services). The agent must catch this.
```bash
"Patient complains of a sore throat. We took an X-ray of their foot."
```
AI Action: Extracts "sore throat" and "X-ray of foot".

Vector Match: Finds J02.9 (Pharyngitis) and 73630 (Radiologic exam, foot).

Rule Engine: Flags as `R2_MEDICAL_NECESSITY_FAIL`. The deterministic rules explicitly forbid billing a foot x-ray for a throat complaint. Claim is hard-rejected.
