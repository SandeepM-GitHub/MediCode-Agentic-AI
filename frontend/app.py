import streamlit as st
import pandas as pd
import json
import sys
import os
import torch

# 1. System Path Routing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Backend Imports
from backend.core.agent import build_agent
from backend.core.review import submit_human_review
from backend.data.db import SessionLocal, Claim

# 3. UI Configuration
st.set_page_config(
    page_title="MediCodeAgent Architect",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 4. Data Layer
def fetch_global_metrics():
    """Fetches real-time stats from the SQLite database."""
    db = SessionLocal()
    try:
        # Querying the database for aggregate data
        total_claims = db.query(Claim).count()
        approved_claims = db.query(Claim).filter(Claim.status == "approved").all()
        suspicious_count = db.query(Claim).filter(Claim.status == "suspicious").count()
        rejected_count = db.query(Claim).filter(Claim.status == "rejected").count()
        
        # Calculate revenue and rejection percentage
        total_revenue = sum(c.payment_amount for c in approved_claims if c.payment_amount)
        rejection_rate = (rejected_count / total_claims * 100) if total_claims > 0 else 0
        
        return total_claims, total_revenue, suspicious_count, rejection_rate
    finally:
        db.close()

# 5. UI HEADER & METRICS
st.title("⚕️ MediCodeAgent: Systems Architect Dashboard")

# Dynamically detect hardware
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
device_status = "ACTIVE" if torch.cuda.is_available() else "INACTIVE"
vector_db = "FAISS"
llm_model = "Llama 3.2 3B"

st.markdown(f"**Hardware Status:** `{device_name} (CUDA): {device_status}` | **Vector DB:** `{vector_db}` | **LLM:** `{llm_model}`")
st.divider()

total, rev, susp, rej_rate = fetch_global_metrics()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Claims Processed", value=total)
with col2:
    st.metric(label="Total Stripe Revenue", value=f"${rev:,.2f}")
with col3:
    # Add a mock delta here just so to see the inverse color in action!
    st.metric(label="Pending Human Review", value=susp, delta=f"{susp} awaiting action", delta_color="inverse") 
with col4:
    st.metric(label="Rule Rejection Rate", value=f"{rej_rate:.1f}%")

st.divider()

# 6. INITIALIZE AGENT
# @st.cache_resource ensures we only build the LangGraph agent ONCE when the app loads, 
# preventing massive memory leaks and slow reloads.
@st.cache_resource
def get_agent():
    return build_agent()

agent = get_agent()

# --- LAYOUT: SIDEBAR (Intake) ---
st.sidebar.header("📄 Clinical Intake")
st.sidebar.markdown("Paste the physician's note below to begin the Agentic pipeline.")

clinical_note = st.sidebar.text_area(
    "Physician Note:", 
    height=250, 
    value="Patient complains of acute pharyngitis. Performed rapid strep test."
)

process_btn = st.sidebar.button("🚀 Process Claim", type="primary", use_container_width=True)

# --- MAIN STAGE: AGENT EXECUTION ---
st.subheader("Live Agent Execution")

# This block ONLY runs when the user clicks the button
if process_btn:
    if not clinical_note.strip():
        st.sidebar.error("Please enter a clinical note.")
    else:
        # st.status creates a beautiful expanding loading animation
        with st.status("Executing LangGraph Pipeline...", expanded=True) as status:
            st.write("🧠 Extracting medical entities via LLM...")
            st.write(f"⚡ Querying {vector_db} on {device_name}...")
            st.write("⚖️ Evaluating against Deterministic Payer Rules...")
            
            # 🔥 THE CROWN JEWEL: Running your backend pipeline!
            result = agent.invoke({"clinical_note": clinical_note, "messages": []})
            
            status.update(label="Pipeline Complete!", state="complete", expanded=False)
        
        # Extract the results cleanly
        final_status = result.get("status", "error").lower()
        tx_id = result.get("stripe_transaction_id") or "N/A"
        rule_id = result.get("rule_id", "UNKNOWN")
        reason = result.get("rejection_reason", "No reason provided.")
        icd10 = result.get("final_icd10_code", "None")
        cpt = result.get("final_cpt_code", "None")
        
        # Display the Final Verdict Card using Streamlit's colored boxes
        if final_status == "approved":
            st.success(f"### ✅ CLAIM APPROVED\n**Rule Trigger:** `{rule_id}` | **Stripe TX:** `{tx_id}`")
        elif final_status == "rejected":
            st.error(f"### ❌ CLAIM REJECTED\n**Rule Trigger:** `{rule_id}`\n\n**Reason:** {reason}")
        elif final_status == "suspicious":
            st.warning(f"### ⚠️ CLAIM SUSPICIOUS (Review Required)\n**Rule Trigger:** `{rule_id}`\n\n**Reason:** {reason}")
        
        # Display the specific codes the Agent chose
        st.markdown("#### Extraction Results")
        colA, colB = st.columns(2)
        colA.info(f"**ICD-10 Code:**\n\n`{icd10}`")
        colB.info(f"**CPT Code:**\n\n`{cpt}`")
            
        # Architect Debug View (Allows you to inspect the exact memory state)
        with st.expander("👨‍💻 Architect Debug Data (Raw JSON State)"):
            st.json(result)

st.divider()

