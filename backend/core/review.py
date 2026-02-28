from backend.data.db import SessionLocal, Claim
from backend.core.payments import process_claim_payout

def submit_human_review(claim_id: int, decision: str, reviewer_name: str, notes: str):
    """
    Simulates Stage 10: Human-in-the-loop.
    Allows a human to manually override a suspicious claim.
    """
    db = SessionLocal()
    try:
        # 1. Find the claim in the filing cabinet
        claim = db.query(Claim).filter(Claim.id == claim_id).first()

        if not claim:
            return "Error: Claim not found"
        
        if claim.status != "suspicious":
            return f"Notice: Claim {claim_id} is currently '{claim.status}', not pending review."
        # 2. Apply the Human's decision
        if decision.lower() == "approved":
            amount = 50.0 if claim.cpt_code == "87880" else 20.0
        
            print(f"Auditor Approved Manually! Triggering Stripe payout of ${amount}...")
            payment_res = process_claim_payout(claim.id, amount)

            if payment_res["success"]:
                claim.status = "approved"
                claim.stripe_transaction_id = payment_res["transaction_id"]
                claim.payment_amount = amount
                claim.rejection_reason = f"Human Approved by {reviewer_name}: {notes}"

            else:
                return f" Stripe Error: {payment_res['error']}"
        
        else:
            claim.status = "rejected" # Will be 'approved' or 'rejected'
            claim.rejection_reason = f"Human rejected by {reviewer_name}: {notes}"

        # 3. Save the override permanently
        db.commit()
        return f"SUCCESS: Claim {claim_id} manually marked as {decision.upper()}" 
    finally:
        db.close()