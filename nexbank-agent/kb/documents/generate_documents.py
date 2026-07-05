import json
import os
from pathlib import Path
from datetime import datetime, timezone

# Ensure output directory exists
DOC_DIR = Path(__file__).parent
DOC_DIR.mkdir(parents=True, exist_ok=True)

# Define categories, product types, and regions
categories = [
    "faq",
    "product_terms",
    "dispute_and_complaint",
    "regulatory_rbi"
]

products = [
    "savings",
    "credit_card",
    "personal_loan",
    "fixed_deposit"
]

regions = [
    "IN-national",
    "IN-KA",
    "IN-MH",
    "IN-DL"
]

# Simple template structures to generate high-quality, realistic synthetic text programmatically.
faq_templates = [
    ("How do I activate my {product_name}?",
     "To activate your NexBank {product_name}, please log in to the NexBank mobile banking application, navigate to the 'Cards & Services' tab, select your specific active {product_name}, and click on the 'Activate Now' button. You will be prompted to enter a 6-digit OTP sent to your registered mobile number. Alternatively, you can activate it by performing a transaction at any local NexBank ATM by inserting your card and setting a personalized secure PIN. Please contact customer support if you encounter any authentication issues."),
    ("What are the maintenance fees for {product_name}?",
     "NexBank {product_name} products are subject to a nominal annual maintenance charge depending on the account tier. For standard accounts, the yearly charge is Rs. 150 plus applicable GST. Premium tier accounts have zero maintenance fees, provided that a monthly average balance of Rs. 25,000 is maintained consistently. If the monthly balance falls below this threshold, a fee of Rs. 100 per billing cycle will be deducted automatically from the principal account. Please review the complete fee schedule online for details."),
    ("How do I troubleshoot login issues for {product_name} app?",
     "If you are unable to log in to the NexBank mobile app to manage your {product_name}, first ensure your internet connection is stable. Try clearing the application cache from your mobile settings menu and relaunching the application. If you have forgotten your credentials, click 'Forgot Password' or 'Reset PIN' to generate a password reset link sent via SMS and registered email. For account security, after three unsuccessful login attempts, your digital access to {product_name} services will be temporarily locked for a duration of 2 hours."),
    ("Can I change transaction limits on my {product_name}?",
     "Yes, you can customize your spending and transaction limits for your {product_name} directly through the NexBank internet portal or mobile banking app. Navigate to 'Limit Settings' under the account menu to adjust daily limits for ATM withdrawals, online shopping, merchant POS terminals, and international transactions. Limit changes are applied in real-time and require verification via a secure mobile OTP. If you need a temporary limit increase beyond the default threshold, please call our 24/7 service line."),
]

terms_templates = [
    ("NexBank {product_name} interest rates and yields",
     "Interest rates for NexBank {product_name} products are determined quarterly based on prevailing market conditions and RBI repo rates. Currently, standard yields offer up to 4.5% per annum, calculated on a daily balance basis and credited to the account holder at the end of each month. For high-value balances exceeding Rs. 10 lakhs, an enhanced interest rate tier of 5.5% is applied. Rates are subject to change, and any modifications will be communicated to customers 15 days in advance via registered email."),
    ("Eligibility criteria for {product_name}",
     "To apply for a NexBank {product_name}, applicants must be residents of India and aged between 18 and 60 years. Documents required for verification include valid proof of identity (such as Aadhaar, PAN card, or passport), residential address proof, and proof of income (such as salary slips for the last three months or tax return sheets for self-employed professionals). Minimum salary thresholds vary by region, with metropolitan cities requiring a minimum net income of Rs. 35,000 per month for prime approval."),
    ("Tenure options and payout terms for {product_name}",
     "NexBank {product_name} products offer flexible tenure options ranging from a minimum of 7 days up to a maximum duration of 10 years. Payout frequencies can be customized at the inception of the contract to be disbursed monthly, quarterly, or accumulated at maturity. Premature withdrawals are permitted but subject to a nominal penalty charge of 0.5% on the contracted interest rate. Automatic renewal facility is enabled by default unless written instruction is received 48 hours prior to maturity."),
    ("Terms and conditions for {product_name} default",
     "Failure to meet repayment terms or maintain required balances on your NexBank {product_name} constitutes a default event. In the event of default, penal interest at the rate of 2% per month will be charged on all overdue amounts. Additionally, the account status will be reported to credit reference bureaus, including CIBIL, which may negatively impact your overall credit score. NexBank reserves the right to initiate recovery actions in compliance with fair practice codes and regulatory guidelines."),
]

dispute_templates = [
    ("Policy for unauthorized {product_name} disputes",
     "In the event of an unauthorized transaction on your NexBank {product_name}, customers must report the incident to the bank immediately to limit personal liability. Reports can be filed online, via the mobile app, or by calling our hotlines. If reported within 3 working days of the transaction, the customer has zero liability, and the bank will initiate a provisional credit within 10 days. If reported between 4 to 7 working days, customer liability is capped at Rs. 10,000. Reports filed after 7 days will be resolved per board-approved policies."),
    ("Handling duplicate charges on {product_name}",
     "If you have been charged twice for a single transaction on your NexBank {product_name}, please submit a dispute form along with the transaction reference IDs and payment receipt. Our merchant services team will review the transaction logs against the card network system. If the duplicate debit is confirmed, a reversal credit will be processed back to the original source account within 5 working days. Dispute requests must be submitted within 60 days of the statement date to be eligible for merchant chargeback recourse."),
    ("Refund and dispute resolution timelines for {product_name}",
     "The turnaround time (TAT) for resolving standard disputes related to NexBank {product_name} is 30 working days from the official date of dispute registration. During this investigation window, the bank coordinates with merchant payment gateways and acquiring banks to collect transaction logs. If complex international transactions are involved, the resolution window may be extended up to 45 working days. Regular progress updates will be shared with the customer via SMS notifications at key stages."),
    ("Escalation matrix for {product_name} customer grievances",
     "NexBank is committed to resolving customer disputes promptly. If you are unsatisfied with the resolution provided by our customer support executive regarding your {product_name}, you can escalate your grievance to Level 2 (Grievance Redressal Officer) via email or postal mail. Level 2 reviews are resolved within 10 working days. If the issue remains unresolved, you can further escalate the matter to Level 3 (Principal Nodal Officer) at our corporate office, who will issue a final decision within 7 working days."),
]

regulatory_templates = [
    ("RBI Ombudsman escalation path for {product_name} complaints",
     "Under the Reserve Bank of India Integrated Ombudsman Scheme, if your grievance regarding NexBank {product_name} is not resolved to your satisfaction within 30 days of filing a complaint with the bank, you have the right to escalate the matter directly to the RBI Ombudsman. Complaints can be registered online through the RBI CMS portal or via physical letters. The Ombudsman acts as an independent arbitrator, and the bank is bound by any final awards or directions issued. There are no fees associated with this filing."),
    ("Redressal timelines and compensation policies for {product_name} delays",
     "Per regulatory guidelines, banks must resolve failed transaction disputes for {product_name} and reverse funds within specified Turnaround Times (T+5 days for card transactions). If NexBank fails to credit the disputed amount back to the customer's account within T+5 days, the bank will pay a daily compensation of Rs. 100 to the customer for each day of delay beyond the SLA. This compensation is credited automatically without requiring a separate request, in accordance with the RBI compensation framework."),
    ("Turnaround time (TAT) commitments for {product_name} issues",
     "NexBank adheres to RBI-mandated turnaround times for all service requests linked to {product_name}. ATM transaction complaints must be resolved within a maximum of 7 days from the receipt of the complaint. Mobile banking and digital wallet failure redressals are bound by a 48-hour resolution window. For normal billing disputes, the maximum redressal timeline is 30 days. Regular audits are conducted by compliance teams to ensure strict adherence to these service delivery standards."),
    ("Fair practices code and consumer protection for {product_name}",
     "NexBank's customer protection policy for {product_name} is aligned with the RBI Charter of Customer Rights. We guarantee the Right to Fair Treatment, Right to Transparency, Right to Suitability, and Right to Grievance Redressal. Customers are protected against coercive collections and unauthorized disclosure of personal data. All terms, fee changes, and interest rates are clearly disclosed. Annual compliance audits are published to maintain transparency with regulatory bodies and our account holders."),
]

def generate_docs():
    doc_id = 1
    generated_count = 0
    
    # Loop over categories, products, and regions to generate a dense, balanced set
    # 4 categories * 4 products * 4 regions = 64 combinations
    # For each combination, let's generate 4 distinct documents = 256 documents total
    for cat in categories:
        if cat == "faq":
            templates = faq_templates
        elif cat == "product_terms":
            templates = terms_templates
        elif cat == "dispute_and_complaint":
            templates = dispute_templates
        else:
            templates = regulatory_templates
            
        for prod in products:
            prod_name = prod.replace("_", " ").title()
            for reg in regions:
                for idx, (title_tpl, content_tpl) in enumerate(templates):
                    doc_data = {
                        "id": f"kb_doc_{doc_id:04d}",
                        "title": title_tpl.format(product_name=prod_name),
                        "category": cat,
                        "product_type": prod,
                        "region": reg,
                        "content": "[SYNTHETIC POLICY FOR DEMO PURPOSES ONLY] " + content_tpl.format(product_name=prod_name),
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Write as JSON file
                    file_path = DOC_DIR / f"{doc_data['id']}.json"
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(doc_data, f, indent=2)
                        
                    doc_id += 1
                    generated_count += 1
                    
    print(f"Successfully generated {generated_count} synthetic knowledge documents under {DOC_DIR}")

if __name__ == "__main__":
    generate_docs()
