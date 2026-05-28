"""
ml_model.py â€” Groq LLM based IT complaint classifier.
views.py and n8n workflow remain completely unchanged.

Setup:
    pip install groq
    Replace YOUR_GROQ_API_KEY_HERE with your key from console.groq.com
"""

import os
import re
from groq import Groq

# â”€â”€ Reads from the GROQ_API_KEY environment variable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Set it in your shell, a .env file (see .env.example), or your OS environment.
# Never hardcode the key here - it gets committed to git otherwise.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set. "
        "Copy .env.example to .env and add your key, or export it in your shell."
    )

client = Groq(api_key=GROQ_API_KEY)

VALID_CATEGORIES = {"network", "hardware", "software", "access"}


def preprocess(text: str) -> str:
    text = str(text).strip()
    text = re.sub(
        r'(best regards|regards|sincerely|thank you|thanks)[,.\s\w]*$',
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'^(dear\s+[\w\s,\[\]]+?,|hi[\s,]+|hello[\s,]+)',
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_category(text: str) -> str:
    cleaned = preprocess(text)

    prompt = f"""You are an IT helpdesk complaint classifier.

Classify the following complaint email into EXACTLY one of these four categories:

- network  : WiFi, internet, LAN, VPN connectivity, network drives, bandwidth, DNS, connection drops
- hardware : laptop, printer, monitor, keyboard, mouse, battery, charger, scanner, webcam, docking station, physical devices
- software : applications, ERP, crash, slow performance, installation errors, OS issues, Teams, Zoom, Excel, browser, database errors
- access   : password reset, account locked, login issues, permissions, MFA, OTP, credentials, account provisioning

Complaint:
\"\"\"{cleaned}\"\"\"

Reply with ONLY one word â€” the category name. Nothing else. No explanation."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )

        category = response.choices[0].message.content.strip().lower()
        category = re.sub(r'[^a-z]', '', category)

        if category in VALID_CATEGORIES:
            return category

        return _retry(cleaned)

    except Exception as e:
