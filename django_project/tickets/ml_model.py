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

