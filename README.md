# Email Complaint Classifier + RAG Auto-Reply Drafting â€” IT Helpdesk
n8n + Django + Groq (Llama 3) + Chroma

## What's in here
- `/api/predict/` â€” unchanged. Classifies an email into network/hardware/software/access.
- `/api/draft-reply/` â€” new. Classifies, retrieves similar past tickets + SOP docs from a
  local vector store, and asks Groq to draft a grounded first-pass reply for a human to review.

## First Time Setup

### 1. Create & activate virtual environment
```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Add your Groq API key
- Go to https://console.groq.com, sign up, create a free API key
- Copy `.env.example` to `.env` in the project root
- Set `GROQ_API_KEY=your_key_here` inside `.env`

**If you previously had a key hardcoded in `ml_model.py`: rotate/revoke it on
console.groq.com â€” a hardcoded key in source is treated as compromised.**

### 4. Build the knowledge base (one-time, re-run anytime tickets.csv or docs change)
```
cd django_project
python manage.py build_kb
```
This embeds every row of `dataset/tickets.csv` plus any `.txt`/`.md` files in
`django_project/knowledge_base/docs/` into a local Chroma vector store at
`django_project/chroma_db/` (gitignored â€” rebuild it on each machine, don't commit it).

### 5. Run Django
```
python manage.py migrate
python manage.py runserver
```

### 6. Start n8n (separate terminal)
```
n8n start
```

## Endpoints
- `POST http://127.0.0.1:8000/api/predict/` â€” `{"text": "..."}` â†’ `{"category": "..."}`
- `POST http://127.0.0.1:8000/api/draft-reply/` â€” `{"text": "..."}` â†’ `{"category", "draft_reply", "sources"}`

## Adding your own SOPs / FAQ docs
Drop `.txt` or `.md` files into `django_project/knowledge_base/docs/`, then re-run:
```
python manage.py build_kb
```
A sample (`vpn-reset-procedure.md`) is included to show the expected format.

## Every Day
```
venv\Scripts\activate
cd django_project
python manage.py runserver
```
And `n8n start` in a second terminal.

## Categories
- network  â†’ WiFi, internet, VPN connectivity, network drops
- hardware â†’ laptop, printer, monitor, battery, physical devices
- software â†’ apps, ERP, crash, Teams, Zoom, Excel, browser
- access   â†’ password, account locked, permissions, MFA
- unclassified â†’ Groq could not determine category (rare)


