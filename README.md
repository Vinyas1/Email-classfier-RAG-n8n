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
