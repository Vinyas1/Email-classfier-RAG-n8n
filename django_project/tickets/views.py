from django.http import JsonResponse
from .ml_model import predict_category
from .models import Ticket
from .rag import retrieve_similar, draft_reply as rag_draft_reply

from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def classify_ticket(request):

    data = json.loads(request.body)

    text = data["text"]

    category = predict_category(text)

    Ticket.objects.create(
        text=text,
        predicted_category=category
    )

    return JsonResponse({"category": category})


@csrf_exempt
def draft_reply_view(request):
    """RAG endpoint: classify, retrieve similar past tickets, draft a grounded reply.

    Separate from classify_ticket on purpose - the n8n workflow and any
    existing integrations keep calling /api/predict/ exactly as before.
    This is an additive endpoint, not a replacement.
    """
    data = json.loads(request.body)
    text = data["text"]

    category = predict_category(text)
    retrieved = retrieve_similar(text, k=3)
    reply = rag_draft_reply(text, category, retrieved)

    Ticket.objects.create(
        text=text,
        predicted_category=category
    )

    return JsonResponse({
        "category": category,
        "draft_reply": reply,
        "sources": [
            {"category": m["metadata"].get("category"), "snippet": m["text"][:150]}
            for m in retrieved
        ],
    })
