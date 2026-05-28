from django.urls import path
from .views import classify_ticket, draft_reply_view

urlpatterns = [
    path("predict/", classify_ticket),
    path("draft-reply/", draft_reply_view),
]
