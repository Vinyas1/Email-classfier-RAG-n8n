from django.core.management.base import BaseCommand
from tickets.rag import ingest_tickets_csv, ingest_docs_folder, DOCS_DIR


class Command(BaseCommand):
    help = "Build/refresh the RAG knowledge base from dataset/tickets.csv and knowledge_base/docs/"

    def handle(self, *args, **options):
        self.stdout.write("Ingesting dataset/tickets.csv ...")
        n_tickets = ingest_tickets_csv()
        self.stdout.write(self.style.SUCCESS(f"  embedded {n_tickets} past tickets"))
