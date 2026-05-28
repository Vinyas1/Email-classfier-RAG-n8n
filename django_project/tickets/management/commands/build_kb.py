from django.core.management.base import BaseCommand
from tickets.rag import ingest_tickets_csv, ingest_docs_folder, DOCS_DIR


class Command(BaseCommand):
    help = "Build/refresh the RAG knowledge base from dataset/tickets.csv and knowledge_base/docs/"

    def handle(self, *args, **options):
        self.stdout.write("Ingesting dataset/tickets.csv ...")
        n_tickets = ingest_tickets_csv()
        self.stdout.write(self.style.SUCCESS(f"  embedded {n_tickets} past tickets"))

        self.stdout.write(f"Ingesting docs from {DOCS_DIR} ...")
        n_docs = ingest_docs_folder()
        if n_docs:
            self.stdout.write(self.style.SUCCESS(f"  embedded {n_docs} doc chunks"))
        else:
            self.stdout.write(
                f"  no docs found yet - drop .txt/.md SOPs into {DOCS_DIR} and re-run this command"
            )

        self.stdout.write(self.style.SUCCESS("Knowledge base ready."))
