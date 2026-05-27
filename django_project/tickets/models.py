from django.db import models

class Ticket(models.Model):

    text = models.TextField()

    predicted_category = models.CharField(max_length=100)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.text
