from django.db import models
from django.conf import settings

class ChatSession(models.Model):
    """Represents a single conversation thread for a user."""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='chat_sessions'
    )
    title = models.CharField(max_length=255, default="New Conversation")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['user', '-updated_at']),
        ]

    def __str__(self):
        return f"[{self.id}] {self.title} ({self.user.username})"


class ChatMessage(models.Model):
    """Represents a single turn (user query or assistant response) in a ChatSession."""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]

    session = models.ForeignKey(
        ChatSession, 
        on_delete=models.CASCADE, 
        related_name='messages'
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    metadata = models.JSONField(
        default=dict, 
        blank=True, 
        help_text="Stores RAG-specific data like mode, sources, entities, etc."
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"[{self.role.capitalize()}] message in {self.session.title} - {self.session.id}"


class IngestedFile(models.Model):
    """Tracks files or pasted text processed within a specific chat session."""
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name='files'
    )
    file_name = models.CharField(
        max_length=255, 
        default="Pasted Text",
        help_text="Original file name or 'Pasted Text' for manual inputs"
    )
    text = models.TextField(help_text="The full text content ingested into the session")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.file_name} in Session: {self.session.title}"