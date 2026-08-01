from django.contrib import admin
from core.models import ChatSession, ChatMessage, IngestedFile

admin.site.register(ChatSession)
admin.site.register(ChatMessage)
admin.site.register(IngestedFile)