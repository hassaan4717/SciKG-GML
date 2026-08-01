from django.urls import path
from core.views import ChatSessionView, ChatMessageView, GraphStructureView

urlpatterns = [
    # Get all sessions or create a new session
    path('', ChatSessionView.as_view(), name='chat-session-list'),
    
    # Get specific session details or delete it
    path('<int:session_id>/', ChatSessionView.as_view(), name='chat-session-detail'),
    
    # Get message history or send ingestion payloads
    path('<int:session_id>/messages/', ChatMessageView.as_view(), name='chat-message-list'),

    # Get graph connections
    path('<int:session_id>/graph/', GraphStructureView.as_view(), name='session-graph'),
]