from rest_framework import serializers
from core.models import ChatSession, ChatMessage, IngestedFile

class IngestedFileSerializer(serializers.ModelSerializer):
    """Serializer for tracking ingested files and pasted text."""
    class Meta:
        model = IngestedFile
        fields = ['id', 'session', 'file_name', 'text', 'uploaded_at']
        read_only_fields = ['id', 'uploaded_at']


class ChatMessageSerializer(serializers.ModelSerializer):
    """Serializer for individual chat turns within a session."""
    class Meta:
        model = ChatMessage
        fields = ['id', 'session', 'role', 'content', 'metadata', 'created_at']
        read_only_fields = ['id', 'created_at']


class ChatSessionSerializer(serializers.ModelSerializer):
    """Serializer for chat sessions with nested messages and files."""
    messages = ChatMessageSerializer(many=True, read_only=True)
    files = IngestedFileSerializer(many=True, read_only=True)

    class Meta:
        model = ChatSession
        fields = ['id', 'title', 'created_at', 'updated_at', 'messages', 'files']
        read_only_fields = ['id', 'created_at', 'updated_at']

    def create(self, validated_data):
        validated_data['user'] = self.context.get('request').user
        return super().create(validated_data)