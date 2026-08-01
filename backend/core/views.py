from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from rest_framework import status

from core.models import ChatSession, ChatMessage, IngestedFile
from core.serializers import ChatSessionSerializer, ChatMessageSerializer
from core.services import process_chat_message, ingest_raw_text_segment, get_session_graph_metadata


class ChatSessionView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id=None):
        """
        List all sessions or fetch a single session.
        """
        if session_id:
            try:
                session = ChatSession.objects.get(id=session_id, user=request.user)
                return Response(ChatSessionSerializer(session).data)
            except ChatSession.DoesNotExist:
                return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)
        
        sessions = ChatSession.objects.filter(user=request.user)
        return Response(ChatSessionSerializer(sessions, many=True).data)

    def post(self, request):
        """
        Create a new chat session.
        """
        serializer = ChatSessionSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, session_id):
        """
        Delete a chat session.
        """
        try:
            session = ChatSession.objects.get(id=session_id, user=request.user)
            session.delete()
            return Response({"status": "success", "detail": f"Session {session_id} deleted."})
        except ChatSession.DoesNotExist:
            return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)


class ChatMessageView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        """
        Get message history for a session.
        """
        try:
            session = ChatSession.objects.get(id=session_id, user=request.user)
            messages = ChatMessage.objects.filter(session=session).order_by('created_at')
            return Response(ChatMessageSerializer(messages, many=True).data)
        except ChatSession.DoesNotExist:
            return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)

    def post(self, request, session_id):
        """
        Handle chat message or knowledge ingestion.
        """
        user_input = request.data.get("content")
        is_ingestion = request.data.get("is_ingestion", False)
        title = request.data.get("title") or "Pasted Text"

        if not user_input:
            return Response({"error": "Content is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            session = ChatSession.objects.get(id=session_id, user=request.user)
            
            # Knowledge ingestion
            if is_ingestion:
                stats = ingest_raw_text_segment(session_id=session_id, title=title, content=user_input)
                ingested_file = IngestedFile.objects.create(session=session, file_name=title, text=user_input)
                
                return Response({
                    "status": "success",
                    "detail": "Data ingested.",
                    "file_id": ingested_file.id,
                    "stats": stats
                }, status=status.HTTP_201_CREATED)

            # Chat message
            else:
                # Update title if it's the default
                if session.title == "New Conversation":
                    clean_title = user_input.strip()[:40]
                    if len(user_input) > 40:
                        clean_title += "..."
                    session.title = clean_title
                    session.save(update_fields=['title'])

                result = process_chat_message(user=request.user, session_id=session_id, user_input=user_input)
                return Response(result, status=status.HTTP_200_OK)
                
        except ChatSession.DoesNotExist:
            return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)


class GraphStructureView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        """
        Fetch graph connections for visualization.
        """
        try:
            # Security fix: added user=request.user
            session = ChatSession.objects.get(id=session_id, user=request.user)
            max_nodes = int(request.query_params.get('max_nodes', 20))
            metadata = get_session_graph_metadata(session.id, max_nodes)
            
            return Response(metadata, status=status.HTTP_200_OK)
        except ChatSession.DoesNotExist:
            return Response({"error": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": f"Error parsing graph: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)