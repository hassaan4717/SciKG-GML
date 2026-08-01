from __future__ import annotations

import os
import re
import pickle
import logging
import tempfile
import networkx as nx
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Mapping


from core.pipeline.graph_rag import GraphRAG
from core.models import ChatMessage, ChatSession
from core.utils import call_llm, generate_rag_answer, get_api_gateway, safe_float, source_filename

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", 12))

API_KEY = os.getenv("API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
BASE_RAG_DIR = os.getenv("BASE_RAG_DIR")
BASE_OPENAI_URL = os.getenv("BASE_OPENAI_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", 600))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", 1024))
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", 12))

INSUFFICIENT_CONTEXT_MESSAGE = (
    "This topic isn't sufficiently covered in the available text."
)
GENERATION_ERROR_MESSAGE = (
    "I'm sorry, I encountered an error while processing your request."
)

ALLOWED_HISTORY_ROLES = {"user", "assistant"}


def rewrite_query_for_retrieval(
    history: list[dict[str, str]],
    query: str,
) -> str:
    """
    Rewrite a follow-up question as a standalone retrieval query.
    """
    query = str(query).strip()

    if not query or not history:
        return query

    recent_history = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history[CHAT_HISTORY_LIMIT:]
        if message.get("role") in {"user", "assistant"} and message.get("content")
    )

    prompt = f"""
        /no_think

        Rewrite the latest user message as one standalone search query.

        Use the conversation only to resolve references such as he, she, it,
        they, that person, or that event.

        Do not answer the question.
        Do not explain anything.
        Return only the rewritten query.

        Conversation:
        {recent_history}

        Latest message:
        {query}
    """.strip()

    try:
        rewritten = call_llm(
            prompt=prompt,
            temperature=0.0,
        ).strip()

        if not rewritten:
            logging.warning(
                "Invalid rewritten query: %r",
                rewritten,
            )
            return query

        return rewritten.strip("\"'")

    except Exception:
        logging.exception(
            "Failed to rewrite retrieval query: %r",
            query,
        )
        return query


def process_chat_message(
    user: Any,
    session_id: str | int,
    user_input: str,
) -> dict[str, Any]:
    """
    Process one GraphRAG chat turn.
    """
    normalized_input = str(user_input).strip()

    if not normalized_input:
        raise ValueError("User input cannot be empty.")

    try:
        session = ChatSession.objects.get(
            id=session_id,
            user=user,
        )
    except ChatSession.DoesNotExist as exc:
        raise ValueError("Session not found or access denied.") from exc

    user_message = ChatMessage.objects.create(
        session=session,
        role="user",
        content=normalized_input,
    )

    # Load recent messages after saving the current user message.
    recent_messages = list(
        ChatMessage.objects.filter(
            session=session,
            role__in=ALLOWED_HISTORY_ROLES,
        ).order_by("-created_at")[:CHAT_HISTORY_LIMIT]
    )
    recent_messages.reverse()

    history = [
        {
            "role": str(message.role).strip().lower(),
            "content": str(message.content).strip(),
        }
        for message in recent_messages
        if str(message.role).strip().lower() in ALLOWED_HISTORY_ROLES
        and str(message.content).strip()
    ]

    # Exclude the current question from the history passed to the rewriter.
    previous_history = history[:-1] if history else []

    retrieval_query = rewrite_query_for_retrieval(
        history=previous_history,
        query=normalized_input,
    )

    result: dict[str, Any] = {
        "hits": [],
        "mode": "global_fallback",
        "summaries": [],
        "entities": [],
    }

    context = ""

    try:
        rag = GraphRAG(
            persist_dir=str(_session_dir(session_id)),
            api_gateway=get_api_gateway(),
        )

        if getattr(rag, "_all_chunks", []):
            # Use the rewritten standalone query for retrieval.
            result = rag.query(retrieval_query)
            context = rag.format_context(result).strip()

    except Exception as exc:
        logging.exception(
            (
                "RAG retrieval failed | session_id=%s | "
                "message_id=%s | retrieval_query=%r | "
                "error_type=%s | error=%s"
            ),
            session_id,
            getattr(user_message, "id", "unknown"),
            retrieval_query,
            type(exc).__name__,
            exc,
        )

        result = {
            "hits": [],
            "mode": "retrieval_error",
            "summaries": [],
            "entities": [],
        }

        context = ""

    has_context = bool(context and context != "No relevant content found.")

    if has_context:
        answer = generate_rag_answer(
            messages=history,
            context=context,
        )
    else:
        answer = INSUFFICIENT_CONTEXT_MESSAGE

    raw_answer = (
        str(answer).strip() if answer is not None else ""
    ) or GENERATION_ERROR_MESSAGE

    # Remove inline citations and append unique source filenames at the end.
    final_content, sources = _append_sources(
        answer=raw_answer,
        result=result,
    )

    metadata = {
        "mode": str(result.get("mode", "hybrid")),
        "retrieval_query": retrieval_query,
        "sources": sources,
        "num_hits": len(result.get("hits", []) or []),
        "communities_used": len(result.get("summaries", []) or []),
        "entities": [str(entity) for entity in (result.get("entities", []) or [])],
    }

    assistant_message = ChatMessage.objects.create(
        session=session,
        role="assistant",
        content=final_content,
        metadata=metadata,
    )

    return {
        "id": assistant_message.id,
        "role": "assistant",
        "content": final_content,
        "metadata": metadata,
    }


def ingest_raw_text_segment(
    session_id: str | int,
    title: str,
    content: str,
) -> dict[str, Any]:
    """
    Ingest raw text and rebuild GraphRAG summaries.
    """
    title = str(title).strip()
    content = str(content).strip()

    if not title:
        raise ValueError("Title cannot be empty.")

    if not content:
        raise ValueError("Content cannot be empty.")

    rag = GraphRAG(
        persist_dir=str(_session_dir(session_id)),
        api_gateway=get_api_gateway(),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / source_filename(title)
        file_path.write_text(content, encoding="utf-8")

        return rag.ingest(file_path=str(file_path))


def get_session_graph_metadata(
    session_id: str | int, max_nodes: int = 25
) -> dict[str, Any]:
    """
    Return graph nodes, edges, and community summaries.
    """
    result: dict[str, Any] = {
        "top_nodes": [],
        "connections": [],
        "communities": [],
    }

    try:
        max_nodes = max(1, min(int(max_nodes), 200))
        state_path = _session_dir(session_id) / "graph_state.pkl"

        if not state_path.is_file():
            return result

        with state_path.open("rb") as file:
            state = pickle.load(file)

        graph = state.get("graph")
        communities = state.get("communities", {})

        if isinstance(
            graph,
            (
                nx.Graph,
                nx.DiGraph,
                nx.MultiGraph,
                nx.MultiDiGraph,
            ),
        ):
            scores = dict(graph.degree(weight="weight"))
            top_nodes = sorted(
                graph.nodes,
                key=lambda node: (
                    -safe_float(scores.get(node, 0), 0),
                    str(node).casefold(),
                ),
            )[:max_nodes]

            result["top_nodes"] = [str(node) for node in top_nodes]

            subgraph = graph.subgraph(top_nodes)

            if subgraph.is_multigraph():
                edge_iterator = (
                    (source, target, data)
                    for source, target, _, data in subgraph.edges(keys=True, data=True)
                )
            else:
                edge_iterator = subgraph.edges(data=True)

            for source, target, data in edge_iterator:
                result["connections"].append(
                    {
                        "source": str(source),
                        "target": str(target),
                        "weight": safe_float((data or {}).get("weight", 1.0)),
                    }
                )

        if isinstance(communities, Mapping):
            for community_id, community in communities.items():
                if not isinstance(community, Mapping):
                    continue

                entities = [
                    str(entity) for entity in (community.get("entities", []) or [])
                ]

                if len(entities) != 1:
                    result["communities"].append(
                        {
                            "id": str(community_id),
                            "title": (str(community.get("title", "")).strip()),
                            "summary": (
                                str(community.get("summary", "")).strip()
                                or "No summary generated yet."
                            ),
                            "top_entities": entities[:5],
                            "size": len(entities),
                            "num_chunks": len(community.get("chunk_keys", []) or []),
                        }
                    )

            result["communities"].sort(key=lambda item: -item["size"])

    except Exception:
        logging.exception(
            "Could not read graph metadata for session %s.",
            session_id,
        )

    return result


# -------------------- Helper ---------------------------


def _session_dir(session_id: str | int) -> Path:
    """
    Return the GraphRAG directory for a session.
    """
    safe_session_id = str(session_id).strip()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", safe_session_id):
        raise ValueError("Invalid session ID.")

    path = Path(BASE_RAG_DIR) / f"session_{safe_session_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _clean_source_name(source: Any) -> str:
    """
    Return the original source filename without modifying it.
    """
    name = Path(str(source)).name.strip()
    org_name = re.sub(
        r"_[a-f0-9]{8,64}$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    return org_name


def _retrieved_source_names(result: Mapping[str, Any]) -> list[str]:
    """
    Return unique readable names for retrieved source files.
    """
    sources: list[str] = []
    seen: set[str] = set()

    for hit in result.get("hits", []) or []:
        if not isinstance(hit, (tuple, list)) or len(hit) < 2:
            continue

        metadata = hit[1]

        if not isinstance(metadata, Mapping):
            continue

        source = metadata.get("source")

        if not source:
            continue

        name = _clean_source_name(source)
        key = name.casefold()

        if name and key not in seen:
            seen.add(key)
            sources.append(name)

    return sources


def _append_sources(answer: str, result: Mapping[str, Any]) -> tuple[str, list[str]]:
    """
    Remove model-generated citations and append unique filenames only once.
    """
    answer = str(answer).strip()

    # Remove a Sources section accidentally generated by the model.
    answer = re.split(
        r"(?im)^\s*#{0,6}\s*Sources?\s*:?\s*$",
        answer,
        maxsplit=1,
    )[0].strip()

    # Remove [1], [1][2], [1, 2], and [Source: filename].
    answer = re.sub(
        r"(?:\[\s*\d+(?:\s*,\s*\d+)*\s*\])+",
        "",
        answer,
    )
    answer = re.sub(
        r"\[\s*Source\s*:[^\]]+\]",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    # Clean spaces left by removed citations.
    answer = re.sub(r"[ \t]+([.,;:!?])", r"\1", answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer).strip()

    sources = _retrieved_source_names(result)

    if not sources:
        return answer, []

    source_list = "\n".join(f"- {source}" for source in sources)

    return (
        f"{answer}\n\n### Sources\n{source_list}",
        sources,
    )
