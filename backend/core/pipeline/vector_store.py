import os
import logging
import chromadb
from dotenv import load_dotenv
from typing import Any, List, Optional, Tuple

from .api_gateway import APIRequestGateway

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "200"))


class VectorStore:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        api_gateway: APIRequestGateway,
    ) -> None:
        """
        Initialize the persistent ChromaDB vector store.
        """
        self.client = chromadb.PersistentClient(
            path=persist_dir,
        )
        self.collection_name = collection_name
        self.api_gateway = api_gateway

    def embed_texts(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate vector embeddings through the shared API gateway.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        batch_size = EMBEDDING_BATCH_SIZE

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            embeddings = self.api_gateway.embed_texts(
                batch,
                encoding_format="float",
            )

            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(
        self,
        query: str,
    ) -> List[float]:
        """
        Generate one query embedding through the shared API gateway.
        """
        query = str(query).strip()

        if not query:
            raise ValueError("Embedding query cannot be empty.")

        return self.api_gateway.embed_query(
            query,
            encoding_format="float",
        )

    def index_chunks(
        self,
        chunks: List[Any],
        source_id: str,
    ) -> Optional[chromadb.Collection]:
        """
        Index text chunks into the ChromaDB collection.
        """
        if not chunks:
            return None

        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        try:
            results = collection.get(
                where={"source_id": {"$eq": source_id}},
                limit=1,
            )

            if results["ids"]:
                logging.info(
                    "Source '%s' already exists. Deleting old data to re-index.",
                    source_id,
                )
                collection.delete(where={"source_id": {"$eq": source_id}})

        except Exception as exc:
            logging.warning(
                "Could not check or delete existing source '%s': %s",
                source_id,
                exc,
            )

        texts = [str(chunk.text) for chunk in chunks]

        logging.info(
            "Embedding %d chunks for '%s'",
            len(texts),
            source_id,
        )

        embeddings = self.embed_texts(texts)

        ids = [f"{source_id}_chunk_{chunk.chunk_id}" for chunk in chunks]

        metadatas = [
            {
                "source": str(chunk.source),
                "source_id": str(source_id),
                "chunk_id": int(chunk.chunk_id),
                "token_count": int(chunk.token_count),
            }
            for chunk in chunks
        ]

        upsert_batch_size = 100

        for start in range(0, len(chunks), upsert_batch_size):
            end = start + upsert_batch_size

            collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

        return collection

    def dense_search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Tuple[str, dict, float]]:
        """
        Perform dense vector similarity search against indexed chunks.
        """
        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception:
            return []

        collection_count = collection.count()

        if collection_count == 0:
            return []

        query_embedding = self.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection_count),
            include=["documents", "metadatas", "distances"],
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        search_results: List[Tuple[str, dict, float]] = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            similarity_score = max(
                0.0,
                min(
                    1.0,
                    1.0 - float(distance),
                ),
            )

            search_results.append(
                (
                    str(document),
                    dict(metadata or {}),
                    similarity_score,
                )
            )

        return search_results
