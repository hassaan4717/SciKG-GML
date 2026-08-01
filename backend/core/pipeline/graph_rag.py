import os
import ast
import json
import pickle
import logging
import hashlib
import numpy as np
import networkx as nx
from pathlib import Path
from dotenv import load_dotenv
from networkx.algorithms.community import greedy_modularity_communities
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

from .models import TextChunk
from .graph_utils import GraphBuilder
from .vector_store import VectorStore
from .api_gateway import APIRequestGateway
from .community_summaries import CommunitySummaryMixin
from .text_utils import chunk_blocks, clean_text, parse_text
from .retrieval import BM25Index, Reranker, reciprocal_rank_fusion

from core.utils import safe_float

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

_DEFAULT_TOP_N = max(1, int(os.getenv("RAG_TOP_N", "5")))
_DEFAULT_TOP_K = max(1, int(os.getenv("RAG_TOP_K", "20")))
_DEFAULT_OVERLAP = max(0, int(os.getenv("RAG_OVERLAP", "80")))
_DEFAULT_CHUNK_SIZE = max(1, int(os.getenv("RAG_CHUNK_SIZE", "500")))
RAG_TOP_SUMMARY = max(1, int(os.getenv("RAG_TOP_SUMMARY", "3")))


class GraphRAG(CommunitySummaryMixin):
    def __init__(
        self,
        persist_dir: str,
        api_gateway: APIRequestGateway,
    ) -> None:
        """
        Initialize the GraphRAG pipeline and load any existing cached state.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = _DEFAULT_CHUNK_SIZE
        self.overlap = _DEFAULT_OVERLAP
        self.top_k_retrieve = _DEFAULT_TOP_K
        self.top_n_rerank = _DEFAULT_TOP_N

        if self.overlap >= self.chunk_size:
            raise ValueError("RAG_OVERLAP must be smaller than RAG_CHUNK_SIZE.")

        self.api_gateway = api_gateway

        self.vector_store = VectorStore(
            persist_dir=str(self.persist_dir / "chroma"),
            collection_name="graphrag",
            api_gateway=self.api_gateway,
        )
        self.bm25_index = BM25Index()
        self.reranker = Reranker()
        self.graph_builder = GraphBuilder()

        self._all_chunks: List[TextChunk] = []
        self._ingested_sources: Dict[str, Dict[str, Any]] = {}
        self._communities: Dict[int, Dict[str, Any]] = {}
        self._community_summary_embeddings: Optional[np.ndarray] = None
        self._community_ids_ordered: List[int] = []

        self.state_loaded = self.load_state()

    # ====================== Ingestion ======================

    def ingest(
        self,
        file_path: str,
    ) -> Dict[str, Any]:
        """
        Ingest a text file, chunk it, and build the vector, BM25, and graph indexes.
        """
        source_name = Path(file_path).stem

        chunks_path = self.persist_dir / f"{self._hash(source_name)}_chunks.json"
        stats: Dict[str, Any] = {"source": source_name}

        if chunks_path.exists():
            logging.info(f"Loading cached chunks for: {source_name}")
            source_chunks = self._load_source_chunks(chunks_path)
            stats["cached"] = True
        else:
            logging.info(f"Processing: {source_name}")
            blocks = parse_text(file_path)
            stats["raw_blocks"] = len(blocks)
            blocks = clean_text(blocks)
            stats["clean_blocks"] = len(blocks)
            source_chunks = chunk_blocks(
                blocks, self.chunk_size, self.overlap, source_name
            )
            stats["cached"] = False
            self._save_source_chunks(source_chunks, chunks_path)
            self.vector_store.index_chunks(source_chunks, self._hash(source_name))

        stats["chunks"] = len(source_chunks)

        self._all_chunks = [
            chunk for chunk in self._all_chunks if str(chunk.source) != source_name
        ]
        self._all_chunks.extend(source_chunks)
        self._ingested_sources[source_name] = stats

        logging.info(f"Rebuilding BM25 over {len(self._all_chunks)} total chunks")
        self.bm25_index.build(self._all_chunks)

        logging.info(
            "Extracting entities and rebuilding graph over %d total chunks",
            len(self._all_chunks),
        )
        self.graph_builder.graph.clear()
        self.graph_builder.entity_to_chunks.clear()
        self.graph_builder.build_graph(self._all_chunks)

        logging.info("Detecting communities…")
        self._detect_communities()
        self.generate_summaries()

        stats.update(
            {
                "entities": self.graph_builder.graph.number_of_nodes(),
                "edges": self.graph_builder.graph.number_of_edges(),
                "communities": len(self._communities),
                "total_chunks": len(self._all_chunks),
                "total_sources": len(self._ingested_sources),
            }
        )

        self.save_state()
        return stats

    # ====================== Community Detection ======================

    def _detect_communities(self) -> None:
        """
        Detect thematic graph communities and retain their supporting chunks
        and internal relationships.
        """
        graph = self.graph_builder.graph

        if graph.number_of_nodes() == 0:
            self._communities = {}
            self._community_summary_embeddings = None
            self._community_ids_ordered = []
            return

        detection_graph = graph.to_undirected() if graph.is_directed() else graph

        all_communities: List[frozenset] = []
        components = nx.connected_components(detection_graph)

        for component in components:
            subgraph = detection_graph.subgraph(component)

            if subgraph.number_of_nodes() == 1:
                all_communities.append(frozenset(component))
                continue

            try:
                communities = greedy_modularity_communities(
                    subgraph,
                    weight="weight",
                )
                all_communities.extend(communities)
            except Exception:
                logging.exception(
                    "Community detection failed for a component with %d nodes",
                    subgraph.number_of_nodes(),
                )
                all_communities.append(frozenset(component))

        detected: List[Dict[str, Any]] = []

        for members in all_communities:
            members_sorted = sorted(
                members,
                key=lambda entity: (
                    -self._get_node_frequency(str(entity)),
                    str(entity).casefold(),
                ),
            )

            chunk_keys = set()
            for entity in members_sorted:
                chunk_keys.update(
                    self.graph_builder.entity_to_chunks.get(entity, set())
                )

            detected.append(
                {
                    "entities": members_sorted,
                    "chunk_keys": self._normalize_chunk_keys(chunk_keys),
                    "relationships": self._get_community_relationships(members_sorted),
                    "summary": None,
                }
            )

        detected.sort(
            key=lambda community: (
                -len(community["entities"]),
                (
                    str(community["entities"][0]).casefold()
                    if community["entities"]
                    else ""
                ),
            )
        )
        self._communities = {
            community_id: community for community_id, community in enumerate(detected)
        }

        self._community_summary_embeddings = None
        self._community_ids_ordered = []

    # ====================== Community Helpers ======================

    def _get_node_frequency(self, entity: str) -> float:
        """
        Return the stored mention frequency for one graph entity.
        """
        graph = self.graph_builder.graph

        if entity not in graph:
            return 0.0

        data = graph.nodes[entity]
        value = data.get(
            "mention_count",
            data.get("count", data.get("chunk_count", 0)),
        )
        return safe_float(value, default=0.0)

    def _get_community_relationships(
        self,
        entities: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """
        Return weighted relationships between entities in one community.
        """
        graph = self.graph_builder.graph
        community_nodes = [entity for entity in entities if entity in graph]

        if len(community_nodes) < 2:
            return []

        subgraph = graph.subgraph(community_nodes)
        relationships: List[Dict[str, Any]] = []

        if subgraph.is_multigraph():
            edge_iterator = (
                (source, target, edge_key, data)
                for source, target, edge_key, data in subgraph.edges(
                    keys=True, data=True
                )
            )
        else:
            edge_iterator = (
                (source, target, None, data)
                for source, target, data in subgraph.edges(data=True)
            )

        for source, target, edge_key, raw_data in edge_iterator:
            data = dict(raw_data or {})
            description = " ".join(str(data.get("description", "")).split())

            relationship: Dict[str, Any] = {
                "source": str(source),
                "target": str(target),
                "type": str(data.get("type") or data.get("relation") or "related_to"),
                "weight": safe_float(data.get("weight", 1.0)),
                "description": description,
                "directed": graph.is_directed(),
            }

            if edge_key is not None:
                relationship["edge_key"] = str(edge_key)

            relationships.append(relationship)

        relationships.sort(
            key=lambda relationship: (
                -relationship["weight"],
                relationship["source"].casefold(),
                relationship["target"].casefold(),
            )
        )

        return relationships

    # ====================== Query ======================

    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute a hybrid query combining local and global retrieval.
        """
        question = str(question).strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        return self._hybrid_query(question)

    def _local_query(self, question: str) -> Dict[str, Any]:
        """
        Retrieve context using dense/sparse retrieval and local graph neighborhood expansion.
        """
        dense = self.vector_store.dense_search(question, self.top_k_retrieve)
        sparse = self.bm25_index.search(question, self.top_k_retrieve)
        fused = reciprocal_rank_fusion(dense, sparse)

        query_ents = set(self._extract_graph_entities(question))
        chunk_ents = set()
        for text, _, _ in fused[:5]:
            chunk_ents.update(self._extract_graph_entities(text))

        seed_ents = query_ents | chunk_ents
        neighbor_ents = set()

        for ent in seed_ents:
            if ent in self.graph_builder.graph:
                nbrs = sorted(
                    self.graph_builder.graph.neighbors(ent),
                    key=lambda n: self.graph_builder.graph[ent][n].get("weight", 1),
                    reverse=True,
                )
                neighbor_ents.update(nbrs[:5])

        chunk_lookup = self._make_chunk_lookup()
        extra_hits: List[Tuple[str, Dict[str, Any], float]] = []
        for ent in neighbor_ents:
            chunk_keys = sorted(
                self.graph_builder.entity_to_chunks.get(ent, set()),
                key=str,
            )

            for key in chunk_keys[:20]:
                c = self._lookup_chunk(chunk_lookup, key)

                if c:
                    extra_hits.append(
                        (
                            c.text,
                            {
                                "source": c.source,
                                "chunk_id": c.chunk_id,
                                "token_count": c.token_count,
                            },
                            0.4,
                        )
                    )

        combined = self._dedup(fused + extra_hits)
        reranked = self.reranker.rerank(question, combined, self.top_n_rerank)

        return {
            "hits": reranked,
            "mode": "local",
            "summaries": [],
            "entities": sorted(list(seed_ents | neighbor_ents)),
        }

    def _global_query(self, question: str) -> Dict[str, Any]:
        """
        Retrieve communities by cosine similarity to their summaries.
        """
        if (
            self._community_summary_embeddings is None
            or not self._community_ids_ordered
        ):
            return {
                "hits": [],
                "mode": "global_unavailable",
                "summaries": [],
                "entities": [],
            }

        query_embedding = np.asarray(
            self.vector_store.embed_query(question),
            dtype=np.float32,
        )
        matrix = self._community_summary_embeddings

        query_norm = float(np.linalg.norm(query_embedding))
        row_norms = np.linalg.norm(matrix, axis=1)
        denominator = row_norms * query_norm
        similarities = np.divide(
            matrix @ query_embedding,
            denominator,
            out=np.zeros_like(row_norms, dtype=np.float32),
            where=denominator > 0,
        )
        top_indices = np.argsort(similarities)[::-1][:RAG_TOP_SUMMARY]

        chunk_lookup = self._make_chunk_lookup()
        selected_hits: List[Tuple[str, Dict[str, Any], float]] = []
        selected_summaries: List[str] = []

        for index in top_indices:
            community_id = self._community_ids_ordered[int(index)]
            community_data = self._communities.get(community_id, {})
            if not community_data:
                continue

            title = str(community_data.get("title") or "").strip()
            summary = str(community_data.get("summary") or "").strip()

            if summary:
                selected_summaries.append(f"{title}: {summary}" if title else summary)

            chunk_keys = (
                community_data.get("summary_chunk_keys")
                or community_data.get("chunk_keys")
                or []
            )
            normalized_chunk_keys = self._normalize_chunk_keys(chunk_keys)

            for chunk_key in normalized_chunk_keys[:6]:
                chunk = self._lookup_chunk(chunk_lookup, chunk_key)
                if chunk:
                    selected_hits.append(
                        (
                            chunk.text,
                            {
                                "source": chunk.source,
                                "chunk_id": chunk.chunk_id,
                                "token_count": chunk.token_count,
                            },
                            float(similarities[int(index)]),
                        )
                    )

        selected_hits = self._dedup(selected_hits)
        if len(selected_hits) >= 2:
            reranked = self.reranker.rerank(
                question,
                selected_hits,
                self.top_n_rerank,
            )
        else:
            reranked = selected_hits[: self.top_n_rerank]

        return {
            "hits": reranked,
            "mode": "global",
            "summaries": selected_summaries,
            "entities": [],
        }

    def _hybrid_query(self, question: str) -> Dict[str, Any]:
        """
        Combine local and global query results, deduplicate, and rerank.
        """
        local = self._local_query(question)
        global_res = self._global_query(question)

        if not global_res["hits"] and not global_res["summaries"]:
            local["mode"] = "hybrid_local_fallback"
            return local

        seen = set()
        combined: List[Tuple[str, Dict[str, Any], float]] = []

        for text, meta, score in local["hits"]:
            key = self._chunk_key(
                meta.get("source", ""),
                meta.get("chunk_id", ""),
            )
            seen.add(key)
            combined.append((text, meta, score))

        for text, meta, score in global_res["hits"]:
            key = self._chunk_key(
                meta.get("source", ""),
                meta.get("chunk_id", ""),
            )
            if key not in seen:
                seen.add(key)
                combined.append((text, meta, score * 0.8))

        reranked = self.reranker.rerank(question, combined[:25], self.top_n_rerank)

        return {
            "hits": reranked,
            "mode": "hybrid",
            "summaries": global_res["summaries"],
            "entities": local["entities"],
        }

    def format_context(self, result: Dict[str, Any]) -> str:
        """
        Format retrieved context using stable numeric source references.
        """
        parts: List[str] = []

        summaries = result.get("summaries", [])

        if summaries:
            summary_text = "\n\n".join(
                f"- {summary}" for summary in summaries if summary
            )

            if summary_text:
                parts.append(
                    f"## Knowledge Graph Community Summaries\n\n{summary_text}"
                )

        hits = result.get("hits", [])

        for index, (text, meta, score) in enumerate(hits, start=1):
            source = str(meta.get("source", "unknown")).strip()

            parts.append(
                f'<source id="{index}" name="{source}">\n{text.strip()}\n</source>'
            )

        return "\n\n---\n\n".join(parts) if parts else "No relevant content found."

    # ====================== Graph Analytics ======================

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Return basic statistics about the current knowledge graph.
        """
        g = self.graph_builder.graph
        return {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "communities": len(self._communities),
            "density": round(nx.density(g), 4) if g.number_of_nodes() > 1 else 0.0,
        }

    def get_top_entities(self, n: int = 25) -> List[Tuple[str, int]]:
        """
        Return the top N most frequently mentioned entities in the graph.
        """
        return sorted(
            [
                (str(node), int(self._get_node_frequency(str(node))))
                for node in self.graph_builder.graph.nodes
            ],
            key=lambda item: (-item[1], item[0].casefold()),
        )[:n]

    def get_community_list(self) -> List[Dict[str, Any]]:
        """
        Return a list of all communities with their metadata and summaries.
        """
        return [
            {
                "id": cid,
                "size": len(cd["entities"]),
                "top_entities": cd["entities"][:7],
                "summary": cd.get("summary") or "No summary yet.",
                "num_chunks": len(cd["chunk_keys"]),
            }
            for cid, cd in self._communities.items()
        ]

    def save_state(self) -> None:
        """
        Persist the graph, communities, and ingestion metadata to disk.
        """
        state = {
            "graph": self.graph_builder.graph,
            "entity_to_chunks": self.graph_builder.entity_to_chunks,
            "communities": self._communities,
            "ingested_sources": self._ingested_sources,
        }
        path = self.persist_dir / "graph_state.pkl"

        with path.open("wb") as file:
            pickle.dump(state, file)

        logging.info(
            f"Graph state saved ({self.graph_builder.graph.number_of_nodes()} nodes, "
            f"{len(self._communities)} communities)."
        )

    def load_state(self) -> bool:
        """
        Restore the graph and metadata from disk, rebuilding caches as needed.
        """
        path = self.persist_dir / "graph_state.pkl"

        try:
            if not path.is_file():
                return False

            with path.open("rb") as file:
                state = pickle.load(file)

            self.graph_builder.graph = state["graph"]
            self.graph_builder.entity_to_chunks = state["entity_to_chunks"]
            self._communities = state["communities"]
            self._ingested_sources = state.get("ingested_sources", {})

            self._rebuild_chunks_from_cache()

            if self.has_summaries():
                self._cache_summary_embeddings()

            logging.info("Graph state restored successfully.")
            return True
        except Exception:
            logging.exception("Failed to load GraphRAG state.")
            return False

    def _rebuild_chunks_from_cache(self) -> None:
        """
        Reconstruct the in-memory chunk list and BM25 index from cached JSON files.
        """
        self._all_chunks = []
        for source_name in self._ingested_sources:
            chunks_path = self.persist_dir / f"{self._hash(source_name)}_chunks.json"
            if chunks_path.exists():
                self._all_chunks.extend(self._load_source_chunks(chunks_path))
        if self._all_chunks:
            self.bm25_index.build(self._all_chunks)

    # ====================== Helpers ======================

    @staticmethod
    def _hash(name: str) -> str:
        """
        Generate a short stable hash used for cache filenames.
        """
        return hashlib.md5(name.encode("utf-8")).hexdigest()[:10]

    @staticmethod
    def _chunk_key(source: Any, chunk_id: Any) -> Tuple[str, str]:
        """
        Create a stable chunk key.
        """
        return str(source), str(chunk_id)

    def _normalize_chunk_keys(
        self,
        keys: Iterable[Hashable],
    ) -> List[Hashable]:
        """
        Normalize and sort chunk references while preserving tuple keys.
        """
        normalized: List[Hashable] = []
        seen = set()

        for raw_key in keys:
            if isinstance(raw_key, (tuple, list)) and len(raw_key) == 2:
                key: Hashable = self._chunk_key(raw_key[0], raw_key[1])
            else:
                key = str(raw_key)

            if key in seen:
                continue

            seen.add(key)
            normalized.append(key)

        return sorted(normalized, key=lambda item: str(item))

    def _make_chunk_lookup(self) -> Dict[Hashable, TextChunk]:
        """
        Create a lookup supporting tuple keys and legacy string keys.
        """
        lookup: Dict[Hashable, TextChunk] = {}

        for chunk in self._all_chunks:
            tuple_key = self._chunk_key(chunk.source, chunk.chunk_id)
            legacy_key = f"{chunk.source}_{chunk.chunk_id}"

            lookup[tuple_key] = chunk
            lookup[legacy_key] = chunk

        return lookup

    def _lookup_chunk(
        self,
        chunk_lookup: Dict[Hashable, TextChunk],
        raw_key: Hashable,
    ) -> Optional[TextChunk]:
        """
        Resolve tuple, list, legacy string, and stringified-tuple keys.
        """
        try:
            chunk = chunk_lookup.get(raw_key)
            if chunk is not None:
                return chunk
        except TypeError:
            pass

        if isinstance(raw_key, (tuple, list)) and len(raw_key) == 2:
            tuple_key = self._chunk_key(raw_key[0], raw_key[1])
            chunk = chunk_lookup.get(tuple_key)

            if chunk is not None:
                return chunk

            return chunk_lookup.get(f"{raw_key[0]}_{raw_key[1]}")

        string_key = str(raw_key)
        chunk = chunk_lookup.get(string_key)

        if chunk is not None:
            return chunk

        try:
            parsed_key = ast.literal_eval(string_key)
        except (ValueError, SyntaxError):
            return None

        if isinstance(parsed_key, (tuple, list)) and len(parsed_key) == 2:
            return chunk_lookup.get(self._chunk_key(parsed_key[0], parsed_key[1]))

        return None

    def _extract_graph_entities(self, text: str) -> List[str]:
        """
        Extract graph-compatible entity identifiers.
        """
        extractor = getattr(self.graph_builder, "extract_entity_ids", None)

        if callable(extractor):
            return list(extractor(text))

        return list(self.graph_builder.extract_entities(text))

    def _dedup(
        self,
        hits: List[Tuple[str, Dict[str, Any], float]],
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Remove duplicate retrieval hits using stable source/chunk keys.
        """
        seen = set()
        deduplicated: List[Tuple[str, Dict[str, Any], float]] = []

        for text, meta, score in hits:
            key = self._chunk_key(
                meta.get("source", ""),
                meta.get("chunk_id", ""),
            )

            if key in seen:
                continue

            seen.add(key)
            deduplicated.append((text, meta, score))

        return deduplicated

    def _save_source_chunks(
        self,
        chunks: List[TextChunk],
        path: Path,
    ) -> None:
        """
        Save TextChunk objects to a JSON cache file.
        """
        data = [chunk.__dict__ for chunk in chunks]
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_source_chunks(self, path: Path) -> List[TextChunk]:
        """
        Load TextChunk objects from a JSON cache file.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        return [TextChunk(**item) for item in data]
