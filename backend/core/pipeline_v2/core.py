from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

import networkx as nx
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------


@dataclass
class Chunk:
    id: str
    source: str
    text: str
    page: int | None = None

    @property
    def citation(self) -> str:
        page = f", page {self.page}" if self.page is not None else ""
        return f"{self.id} ({self.source}{page})"


@dataclass
class Entity:
    id: str
    name: str
    type: str
    descriptions: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)

    @property
    def description(self) -> str:
        values = [x.strip() for x in self.descriptions if isinstance(x, str) and x.strip()]
        return " ".join(list(dict.fromkeys(values))[:6])


@dataclass
class Relationship:
    source: str
    target: str
    type: str
    descriptions: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    weight: int = 1

    @property
    def description(self) -> str:
        values = [x.strip() for x in self.descriptions if isinstance(x, str) and x.strip()]
        return " ".join(list(dict.fromkeys(values))[:6])


@dataclass
class Community:
    id: str
    level: int
    entity_ids: list[str]
    parent_id: str | None = None


@dataclass
class CommunityReport:
    community_id: str
    title: str
    summary: str
    findings: list[str]
    entity_ids: list[str]
    chunk_ids: list[str]

    @property
    def text(self) -> str:
        findings = "\n".join(f"- {item}" for item in self.findings)
        return f"{self.title}\n{self.summary}\n{findings}".strip()


@dataclass
class Config:
    # API LLM
    llm_model: str
    llm_base_url: str
    embedding_model: str
    llm_api_key: str
    llm_provider: str = "openai_compatible"
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 3

    # Chunking
    chunk_size_words: int = 650
    chunk_overlap_words: int = 100

    # Generation
    max_new_tokens: int = 900
    temperature: float = 0.1

    # Graph communities
    max_community_levels: int = 3
    min_community_size: int = 5
    community_resolution: float = 1.0

    # DRIFT-inspired search
    top_seed_communities: int = 5
    top_seed_entities: int = 8
    top_chunks_per_step: int = 8
    neighbor_hops: int = 1
    drift_depth: int = 2
    followups_per_step: int = 3

    @classmethod
    def from_path(cls, path: Path) -> "Config":
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        return cls(**json.loads(path.read_text(encoding="utf-8")))


# -----------------------------------------------------------------------------
# API-backed LLM and local embeddings
# -----------------------------------------------------------------------------


class LLMClient(Protocol):
    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_new_tokens: int | None = None,
    ) -> str:
        ...


class OpenAICompatibleLLM:
    """OpenAI-compatible API client. Embeddings and graph processing stay local."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        max_new_tokens: int = 900,
        temperature: float = 0.1,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        if not base_url.strip():
            raise ValueError("base_url cannot be empty.")

        if not api_key.strip():
            raise ValueError("api_key cannot be empty.")

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_retries = max(1, max_retries)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = (
            "You are a precise information extraction and synthesis system."
        ),
        max_new_tokens: int | None = None,
    ) -> str:
        if not user_prompt.strip():
            raise ValueError("user_prompt cannot be empty.")

        output_limit = max_new_tokens or self.max_new_tokens
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    temperature=self.temperature,
                    max_tokens=output_limit,
                )

                if not response.choices:
                    raise RuntimeError(
                        "The LLM API returned no choices."
                    )

                choice = response.choices[0]
                content = choice.message.content

                if not content or not content.strip():
                    raise RuntimeError(
                        "The LLM API returned an empty response."
                    )

                if choice.finish_reason == "length":
                    raise RuntimeError(
                        f"The model response was truncated at "
                        f"{output_limit} output tokens. "
                        f"Increase max_new_tokens or reduce chunk size."
                    )

                return content.strip()

            except Exception as exc:
                last_error = exc

                if attempt >= self.max_retries:
                    break

                wait_seconds = min(2 ** attempt, 10)
                time.sleep(wait_seconds)

        raise RuntimeError(
            f"LLM request failed after "
            f"{self.max_retries} attempts: {last_error}"
        ) from last_error


class LocalEmbedder:
    """Local SentenceTransformer embeddings."""

    def __init__(self, model_name: str) -> None:
        if not model_name.strip():
            raise ValueError("embedding_model cannot be empty")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.uses_e5_prefixes = "e5" in model_name.casefold()

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        prepared = [f"passage: {text}" for text in texts] if self.uses_e5_prefixes else texts
        vectors = self.model.encode(
            prepared,
            batch_size=batch_size,
            show_progress_bar=len(texts) > batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Query cannot be empty")
        prepared = f"query: {text}" if self.uses_e5_prefixes else text
        vector = self.model.encode(
            [prepared],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vector[0], dtype=np.float32)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def normalize_entity_name(name: str) -> str:
    value = name.casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def stable_entity_id(name: str) -> str:
    normalized = normalize_entity_name(name)
    safe = re.sub(r"\W+", "_", normalized, flags=re.UNICODE).strip("_")
    return safe[:120] or "unnamed_entity"


def extract_json_object(text: str) -> dict[str, Any]:
    candidates: list[str] = []
    candidates.extend(
        re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    )
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first : last + 1])
    candidates.append(text.strip())

    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Model did not return valid JSON:\n{text[:1200]}")


def cosine_top_k(
    query_vector: np.ndarray,
    matrix: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    if matrix.size == 0 or len(matrix) == 0 or k <= 0:
        return []
    scores = matrix @ query_vector
    k = min(k, len(scores))
    if k == len(scores):
        ranked = np.argsort(scores)[::-1]
    else:
        candidates = np.argpartition(scores, -k)[-k:]
        ranked = candidates[np.argsort(scores[candidates])[::-1]]
    return [(int(i), float(scores[i])) for i in ranked[:k]]


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# Document ingestion and chunking
# -----------------------------------------------------------------------------


def read_documents(input_dir: Path) -> list[tuple[str, str, int | None]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    records: list[tuple[str, str, int | None]] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                records.append((path.name, text, None))
        elif suffix == ".pdf":
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    records.append((path.name, text, page_number))
    return records


def chunk_documents(
    records: list[tuple[str, str, int | None]],
    chunk_size_words: int,
    overlap_words: int,
) -> list[Chunk]:
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if chunk_size_words <= overlap_words:
        raise ValueError("chunk_size_words must be greater than overlap_words")

    chunks: list[Chunk] = []
    step = chunk_size_words - overlap_words
    for source, text, page in records:
        words = text.split()
        for start in range(0, len(words), step):
            piece = words[start : start + chunk_size_words]
            if not piece:
                continue
            chunks.append(
                Chunk(
                    id=f"CHUNK-{len(chunks):06d}",
                    source=source,
                    page=page,
                    text=" ".join(piece),
                )
            )
            if start + chunk_size_words >= len(words):
                break
    return chunks


# -----------------------------------------------------------------------------
# Entity and relationship extraction
# -----------------------------------------------------------------------------


EXTRACTION_SYSTEM = """
You extract a compact knowledge graph from source text.
Only extract entities and relationships directly supported by the text.
Never invent missing facts. Return strict JSON only.
""".strip()


def extract_chunk_graph(
    llm: LLMClient,
    chunk: Chunk,
) -> dict[str, Any]:
    prompt = f"""
Extract a compact knowledge graph from the following source text.

Return exactly this JSON structure:

{{
  "entities": [
    {{
      "name": "...",
      "type": "...",
      "description": "..."
    }}
  ],
  "relationships": [
    {{
      "source": "...",
      "target": "...",
      "type": "...",
      "description": "..."
    }}
  ]
}}

Strict rules:

1. Extract at most 12 entities.
2. Extract at most 18 relationships.
3. Entity descriptions must be no longer than 20 words.
4. Relationship descriptions must be no longer than 20 words.
5. Only include important entities needed to understand the passage.
6. Relationship source and target names must exactly match entity names.
7. Do not use knowledge that is absent from the source text.
8. Do not output markdown.
9. Do not output explanations before or after the JSON.
10. Always close every JSON array and object.

Chunk ID: {chunk.id}
Source: {chunk.source}
Page: {chunk.page}

Text:
{chunk.text}
""".strip()

    response = llm.generate(
        user_prompt=prompt,
        system_prompt=EXTRACTION_SYSTEM,
        max_new_tokens=2200,
    )

    result = extract_json_object(response)

    entities = result.get("entities", [])
    relationships = result.get("relationships", [])

    if not isinstance(entities, list):
        entities = []

    if not isinstance(relationships, list):
        relationships = []

    return {
        "entities": entities[:12],
        "relationships": relationships[:18],
    }


def merge_extractions(
    chunks: list[Chunk],
    extractions: dict[str, dict[str, Any]],
) -> tuple[dict[str, Entity], dict[tuple[str, str, str], Relationship]]:
    entities: dict[str, Entity] = {}
    relationships: dict[tuple[str, str, str], Relationship] = {}

    for chunk in chunks:
        extraction = extractions.get(chunk.id, {})
        name_to_id: dict[str, str] = {}

        for raw in extraction.get("entities", []):
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            if not name:
                continue
            entity_id = stable_entity_id(name)
            name_to_id[normalize_entity_name(name)] = entity_id
            entity = entities.get(entity_id)
            if entity is None:
                entity = Entity(
                    id=entity_id,
                    name=name,
                    type=str(raw.get("type", "UNKNOWN")).strip().upper() or "UNKNOWN",
                )
                entities[entity_id] = entity
            description = str(raw.get("description", "")).strip()
            if description and description not in entity.descriptions:
                entity.descriptions.append(description)
            if chunk.id not in entity.chunk_ids:
                entity.chunk_ids.append(chunk.id)

        for raw in extraction.get("relationships", []):
            if not isinstance(raw, dict):
                continue
            source_name = str(raw.get("source", "")).strip()
            target_name = str(raw.get("target", "")).strip()
            relation_type = str(raw.get("type", "RELATED_TO")).strip().upper() or "RELATED_TO"
            source_id = name_to_id.get(normalize_entity_name(source_name))
            target_id = name_to_id.get(normalize_entity_name(target_name))
            if not source_id or not target_id or source_id == target_id:
                continue
            key = (source_id, target_id, relation_type)
            relation = relationships.get(key)
            if relation is None:
                relation = Relationship(source=source_id, target=target_id, type=relation_type, weight=0)
                relationships[key] = relation
            relation.weight += 1
            description = str(raw.get("description", "")).strip()
            if description and description not in relation.descriptions:
                relation.descriptions.append(description)
            if chunk.id not in relation.chunk_ids:
                relation.chunk_ids.append(chunk.id)

    return entities, relationships


def make_graph(
    entities: dict[str, Entity],
    relationships: dict[tuple[str, str, str], Relationship],
) -> nx.Graph:
    graph = nx.Graph()
    for entity in entities.values():
        graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.type,
            description=entity.description,
        )

    for relation in relationships.values():
        if relation.source not in graph or relation.target not in graph:
            continue
        if graph.has_edge(relation.source, relation.target):
            edge = graph[relation.source][relation.target]
            edge["weight"] = float(edge.get("weight", 0)) + relation.weight
            edge.setdefault("types", [])
            if relation.type not in edge["types"]:
                edge["types"].append(relation.type)
            edge.setdefault("descriptions", [])
            if relation.description:
                edge["descriptions"].append(relation.description)
        else:
            graph.add_edge(
                relation.source,
                relation.target,
                weight=float(relation.weight),
                types=[relation.type],
                descriptions=[relation.description] if relation.description else [],
            )
    return graph


# -----------------------------------------------------------------------------
# Community detection
# -----------------------------------------------------------------------------


def leiden_partition(graph: nx.Graph, resolution: float) -> list[list[str]]:
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return [[node] for node in graph.nodes]

    try:
        import igraph as ig
        import leidenalg

        nodes = list(graph.nodes)
        node_index = {node: i for i, node in enumerate(nodes)}
        edges = [(node_index[u], node_index[v]) for u, v in graph.edges]
        weights = [float(graph[u][v].get("weight", 1.0)) for u, v in graph.edges]
        ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=resolution,
            seed=42,
        )
        return [[nodes[i] for i in community] for community in partition]
    except Exception:
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=resolution,
            seed=42,
        )
        return [sorted(list(c)) for c in communities]


def build_hierarchical_communities(
    graph: nx.Graph,
    max_levels: int,
    min_community_size: int,
    resolution: float,
) -> list[Community]:
    if max_levels <= 0:
        return []

    all_communities: list[Community] = []
    current: list[tuple[str | None, list[str]]] = [(None, list(graph.nodes))]
    counters: dict[int, int] = defaultdict(int)

    for level in range(max_levels):
        next_level: list[tuple[str | None, list[str]]] = []
        for parent_id, nodes in current:
            if not nodes:
                continue
            partitions = leiden_partition(graph.subgraph(nodes).copy(), resolution)
            if level > 0 and len(partitions) == 1 and set(partitions[0]) == set(nodes):
                continue

            for members in partitions:
                community_id = f"L{level}-C{counters[level]:05d}"
                counters[level] += 1
                all_communities.append(
                    Community(
                        id=community_id,
                        level=level,
                        entity_ids=sorted(members),
                        parent_id=parent_id,
                    )
                )
                if level + 1 < max_levels and len(members) >= max(2 * min_community_size, 4):
                    next_level.append((community_id, members))
        current = next_level
        if not current:
            break
    return all_communities


# -----------------------------------------------------------------------------
# Community reports
# -----------------------------------------------------------------------------


REPORT_SYSTEM = """
You write evidence-grounded community reports for a GraphRAG index.
Use only the supplied graph elements and source excerpts.
Return strict JSON only. Never invent facts.
""".strip()


def build_community_context(
    community: Community,
    graph: nx.Graph,
    entities: dict[str, Entity],
    relationships: dict[tuple[str, str, str], Relationship],
    chunks_by_id: dict[str, Chunk],
    max_entities: int = 25,
    max_edges: int = 35,
    max_chunks: int = 12,
) -> tuple[str, list[str]]:
    ranked_ids = sorted(
        community.entity_ids,
        key=lambda node: graph.degree(node, weight="weight"),
        reverse=True,
    )
    selected_ids = ranked_ids[:max_entities]
    selected = set(selected_ids)
    entity_lines: list[str] = []
    related_chunk_ids: list[str] = []

    for entity_id in selected_ids:
        entity = entities.get(entity_id)
        if entity is None:
            continue
        entity_lines.append(f"- {entity.name} [{entity.type}]: {entity.description}")
        related_chunk_ids.extend(entity.chunk_ids)

    edge_candidates = [
        relation
        for relation in relationships.values()
        if relation.source in selected and relation.target in selected
    ]
    edge_candidates.sort(key=lambda x: x.weight, reverse=True)
    edge_lines: list[str] = []

    for relation in edge_candidates[:max_edges]:
        source = entities[relation.source].name
        target = entities[relation.target].name
        edge_lines.append(
            f"- {source} --{relation.type}--> {target}: {relation.description}"
        )
        related_chunk_ids.extend(relation.chunk_ids)

    unique_chunk_ids = list(dict.fromkeys(related_chunk_ids))[:max_chunks]
    chunk_lines: list[str] = []
    for chunk_id in unique_chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        page = f", page {chunk.page}" if chunk.page is not None else ""
        chunk_lines.append(
            f"[{chunk.id}] {chunk.source}{page}\n{chunk.text[:900]}"
        )

    context = (
        "ENTITIES\n"
        + "\n".join(entity_lines)
        + "\n\nRELATIONSHIPS\n"
        + "\n".join(edge_lines)
        + "\n\nSOURCE EXCERPTS\n"
        + "\n\n".join(chunk_lines)
    )
    return context, unique_chunk_ids


def generate_community_report(
    llm: LLMClient,
    community: Community,
    context: str,
    chunk_ids: list[str],
) -> CommunityReport:
    prompt = f"""
Create a compact report for this graph community.

Return exactly:
{{
  "title": "short thematic title",
  "summary": "one concise paragraph",
  "findings": [
    "finding with supporting chunk citations such as [CHUNK-000001]"
  ]
}}

Requirements:
- Include 3 to 7 findings when evidence permits.
- Cite supporting chunk IDs in each finding.
- Describe the community as a coherent theme.
- Use only the supplied context.
- Output JSON only.

Community ID: {community.id}

Context:
{context}
""".strip()
    raw = extract_json_object(llm.generate(prompt, system_prompt=REPORT_SYSTEM))
    findings_raw = raw.get("findings", [])
    findings = (
        [str(x).strip() for x in findings_raw if str(x).strip()]
        if isinstance(findings_raw, list)
        else []
    )
    return CommunityReport(
        community_id=community.id,
        title=str(raw.get("title", community.id)).strip() or community.id,
        summary=str(raw.get("summary", "")).strip(),
        findings=findings,
        entity_ids=community.entity_ids,
        chunk_ids=chunk_ids,
    )


# -----------------------------------------------------------------------------
# Index build/load
# -----------------------------------------------------------------------------


class GraphRAGIndex:
    def __init__(
        self,
        root: Path,
        config: Config,
        llm: LLMClient | None = None,
        embedder: LocalEmbedder | None = None,
    ) -> None:
        self.root = root
        self.config = config
        self.input_dir = root / "input"
        self.index_dir = root / "index"
        self.llm = llm
        self.embedder = embedder

    def build(self) -> None:
        if self.llm is None:
            raise RuntimeError("Index building requires an LLM client")
        if self.embedder is None:
            raise RuntimeError("Index building requires a local embedder")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        records = read_documents(self.input_dir)
        chunks = chunk_documents(
            records,
            self.config.chunk_size_words,
            self.config.chunk_overlap_words,
        )
        if not chunks:
            raise RuntimeError(f"No readable documents found in {self.input_dir}")

        save_json(self.index_dir / "chunks.json", [asdict(x) for x in chunks])
        extraction_path = self.index_dir / "extractions.json"
        extractions: dict[str, dict[str, Any]] = (
            load_json(extraction_path) if extraction_path.exists() else {}
        )

        for chunk in tqdm(
            chunks,
            desc="Extracting graph elements",
        ):
            existing = extractions.get(chunk.id)

            if (
                existing is not None
                and "_error" not in existing
                and existing.get("entities") is not None
                and existing.get("relationships") is not None
            ):
                continue

            try:
                extractions[chunk.id] = extract_chunk_graph(
                    self.llm,
                    chunk,
                )

            except Exception as exc:
                extractions[chunk.id] = {
                    "entities": [],
                    "relationships": [],
                    "_error": str(exc),
                }

            save_json(
                extraction_path,
                extractions,
            ) 

        entities, relationships = merge_extractions(chunks, extractions)
        graph = make_graph(entities, relationships)

        save_json(self.index_dir / "entities.json", [asdict(x) for x in entities.values()])
        save_json(
            self.index_dir / "relationships.json",
            [asdict(x) for x in relationships.values()],
        )

        communities = build_hierarchical_communities(
            graph,
            max_levels=self.config.max_community_levels,
            min_community_size=self.config.min_community_size,
            resolution=self.config.community_resolution,
        )
        save_json(self.index_dir / "communities.json", [asdict(x) for x in communities])

        chunks_by_id = {x.id: x for x in chunks}
        report_checkpoint = self.index_dir / "community_reports_checkpoint.json"
        existing_raw = load_json(report_checkpoint) if report_checkpoint.exists() else []
        existing = {
            raw["community_id"]: CommunityReport(**raw)
            for raw in existing_raw
        }
        reports: list[CommunityReport] = []

        for community in tqdm(communities, desc="Generating community reports"):
            if community.id in existing:
                reports.append(existing[community.id])
                continue
            context, chunk_ids = build_community_context(
                community,
                graph,
                entities,
                relationships,
                chunks_by_id,
            )
            report = generate_community_report(self.llm, community, context, chunk_ids)
            reports.append(report)
            existing[community.id] = report
            save_json(report_checkpoint, [asdict(x) for x in existing.values()])

        save_json(
            self.index_dir / "community_reports.json",
            [asdict(x) for x in reports],
        )

        entity_items = list(entities.values())
        chunk_vectors = self.embedder.encode_documents([x.text for x in chunks])
        entity_vectors = self.embedder.encode_documents(
            [f"{x.name}. {x.description}" for x in entity_items]
        )
        report_vectors = self.embedder.encode_documents([x.text for x in reports])

        np.save(self.index_dir / "chunk_vectors.npy", chunk_vectors)
        np.save(self.index_dir / "entity_vectors.npy", entity_vectors)
        np.save(self.index_dir / "report_vectors.npy", report_vectors)
        save_json(
            self.index_dir / "vector_order.json",
            {
                "chunks": [x.id for x in chunks],
                "entities": [x.id for x in entity_items],
                "reports": [x.community_id for x in reports],
            },
        )

        save_json(
            self.index_dir / "graph.json",
            nx.node_link_data(graph),
        )

    def load(self) -> dict[str, Any]:
        chunks = {
            raw["id"]: Chunk(**raw)
            for raw in load_json(
                self.index_dir / "chunks.json"
            )
        }

        entities = {
            raw["id"]: Entity(**raw)
            for raw in load_json(
                self.index_dir / "entities.json"
            )
        }

        relationship_items = [
            Relationship(**raw)
            for raw in load_json(
                self.index_dir / "relationships.json"
            )
        ]

        relationships = {
            (
                relationship.source,
                relationship.target,
                relationship.type,
            ): relationship
            for relationship in relationship_items
        }

        communities = {
            raw["id"]: Community(**raw)
            for raw in load_json(
                self.index_dir / "communities.json"
            )
        }

        reports = {
            raw["community_id"]: CommunityReport(**raw)
            for raw in load_json(
                self.index_dir / "community_reports.json"
            )
        }

        # Load the NetworkX graph from node-link JSON.
        # JSON supports list-valued attributes such as:
        # types=["RELATED_TO"] and descriptions=["..."].
        graph_data = load_json(
            self.index_dir / "graph.json"
        )

        graph = nx.node_link_graph(
            graph_data
        )

        orders = load_json(
            self.index_dir / "vector_order.json"
        )

        chunk_vectors = np.load(
            self.index_dir / "chunk_vectors.npy"
        )

        entity_vectors = np.load(
            self.index_dir / "entity_vectors.npy"
        )

        report_vectors = np.load(
            self.index_dir / "report_vectors.npy"
        )

        return {
            "chunks": chunks,
            "entities": entities,
            "relationships": relationships,
            "communities": communities,
            "reports": reports,
            "graph": graph,
            "orders": orders,
            "chunk_vectors": chunk_vectors,
            "entity_vectors": entity_vectors,
            "report_vectors": report_vectors,
        }


# -----------------------------------------------------------------------------
# DRIFT-inspired search
# -----------------------------------------------------------------------------


DRIFT_SYSTEM = """
You are an evidence-grounded GraphRAG reasoning system.
Use only the supplied community reports, graph entities, and source excerpts.
Cite exact source chunk IDs in square brackets. Never fabricate citations.
Clearly distinguish evidence from inference.
""".strip()


class DriftSearcher:
    def __init__(
        self,
        index: GraphRAGIndex,
        llm: LLMClient,
        embedder: LocalEmbedder,
    ) -> None:
        self.index = index
        self.llm = llm
        self.embedder = embedder
        data = index.load()
        self.chunks: dict[str, Chunk] = data["chunks"]
        self.entities: dict[str, Entity] = data["entities"]
        self.reports: dict[str, CommunityReport] = data["reports"]
        self.graph: nx.Graph = data["graph"]
        self.orders: dict[str, list[str]] = data["orders"]
        self.chunk_vectors: np.ndarray = data["chunk_vectors"]
        self.entity_vectors: np.ndarray = data["entity_vectors"]
        self.report_vectors: np.ndarray = data["report_vectors"]

    def retrieve_reports(self, query: str, k: int) -> list[CommunityReport]:
        ranked = cosine_top_k(self.embedder.encode_query(query), self.report_vectors, k)
        ids = self.orders["reports"]
        return [self.reports[ids[i]] for i, _ in ranked if ids[i] in self.reports]

    def retrieve_entities(self, query: str, k: int) -> list[Entity]:
        ranked = cosine_top_k(self.embedder.encode_query(query), self.entity_vectors, k)
        ids = self.orders["entities"]
        return [self.entities[ids[i]] for i, _ in ranked if ids[i] in self.entities]

    def retrieve_chunks(self, query: str, k: int) -> list[Chunk]:
        ranked = cosine_top_k(self.embedder.encode_query(query), self.chunk_vectors, k)
        ids = self.orders["chunks"]
        return [self.chunks[ids[i]] for i, _ in ranked if ids[i] in self.chunks]

    def expand_entity_neighbors(self, entity_ids: Iterable[str], hops: int) -> set[str]:
        visited = set(entity_ids)
        frontier = set(entity_ids)
        for _ in range(max(hops, 0)):
            next_frontier: set[str] = set()
            for entity_id in frontier:
                if entity_id in self.graph:
                    next_frontier.update(self.graph.neighbors(entity_id))
            next_frontier -= visited
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        return visited

    def chunks_for_entities(self, entity_ids: Iterable[str]) -> list[Chunk]:
        chunk_ids: list[str] = []
        for entity_id in entity_ids:
            entity = self.entities.get(entity_id)
            if entity:
                chunk_ids.extend(entity.chunk_ids)
        return [
            self.chunks[cid]
            for cid in dict.fromkeys(chunk_ids)
            if cid in self.chunks
        ]

    @staticmethod
    def _format_reports(reports: list[CommunityReport]) -> str:
        return "\n\n".join(
            f"COMMUNITY {report.community_id}\n{report.text}" for report in reports
        )

    @staticmethod
    def _format_entities(entities: list[Entity]) -> str:
        return "\n".join(
            f"- {entity.name} [{entity.type}]: {entity.description}"
            for entity in entities
        )

    @staticmethod
    def _format_chunks(chunks: list[Chunk], max_chars: int = 1400) -> str:
        blocks: list[str] = []
        for chunk in chunks:
            page = f", page {chunk.page}" if chunk.page is not None else ""
            blocks.append(
                f"[{chunk.id}] Source: {chunk.source}{page}\n{chunk.text[:max_chars]}"
            )
        return "\n\n".join(blocks)

    def _generate_step(
        self,
        original_question: str,
        current_question: str,
        reports: list[CommunityReport],
        entities: list[Entity],
        chunks: list[Chunk],
        followup_count: int,
    ) -> dict[str, Any]:
        prompt = f"""
Original user question:
{original_question}

Current exploration question:
{current_question}

Relevant community reports:
{self._format_reports(reports)}

Relevant entities:
{self._format_entities(entities)}

Source excerpts:
{self._format_chunks(chunks)}

Return strict JSON:
{{
  "partial_answer": "evidence-grounded answer with [CHUNK-ID] citations",
  "follow_up_questions": [
    "a question that would resolve an important uncertainty"
  ]
}}

Rules:
- Generate no more than {followup_count} follow-up questions.
- Do not repeat the current question.
- Use only supplied evidence.
- Cite exact chunk IDs.
- If evidence is insufficient, state it.
- Output JSON only.
""".strip()
        raw = extract_json_object(self.llm.generate(prompt, system_prompt=DRIFT_SYSTEM))
        followups_raw = raw.get("follow_up_questions", [])
        return {
            "partial_answer": str(raw.get("partial_answer", "")).strip(),
            "follow_up_questions": (
                [str(x).strip() for x in followups_raw if str(x).strip()][:followup_count]
                if isinstance(followups_raw, list)
                else []
            ),
        }

    def search(self, question: str) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        cfg = self.index.config
        seed_reports = self.retrieve_reports(question, cfg.top_seed_communities)
        seed_entities = self.retrieve_entities(question, cfg.top_seed_entities)
        seed_chunks = self.retrieve_chunks(question, cfg.top_chunks_per_step)
        expanded_ids = self.expand_entity_neighbors(
            [x.id for x in seed_entities],
            cfg.neighbor_hops,
        )
        graph_chunks = self.chunks_for_entities(expanded_ids)
        evidence_by_id = {x.id: x for x in seed_chunks + graph_chunks}
        evidence = list(evidence_by_id.values())[: cfg.top_chunks_per_step * 2]

        first = self._generate_step(
            original_question=question,
            current_question=question,
            reports=seed_reports,
            entities=seed_entities,
            chunks=evidence,
            followup_count=cfg.followups_per_step,
        )
        first_answer = str(first.get("partial_answer", "")).strip()
        partial_answers = [first_answer] if first_answer else []
        trace: list[dict[str, Any]] = [
            {
                "depth": 0,
                "question": question,
                "answer": first_answer,
                "chunk_ids": [x.id for x in evidence],
            }
        ]
        frontier = list(first.get("follow_up_questions", []))[: cfg.followups_per_step]
        seen_questions = {question.casefold()}

        for depth in range(1, cfg.drift_depth + 1):
            next_frontier: list[str] = []
            for followup in frontier:
                normalized = followup.casefold()
                if normalized in seen_questions:
                    continue
                seen_questions.add(normalized)

                reports = self.retrieve_reports(followup, max(2, cfg.top_seed_communities // 2))
                entities = self.retrieve_entities(followup, cfg.top_seed_entities)
                chunks = self.retrieve_chunks(followup, cfg.top_chunks_per_step)
                expanded = self.expand_entity_neighbors(
                    [x.id for x in entities],
                    cfg.neighbor_hops,
                )
                related = self.chunks_for_entities(expanded)
                combined_by_id = {x.id: x for x in chunks + related}
                combined = list(combined_by_id.values())[: cfg.top_chunks_per_step * 2]
                evidence_by_id.update(combined_by_id)

                step = self._generate_step(
                    original_question=question,
                    current_question=followup,
                    reports=reports,
                    entities=entities,
                    chunks=combined,
                    followup_count=cfg.followups_per_step,
                )
                step_answer = str(step.get("partial_answer", "")).strip()
                if step_answer:
                    partial_answers.append(step_answer)
                trace.append(
                    {
                        "depth": depth,
                        "question": followup,
                        "answer": step_answer,
                        "chunk_ids": [x.id for x in combined],
                    }
                )
                for candidate in step.get("follow_up_questions", []):
                    candidate = str(candidate).strip()
                    if candidate and candidate.casefold() not in seen_questions:
                        next_frontier.append(candidate)

            frontier = list(dict.fromkeys(next_frontier))[: cfg.followups_per_step]
            if not frontier:
                break

        final_evidence = list(evidence_by_id.values())
        partial_text = "\n".join(f"- {x}" for x in partial_answers if x)
        final_prompt = f"""
Answer the original question by synthesizing the partial analyses.

Original question:
{question}

Partial analyses:
{partial_text}

Available source excerpts:
{self._format_chunks(final_evidence[:30], max_chars=1000)}

Requirements:
- Answer directly and comprehensively.
- Use only supported information.
- Cite claims using exact chunk IDs such as [CHUNK-000123].
- Distinguish evidence from inference.
- State unresolved uncertainty.
- Never invent facts or citations.
""".strip()
        answer = self.llm.generate(
            final_prompt,
            system_prompt=DRIFT_SYSTEM,
            max_new_tokens=max(cfg.max_new_tokens, 1200),
        )

        cited_ids = sorted(set(re.findall(r"\[(CHUNK-\d{6})\]", answer)))
        sources = [
            {
                "chunk_id": cid,
                "source": self.chunks[cid].source,
                "page": self.chunks[cid].page,
            }
            for cid in cited_ids
            if cid in self.chunks
        ]
        return {"answer": answer, "sources": sources, "trace": trace}