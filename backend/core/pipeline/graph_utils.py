import os
import re
import json
import spacy
import logging
import networkx as nx
from dotenv import load_dotenv
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Set, Tuple, Any

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

WORKER_LIMIT = max(1, int(os.getenv("WORKER_LIMIT", "5")))
NODE_BATCH_SIZE = max(1, int(os.getenv("NODE_BATCH_SIZE", "25")))
REQUEST_MAX_ATTEMPTS = max(1, int(os.getenv("REQUEST_MAX_ATTEMPTS", "3")))
CLEAN_NODE_MAX_TOKENS = max(1, int(os.getenv("CLEAN_NODE_MAX_TOKENS", "2048")))

from .models import TextChunk
from core.utils import call_llm

DEFAULT_ENTITY_LABELS = {
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
}


class GraphBuilder:
    def __init__(
        self,
        model_name: str = "en_core_web_lg",
        entity_labels: Optional[Iterable[str]] = None,
        batch_size: int = 128,
        min_node_chunks: int = 1,
        min_edge_weight: int = 1,
        entity_clean_batch_size: int = NODE_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.entity_labels = set(entity_labels or DEFAULT_ENTITY_LABELS)
        self.batch_size = batch_size
        self.min_node_chunks = min_node_chunks
        self.min_edge_weight = min_edge_weight
        self.entity_clean_batch_size = max(1, int(entity_clean_batch_size))

        self.graph: nx.Graph = nx.Graph()
        self.entity_to_chunks: Dict[str, Set[str]] = {}
        self._nlp = None

    def _get_nlp(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError as exc:
                raise RuntimeError(
                    f"spaCy model missing. Run: "
                    f"python -m spacy download {self.model_name}"
                ) from exc

            if not any(
                component in self._nlp.pipe_names
                for component in (
                    "parser",
                    "senter",
                    "sentencizer",
                )
            ):
                self._nlp.add_pipe("sentencizer")

        return self._nlp

    @staticmethod
    def _normalize_entity(text: str) -> str:
        return re.sub(r"\s+", " ", str(text)).strip().casefold()

    @staticmethod
    def _is_valid_entity(name: str) -> bool:
        if len(name) <= 2:
            return False

        if re.fullmatch(r"^\d+[\d\s,.]*$", name):
            return False

        return True

    @staticmethod
    def _node_id(name: str, label: str) -> str:
        return f"{label}::{name}"

    @staticmethod
    def _parse_llm_json(raw_response: str) -> Dict[str, Any]:
        response = str(raw_response).strip()
        response = re.sub(
            r"^```(?:json)?\s*",
            "",
            response,
            flags=re.IGNORECASE,
        )
        response = re.sub(r"\s*```$", "", response)

        start = response.find("{")
        end = response.rfind("}")

        if start == -1 or end <= start:
            raise ValueError("The LLM response does not contain a JSON object.")

        parsed = json.loads(response[start : end + 1])

        if not isinstance(parsed, dict):
            raise ValueError("The LLM response must be a JSON object.")

        return parsed

    def extract_entities(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        doc = self._get_nlp()(text)

        entities: List[str] = []
        seen: Set[str] = set()

        for ent in doc.ents:
            if ent.label_ not in self.entity_labels:
                continue

            name = self._normalize_entity(ent.text)

            if not self._is_valid_entity(name):
                continue

            node_id = self._node_id(name, ent.label_)

            if node_id in seen:
                continue

            seen.add(node_id)
            entities.append(node_id)

        return entities

    def _clean_entity_batch(
        self,
        entities: List[Dict[str, str]],
    ) -> Tuple[Dict[str, Tuple[str, str]], Set[str]]:
        """
        Clean, canonicalize, and correct entity types for one batch.

        Returns:
            mappings:
                old_node_id -> (canonical_name, corrected_type)

            removed_nodes:
                set of old node IDs that should be removed
        """
        if not entities:
            return {}, set()

        input_entities = [
            {
                "name": str(entity.get("name", "")).strip(),
                "current_type": str(entity.get("current_type", "")).strip(),
            }
            for entity in entities
            if str(entity.get("name", "")).strip()
            and str(entity.get("node_id", "")).strip()
        ]

        if not input_entities:
            return {}, set()

        prompt = f"""
            /no_think

            Return exactly one valid JSON object.

            Do not include reasoning, explanations, markdown, or code fences.
            The first character must be {{
            The last character must be }}

            You clean, canonicalize, and correct the entity types of named entities
            extracted by spaCy.

            Allowed entity types:

            {json.dumps(sorted(DEFAULT_ENTITY_LABELS), ensure_ascii=False)}

            Required output schema:

            {{
            "entities": [
                {{
                "name": "canonical input entity name",
                "type": "correct allowed entity type",
                "aliases": [
                    "exact input entity name"
                ]
                }}
            ],
            "removed_entities": [
                "exact invalid input entity name"
            ]
            }}

            Rules:

            1. Use the entity name and current_type to determine what each entity represents.
            2. Correct current_type when it is wrong.
            3. The returned type must be one of the allowed entity types.
            4. Remove malformed phrases, sentence fragments, headings, verbs, and entities
            that do not match any allowed type.
            5. Merge only clear aliases that refer to the same real entity.
            6. Do not merge entities only because their names are similar.
            7. Titles, abbreviations, shortened names, possessive forms, and formatting
            variants may be merged when they clearly refer to the same entity.
            8. The canonical name must exactly match one input entity name.
            9. Every alias must exactly match one input entity name.
            10. Include the canonical name itself inside aliases.
            11. Do not invent, expand, translate, correct, or complete names.
            12. Every input entity must appear exactly once, either in one aliases list
                or in removed_entities.
            13. Return JSON only.

            INPUT:

            {json.dumps(input_entities, ensure_ascii=False)}
        """.strip()

        response_text = call_llm(
            prompt=prompt,
            max_tokens=CLEAN_NODE_MAX_TOKENS,
            json_response=True,
        )
        result = self._parse_llm_json(response_text)

        name_to_nodes: Dict[str, List[str]] = {}

        for entity in entities:
            node_id = str(entity.get("node_id", "")).strip()
            name = self._normalize_entity(entity.get("name", ""))

            if not node_id or not name:
                continue

            name_to_nodes.setdefault(name, []).append(node_id)

        mappings: Dict[str, Tuple[str, str]] = {}
        removed_nodes: Set[str] = set()

        cleaned_entities = result.get("entities", [])

        if not isinstance(cleaned_entities, list):
            raise ValueError("'entities' must be a list.")

        for item in cleaned_entities:
            if not isinstance(item, dict):
                continue

            canonical_name = self._normalize_entity(item.get("name", ""))
            corrected_type = str(item.get("type", "")).strip().upper()
            aliases = item.get("aliases", [])

            if canonical_name not in name_to_nodes:
                continue

            if corrected_type not in DEFAULT_ENTITY_LABELS:
                continue

            if not isinstance(aliases, list):
                continue

            for alias in aliases:
                normalized_alias = self._normalize_entity(alias)

                if normalized_alias not in name_to_nodes:
                    continue

                for old_node_id in name_to_nodes[normalized_alias]:
                    mappings[old_node_id] = (
                        canonical_name,
                        corrected_type,
                    )

        removed_entities = result.get("removed_entities", [])

        if not isinstance(removed_entities, list):
            raise ValueError("'removed_entities' must be a list.")

        for removed_name in removed_entities:
            normalized_name = self._normalize_entity(removed_name)

            for old_node_id in name_to_nodes.get(normalized_name, []):
                removed_nodes.add(old_node_id)

        # Preserve any entity the LLM forgot to return.
        for entity in entities:
            old_node_id = str(entity.get("node_id", "")).strip()
            name = self._normalize_entity(entity.get("name", ""))
            current_type = str(entity.get("current_type", "")).strip()

            if not old_node_id:
                continue

            if old_node_id in mappings or old_node_id in removed_nodes:
                continue

            mappings[old_node_id] = (
                name,
                current_type,
            )

        return mappings, removed_nodes

    def _clean_graph_entities(
        self,
    ) -> None:
        """
        Clean graph entities concurrently and rebuild the graph.
        """
        entity_items: List[Dict[str, str]] = []

        for node_id, data in self.graph.nodes(data=True):
            name = str(data.get("name", "")).strip()
            label = str(data.get("label", "")).strip()

            if not name or not label:
                continue

            entity_items.append(
                {
                    "node_id": str(node_id),
                    "name": name,
                    "current_type": label,
                }
            )

        entity_items.sort(key=lambda item: item["node_id"])

        mappings: Dict[str, Tuple[str, str]] = {}
        removed_nodes: Set[str] = set()

        batches = [
            entity_items[start : start + self.entity_clean_batch_size]
            for start in range(
                0,
                len(entity_items),
                self.entity_clean_batch_size,
            )
        ]

        def process_batch(
            batch_index: int,
            batch: List[Dict[str, str]],
        ) -> Tuple[Dict[str, Tuple[str, str]], Set[str]]:
            """Clean one entity batch with limited retries."""
            last_error: Exception | None = None

            for attempt in range(1, REQUEST_MAX_ATTEMPTS + 1):
                try:
                    return self._clean_entity_batch(
                        entities=batch,
                    )

                except Exception as exc:
                    last_error = exc

                    logging.warning(
                        "Entity cleaning failed for batch %d/%d | attempt=%d/%d",
                        batch_index,
                        len(batches),
                        attempt,
                        REQUEST_MAX_ATTEMPTS,
                    )

            raise RuntimeError(
                f"Entity cleaning failed for batch {batch_index}/{len(batches)} "
                f"after {REQUEST_MAX_ATTEMPTS} attempts."
            ) from last_error

        if batches:
            max_workers = min(
                WORKER_LIMIT,
                len(batches),
            )

            logging.info(
                "Cleaning %d entities in %d batches with %d workers",
                len(entity_items),
                len(batches),
                max_workers,
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {
                    executor.submit(
                        process_batch,
                        batch_index,
                        batch,
                    ): (batch_index, batch)
                    for batch_index, batch in enumerate(
                        batches,
                        start=1,
                    )
                }

                for future in as_completed(future_to_batch):
                    batch_index, batch = future_to_batch[future]

                    try:
                        batch_mappings, batch_removed = future.result()

                    except Exception:
                        logging.exception(
                            "Entity cleaning permanently failed for batch %d/%d",
                            batch_index,
                            len(batches),
                        )
                        raise

                    mappings.update(batch_mappings)
                    removed_nodes.update(batch_removed)

                    logging.info(
                        "Entity cleaning completed for batch %d/%d | entities=%d",
                        batch_index,
                        len(batches),
                        len(batch),
                    )

        new_graph: nx.Graph = nx.Graph()
        new_entity_to_chunks: Dict[str, Set[str]] = {}
        node_mapping: Dict[str, Optional[str]] = {}

        for old_node_id, data in self.graph.nodes(data=True):
            old_node_key = str(old_node_id)
            name = str(data.get("name", "")).strip()
            label = str(data.get("label", "")).strip()

            if old_node_key in removed_nodes:
                node_mapping[old_node_key] = None
                continue

            canonical_name, corrected_label = mappings.get(
                old_node_key,
                (name, label),
            )

            new_node_id = self._node_id(
                canonical_name,
                corrected_label,
            )

            node_mapping[old_node_key] = new_node_id

            if not new_graph.has_node(new_node_id):
                new_graph.add_node(
                    new_node_id,
                    name=canonical_name,
                    label=corrected_label,
                    mention_count=0,
                    chunk_count=0,
                )

            new_graph.nodes[new_node_id]["mention_count"] += int(
                data.get("mention_count", 0)
            )

            old_chunk_keys = self.entity_to_chunks.get(
                old_node_id,
                self.entity_to_chunks.get(
                    old_node_key,
                    set(),
                ),
            )

            new_entity_to_chunks.setdefault(
                new_node_id,
                set(),
            ).update(old_chunk_keys)

        for node_id, chunk_keys in new_entity_to_chunks.items():
            new_graph.nodes[node_id]["chunk_count"] = len(chunk_keys)

        for source, target, data in self.graph.edges(data=True):
            new_source = node_mapping.get(str(source))
            new_target = node_mapping.get(str(target))

            if new_source is None or new_target is None or new_source == new_target:
                continue

            chunks = set(data.get("chunks", set()))
            weight = int(data.get("weight", 1))

            if new_graph.has_edge(new_source, new_target):
                edge = new_graph[new_source][new_target]
                edge["weight"] += weight
                edge["chunks"].update(chunks)
                edge["chunk_count"] = len(edge["chunks"])

            else:
                new_graph.add_edge(
                    new_source,
                    new_target,
                    weight=weight,
                    chunks=chunks,
                    chunk_count=len(chunks),
                )

        self.graph = new_graph
        self.entity_to_chunks = new_entity_to_chunks

    def build_graph(
        self,
        chunks: List[TextChunk],
    ) -> nx.Graph:
        self.graph.clear()
        self.entity_to_chunks.clear()

        valid_chunks = [chunk for chunk in chunks if chunk.text and chunk.text.strip()]

        if not valid_chunks:
            return self.graph

        logging.info(
            "Building graph from %d chunks using %s",
            len(valid_chunks),
            self.model_name,
        )

        nlp = self._get_nlp()

        docs = nlp.pipe(
            (chunk.text for chunk in valid_chunks),
            batch_size=self.batch_size,
        )

        for chunk, doc in zip(valid_chunks, docs):
            chunk_key = f"{chunk.source}_{chunk.chunk_id}"
            chunk_nodes: Set[str] = set()

            for ent in doc.ents:
                if ent.label_ not in self.entity_labels:
                    continue

                name = self._normalize_entity(ent.text)

                if not self._is_valid_entity(name):
                    continue

                node_id = self._node_id(name, ent.label_)

                if not self.graph.has_node(node_id):
                    self.graph.add_node(
                        node_id,
                        name=name,
                        label=ent.label_,
                        mention_count=0,
                        chunk_count=0,
                    )

                self.graph.nodes[node_id]["mention_count"] += 1
                chunk_nodes.add(node_id)

                self.entity_to_chunks.setdefault(
                    node_id,
                    set(),
                ).add(chunk_key)

            for node_id in chunk_nodes:
                self.graph.nodes[node_id]["chunk_count"] = len(
                    self.entity_to_chunks[node_id]
                )

            for sentence in doc.sents:
                sentence_nodes: List[str] = []
                seen_sentence_nodes: Set[str] = set()

                for ent in sentence.ents:
                    if ent.label_ not in self.entity_labels:
                        continue

                    name = self._normalize_entity(ent.text)

                    if not self._is_valid_entity(name):
                        continue

                    node_id = self._node_id(name, ent.label_)

                    if node_id in seen_sentence_nodes:
                        continue

                    seen_sentence_nodes.add(node_id)
                    sentence_nodes.append(node_id)

                for source, target in combinations(sentence_nodes, 2):
                    if self.graph.has_edge(source, target):
                        edge = self.graph[source][target]
                        edge["weight"] += 1
                        edge["chunks"].add(chunk_key)
                        edge["chunk_count"] = len(edge["chunks"])
                    else:
                        self.graph.add_edge(
                            source,
                            target,
                            weight=1,
                            chunk_count=1,
                            chunks={chunk_key},
                        )

        weak_nodes = [
            node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("chunk_count", 0) < self.min_node_chunks
        ]

        self.graph.remove_nodes_from(weak_nodes)

        for node_id in weak_nodes:
            self.entity_to_chunks.pop(node_id, None)

        weak_edges = [
            (source, target)
            for source, target, data in self.graph.edges(data=True)
            if data.get("weight", 0) < self.min_edge_weight
        ]

        self.graph.remove_edges_from(weak_edges)

        if self.graph.number_of_nodes() > 0:
            logging.info(
                "Sending %d graph nodes to the LLM for cleaning",
                self.graph.number_of_nodes(),
            )
            self._clean_graph_entities()

        logging.info(
            "Graph completed: %d nodes and %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

        return self.graph
