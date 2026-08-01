from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from core import (
    Config,
    DriftSearcher,
    GraphRAGIndex,
    LocalEmbedder,
    OpenAICompatibleLLM,
)


def init_project(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "input").mkdir(exist_ok=True)
    (root / "index").mkdir(exist_ok=True)

    destination = root / "config.json"

    print(f"Initialized project at: {root}")
    print(f"Put .txt, .md, or .pdf files in: {root / 'input'}")
    print(f"Review configuration in: {destination}")


def load_runtime(root: Path):
    config = Config.from_path(root / "config.json")

    llm = OpenAICompatibleLLM(
        model_name=config.llm_model,
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        timeout=config.llm_timeout_seconds,
        max_retries=config.llm_max_retries,
    )

    embedder = LocalEmbedder(
        config.embedding_model
    )

    index = GraphRAGIndex(
        root=root,
        config=config,
        llm=llm,
        embedder=embedder,
    )

    return config, llm, embedder, index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Framework-free local GraphRAG with DRIFT-inspired search."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init")
    init_parser.add_argument("--root", type=Path, required=True)

    index_parser = sub.add_parser("index")
    index_parser.add_argument("--root", type=Path, required=True)

    query_parser = sub.add_parser("query")
    query_parser.add_argument("--root", type=Path, required=True)
    query_parser.add_argument("--question", required=True)
    query_parser.add_argument("--trace-json", type=Path)

    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    if args.command == "init":
        init_project(root)
        return

    config, llm, embedder, index = load_runtime(root)

    if args.command == "index":
        index.build()
        print(f"Index completed: {root / 'index'}")
        return

    searcher = DriftSearcher(index, llm, embedder)
    result = searcher.search(args.question.strip())
    print(result["answer"])

    if result["sources"]:
        print("\nSources")
        for source in result["sources"]:
            page = f", page {source['page']}" if source["page"] else ""
            print(f"- [{source['chunk_id']}] {source['source']}{page}")

    if args.trace_json:
        destination = args.trace_json.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nTrace written to: {destination}")


if __name__ == "__main__":
    main()
