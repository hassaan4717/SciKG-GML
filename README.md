# SciKG-GML

SciKG-GML is a full-stack GraphRAG system for asking questions over user-provided text. It combines named-entity graphs, graph communities, dense embeddings, BM25 keyword retrieval, reranking, and an OpenAI-compatible language model in one authenticated web application.

The project is designed around a practical research question:

> **Can a knowledge graph improve retrieval and answer generation when it is used together with semantic and lexical search rather than as a replacement for either?**

### Short answer

The implementation provides a credible answer at the system level: graph structure supplies entity neighborhoods and community-level context, while dense and BM25 retrieval preserve semantic and exact-term matching. These signals are fused and reranked before generation, so the graph can add related evidence without discarding direct text matches. The committed `novel12` evaluation report shows strong retrieval-context relevance and high faithfulness for the tested run. It does **not** establish improvement over a vector-only or BM25-only baseline, because those baselines are not included in the repository.

## What the system does

Users create an authenticated chat session, paste text or upload a `.txt` file, and ask questions about the indexed material. Each session has its own persisted GraphRAG state.

![SciKG-GML chat interface](assets/chat-interface.png)

The ingestion and question-answering flow is:

1. **Parse and chunk** input text using the configured chunk size and overlap.
2. **Create dense embeddings** for chunks and store them in a persistent Chroma collection.
3. **Build a lexical index** with BM25 for exact and term-based matching.
4. **Extract entities** with spaCy and create a weighted co-occurrence graph with NetworkX.
5. **Clean and canonicalize entities** through the configured LLM, then detect graph communities with greedy modularity.
6. **Generate community summaries** and embed those summaries for global thematic retrieval.
7. **Rewrite follow-up questions** into standalone retrieval queries when conversation history provides useful references.
8. **Retrieve local evidence** with dense search, BM25, weighted reciprocal-rank fusion, and graph-neighborhood expansion.
9. **Retrieve global evidence** by comparing the question with community-summary embeddings.
10. **Combine and rerank** local and global candidates with a cross-encoder.
11. **Generate an answer** from the selected context. When no relevant context is available, the application returns an explicit insufficient-context response instead of answering from unsupported information.

![SciKG-GML knowledge graph visualization](assets/graph-visualization.png)

## Main capabilities

- Token-authenticated sign-up and login.
- Multiple private chat sessions per user.
- Raw-text and `.txt` ingestion from the browser.
- Persisted per-session chunks, embeddings, graph state, and community summaries.
- Hybrid local retrieval using dense embeddings and BM25.
- Entity-based neighborhood expansion for related evidence.
- Global retrieval over graph-community summaries.
- Cross-encoder reranking of retrieved chunks.
- Interactive force-directed graph visualization with a configurable node limit.
- Source filenames and retrieval metadata attached to assistant responses.
- Evaluation scripts for retrieval context quality and generated-answer quality.

## Evaluation results in this repository

The checked-in report at `backend/evaluation/results/novel12/report_novel12.json` contains results for the `novel12` corpus. Scores are reported on a 0 to 1 scale. Retrieval metrics are evaluated by question type; generation metrics are selected according to question type.

| Question type | Context relevance | Context recall | Answer correctness | Faithfulness |
| --- | ---: | ---: | ---: | ---: |
| Fact Retrieval | 0.865 | 0.738 | 0.767 | Not evaluated |
| Complex Reasoning | 0.914 | 0.882 | 0.720 | 0.916 |
| Contextual Summarize | 0.929 | 0.733 | 0.723 | 0.899 |
| Creative Generation | 0.823 | 0.752 | 0.833 | 0.808 |

Additional generation metrics in the same report are:

| Question type | ROUGE | Coverage |
| --- | ---: | ---: |
| Fact Retrieval | 0.755 | Not evaluated |
| Complex Reasoning | Not evaluated | 0.784 |
| Contextual Summarize | 0.735 | 0.766 |
| Creative Generation | Not evaluated | 0.784 |

These are evaluation-run results, not a controlled ablation. The evaluation uses LLM-assisted metrics for several measures, so comparisons should keep the dataset, model configuration, prompts, and evaluator constant.

## Repository layout

```text
SciKG-GML/
├── backend/
│   ├── authentication/          Token sign-up and login endpoints
│   ├── core/
│   │   ├── pipeline/            Web application's GraphRAG implementation
│   │   ├── pipeline_v2/         Standalone framework-free GraphRAG prototype
│   │   ├── services.py          Chat, ingestion, and graph service layer
│   │   └── views.py             Session, message, and graph API views
│   ├── evaluation/              Dataset preparation, metrics, and reports
│   ├── grag/                    Django settings and URL configuration
│   ├── manage.py
│   └── requirements.txt
├── frontend/
│   ├── src/components/          Authentication and chat workspace
│   ├── src/App.jsx              React routing and token state
│   └── package.json
├── assets/                      README screenshots
└── README.md
```

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer
- Git
- An OpenAI-compatible chat and embedding endpoint
- Enough local memory and disk space for PyTorch, Chroma, spaCy, and the reranker model

The web pipeline's `GraphBuilder` loads `en_core_web_lg` by default. The requirements file also lists `en_core_web_sm`, but installing only the small model will not satisfy the active graph builder unless the code is changed to use it.

## Installation

From the directory where you want the project:

```powershell
git clone https://github.com/hassaan4717/SciKG-GML.git
cd SciKG-GML
```

### Backend

Open a PowerShell terminal in the repository root:

```powershell
cd backend
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_lg
Copy-Item .env.example .env
python manage.py migrate
python manage.py runserver
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate again. Keep this server running at `http://127.0.0.1:8000/`.

### Frontend

Open a second terminal in the repository root:

```powershell
cd frontend
npm install
```

Create `frontend/.env` with the API base path used by Django:

```env
VITE_API_URL=http://127.0.0.1:8000/api/v1
```

Start Vite:

```powershell
npm run dev
```

Open the URL printed by Vite, create an account, and start a session.

## Configuration

Copy `backend/.env.example` to `backend/.env` and set at least:

```env
SECRET_KEY=replace-with-a-random-django-secret
BASE_OPENAI_URL=https://your-provider.example/v1
API_KEY=your-provider-api-key
LLM_MODEL=your-chat-model
EMBEDDING_MODEL=your-embedding-model
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
BASE_RAG_DIR=./user_rag_data
```

Important retrieval settings include:

| Variable | Purpose | Default |
| --- | --- | ---: |
| `RAG_CHUNK_SIZE` | Chunk size used during ingestion | `600` |
| `RAG_OVERLAP` | Overlap between adjacent chunks | `80` |
| `RAG_TOP_K` | Dense and BM25 candidates per retriever | `20` |
| `RAG_TOP_N` | Chunks retained after reranking | `5` |
| `RAG_TOP_SUMMARY` | Community summaries considered globally | `3` |
| `CHAT_HISTORY_LIMIT` | Recent messages retained for context | `10` |
| `WORKER_LIMIT` | Concurrent processing/evaluation workers | `5` |

Run the backend from `backend/` so the relative `BASE_RAG_DIR` and evaluation paths resolve as expected. Never commit `backend/.env` or provider credentials.

## Using the application

1. Register or sign in at the frontend.
2. Create a new conversation.
3. In the Knowledge panel, paste text or select a `.txt` file and submit it.
4. Wait for ingestion to finish. The response includes indexing statistics such as chunks, entities, edges, and communities.
5. Ask a question in Conversation. Follow-up questions can use the preceding chat context.
6. Open Visual Graph to inspect the highest-degree entities and their connections.
7. Review indexed sources and community summaries in the Knowledge panel.

The browser talks to these Django API groups:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/user/signup/` | Create a user and return a token |
| `POST` | `/api/v1/user/login/` | Authenticate and return a token |
| `GET`, `POST` | `/api/v1/chats/` | List or create sessions |
| `GET`, `DELETE` | `/api/v1/chats/<id>/` | Read or delete a session |
| `GET`, `POST` | `/api/v1/chats/<id>/messages/` | Read history, chat, or ingest text |
| `GET` | `/api/v1/chats/<id>/graph/` | Read graph connections and communities |

All chat endpoints require the token returned by authentication in an `Authorization: Token <token>` header.

## Running the evaluation suite

The evaluation datasets are JSON files in `backend/evaluation/dataset/`. Each file contains a corpus, questions, question types, reference answers, and evidence. The evaluation pipeline:

1. Builds or loads a GraphRAG index and writes generated predictions.
2. Measures context relevance and context recall.
3. Measures type-appropriate generation metrics: ROUGE, answer correctness, coverage, and faithfulness.

From `backend/`, configure the dataset in `.env` and run:

```powershell
$env:DATASET_FILES = "novel12"
$env:SAMPLE = "5"
.\evaluation\run_evaluation.ps1
```

Outputs are written under `backend/evaluation/results/<dataset>/` as prediction and report JSON files. `SAMPLE` samples questions per question type; set it to `-1` to process all available questions.

## Standalone `pipeline_v2`

`backend/core/pipeline_v2/` is a separate framework-free implementation with PDF support, local sentence-transformer embeddings, hierarchical communities, and DRIFT-inspired search. It is not the implementation used by the Django web endpoints.

Its CLI expects a project directory containing `config.json` and an `input/` directory:

```powershell
cd backend
python -m core.pipeline_v2.cli init --root .\local_project
# Add .txt, .md, or .pdf files to .\local_project\input
python -m core.pipeline_v2.cli index --root .\local_project
python -m core.pipeline_v2.cli query --root .\local_project --question "What are the main entities?"
```

## Limitations and research next steps

- The current repository reports one checked-in corpus result and does not include vector-only, BM25-only, or graph-disabled baselines.
- Several evaluation metrics call an LLM, so evaluator variance and provider configuration affect scores.
- The web uploader accepts `.txt`; PDF ingestion belongs to `pipeline_v2`, not the web application.
- Django is configured for development (`DEBUG=True`, permissive CORS, SQLite). Harden settings before deployment.
- A stronger experimental answer to the research question would add controlled ablations, repeated runs, latency and indexing-cost measurements, and statistical comparison across multiple corpora.

## License

See [LICENSE](LICENSE).
