# gh-social-ml

Machine-learning and data-pipeline services for [gh-social](https://github.com/SUBHRO71/gh-social). This repository discovers and enriches GitHub repositories, builds repository and user embeddings, retrieves and ranks recommendations, assembles feed slices, and processes recommendation feedback.

## What Is Here

- `acquisition/` and `ingestion/`: GitHub discovery, enrichment, classification, and quality filtering.
- `embedding/`: SentenceTransformer repository embeddings and Qdrant indexing.
- `retrieval/` and `retrieval_engine.py`: candidate retrieval, ranking, and recommendation batches.
- `inference/`: learned ranking assets and final feed assembly.
- `feedback/`: feedback queue producer, consumer, and embedding updates.
- `trending/`: scheduled GitHub Trending ingestion.
- `api/main.py`: the integrated internal FastAPI service.
- `app.py`: a smaller standalone feed-assembly FastAPI service.
- `Ranking_Training_module/`: training and synthetic-data utilities for the heavy ranker.
- `tests/`: unit, integration, and benchmark coverage.

## Requirements

- Python 3.10 or newer
- Qdrant for repository and user vectors
- Redis for durable v2 feedback streaming
- PostgreSQL only for transitional legacy paths and offline tooling
- A GitHub personal access token for acquisition and trending ingestion

The first SentenceTransformer run downloads the configured model and may take longer than subsequent starts.

## Quick Start

```bash
git clone https://github.com/SUBHRO71/gh-social-ml.git
cd gh-social-ml

# Install uv first: https://docs.astral.sh/uv/getting-started/installation/
uv sync --locked
cp .env.example .env
uv run python main.py --validate-config
```

On PowerShell, use `Copy-Item .env.example .env` instead of `cp` if `cp` is not available.

Configure at least the services needed for the command you are running:

```dotenv
GITHUB_TOKEN=github_pat_...
DATABASE_URL=postgresql://user:password@localhost:5432/gh_social
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=osiris_research_corpus
REDIS_URL=redis://localhost:6379/0
INTERNAL_API_SECRET=replace-with-a-long-random-secret
```

Do not commit `.env` or production credentials.

## Run The APIs

Start the integrated API:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Its OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

| Method | Path | Purpose | Authentication |
| --- | --- | --- | --- |
| `GET` | `/api/v2/health` | ML, Qdrant, and feedback-stream readiness | `X-Internal-Secret` |
| `POST` | `/api/v2/recommendations/generate` | Generate canonical `repo_id` recommendations | `X-Internal-Secret` |
| `POST` | `/api/v2/feedback/batch` | Durably accept ordered interaction feedback | `X-Internal-Secret` |
| `POST` | `/api/v2/users/onboard` | Create or update a versioned user profile vector | `X-Internal-Secret` |
| `POST` | `/api/v2/repositories/embed` | Embed versioned canonical repository content | `X-Internal-Secret` |
| `POST` | `/api/v2/repositories/refresh` | Refresh versioned repository features | `X-Internal-Secret` |

The protected endpoints fail closed when `INTERNAL_API_SECRET` is missing. Backend requests must send the same value in the `X-Internal-Secret` header. The `/api/v1` routes remain transitional and are not the contract for new backend integrations.

The standalone assembly service accepts exactly 15 pre-ranked candidates and returns their final repository-ID order:

```bash
uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

```http
POST /api/internal/ml/assemble-feed
Content-Type: application/json

{
  "candidates": [
    { "repo_id": "repository-id", "final_score": 0.91 }
  ]
}
```

The request must contain 15 candidate objects. The response shape is `{ "rankedRepoIds": ["..."] }`.

## Run The Pipelines

Discover, enrich, filter, and index repositories:

```bash
python main.py --limit 150 --batch-size 15 --workers 4
```

Skip Qdrant indexing when validating acquisition only:

```bash
python main.py --limit 30 --no-index-qdrant
```

Refresh GitHub Trending once or run the scheduler:

```bash
python trending_service.py --once
python trending_service.py --scheduled
```

Generate and inspect recommendation batches for onboarded users:

```bash
python retrieval_engine.py
```

Additional evaluation, seeding, onboarding, and integration utilities are in `scripts/`. Run a script with `--help` before using it against shared infrastructure.

Repository acquisition and trending synchronization are offline worker jobs, not request-path services. See the [offline pipeline operations runbook](docs/OFFLINE_PIPELINE_RUNBOOK.md) for configuration validation, bounded runs, safe shutdown, checkpoint resume, and diagnosis.

## Schema Cutover Contract

The v2 boundary keeps backend-owned product state in `app` and immutable delivery telemetry in `telemetry`. Some v1 code still reads or writes legacy tables such as `user_recommendation_batches` and accepts direct feedback; it is retained only during the coordinated rollout.

New work must follow these boundaries:

- Identify every recommendation and feedback signal by canonical `repo_id`. Treat GitHub `owner/name` only as transitional ingestion input.
- Let the gh-social backend own PostgreSQL product mutations and telemetry transactions. The ML service must not receive direct production database credentials after cutover.
- Store typed recommendation entries in Redis with repository ID, score, source, model version, and summary ID.
- Record only the feed slice actually served, with `serve_id`, `session_id`, ordered positions, source, and model version.
- Preserve the original interaction event type. Do not reduce events to a calculated feedback score before they are stored.
- Consume backend-created ML outbox records with idempotency keys and retry semantics. Direct fire-and-forget feedback delivery is transitional and must be removed.
- Ignore quick passive swipes. The coordinated client/backend contract records an impression after one second of visibility and dwell after three seconds.
- Do not recreate a standalone `user_feedback` table. Current state belongs in reactions and saves; immutable events and derived per-user/repository rollups belong in telemetry.

The cutover must be deployed together with the gh-social database, backend, and Expo changes. Keep legacy tables read-only during the rollback window and remove them only after feed, authentication, social actions, boards, and ML delivery have been verified.

## Tests

Run the default suite:

```bash
pytest
```

Useful focused commands:

```bash
pytest -m unit
pytest tests/test_feedback.py
pytest --cov=. --cov-report=term-missing
```

Integration tests require their external services and may use `TEST_DATABASE_URL`. Marker definitions and discovery settings live in `pytest.ini`.

## Contributing

Create a focused branch, add or update tests for behavior changes, and run the relevant test groups before opening a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

This project is licensed under the [MIT License](LICENSE).
