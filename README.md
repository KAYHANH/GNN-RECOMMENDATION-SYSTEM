# Fashion Recommendation System

This project translates the notebook in `mp-model-training.pdf` into a modular recommender stack that follows the four-pillar execution plan:

1. Better ML models
2. Evaluation and metrics rigor
3. Full-stack app / UI
4. MLOps pipeline

The PDF currently combines:

- A semantic search engine built with `SentenceTransformer` and `FAISS`
- A heterogenous `SAGEConv` graph model for customer-to-article recommendations
- Recommendation explanation logic based on prior purchases

This repo upgrades that approach by:

- Replacing `SAGEConv` with `LightGCN`
- Switching link training to `BPR` loss
- Adding a temporal offline evaluation harness
- Preparing a second-stage `LightGBM` reranker
- Exposing the system through `FastAPI`
- Scaffolding a `Next.js` demo UI
- Adding `MLflow`, `DVC`, and CI hooks

## Project Layout

```text
fashion-rec-system/
├── api/
├── artifacts/
├── data/
├── frontend/
├── mlops/
├── models/
├── paper/
├── tests/
└── training/
```

## Recommended Execution Order

1. Run the offline evaluation harness with a temporal split.
2. Train LightGCN with BPR on transaction data.
3. Fine-tune the semantic encoder on fashion triplets.
4. Train the LightGBM reranker over top-k candidates.
5. Serve the recommend/search/explain flows through FastAPI.
6. Plug in MLflow, DVC, and CI for repeatability.

## Python Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset Modes

The project now supports two data modes:

- `subset/dev mode`: the recent 14-day dense slice used for quick debugging
- `full/final mode`: the full cleaned H&M dataset for final training and paper runs

## Build The Original 14-Day H&M Subset

```powershell
python -m training.prepare_hm_subset `
  --articles data\raw\articles.csv `
  --customers data\raw\customers.csv `
  --transactions data\raw\transactions_train.csv `
  --images-dir data\raw\images `
  --output-dir data `
  --mode subset `
  --days 14
```

This will:

- filter transactions to the most recent 14 days
- keep only customers and articles that appear in that window
- generate `articles_cleaned.csv`, `customers_cleaned.csv`, and `transactions_cleaned.csv`

## Build The Full Cleaned Dataset

Use this for the final run without throwing away historical interactions:

```powershell
python -m training.prepare_hm_subset `
  --articles data\raw\articles.csv `
  --customers data\raw\customers.csv `
  --transactions data\raw\transactions_train.csv `
  --images-dir data\raw\images `
  --output-dir data `
  --mode full `
  --output-prefix full_
```

This writes:

- `data\full_articles_cleaned.csv`
- `data\full_customers_cleaned.csv`
- `data\full_transactions_cleaned.csv`

## Build The Heterogeneous Graph

To preserve the original notebook architecture for graph explainability experiments:

```powershell
python training\build_hetero_graph.py `
  --articles data\articles_cleaned.csv `
  --customers data\customers_cleaned.csv `
  --transactions data\transactions_cleaned.csv `
  --output-dir artifacts
```

## Run Offline Evaluation

```powershell
python training\evaluate.py `
  --transactions data\transactions_cleaned.csv `
  --output artifacts\metrics.json
```

For a full-data final evaluation run, prefer an explicit day-based holdout:

```powershell
python -m training.evaluate `
  --transactions data\full_transactions_cleaned.csv `
  --test-days 7 `
  --output artifacts\full_metrics.json
```

## Prepare Raw H&M Articles Data

If you only have the original Kaggle `articles.csv`, convert it to the expected cleaned format first:

```powershell
python training\prepare_articles.py `
  --input data\raw\articles.csv `
  --images-dir data\raw\images `
  --output data\articles_cleaned.csv
```

## H&M Images

The Kaggle competition also provides product images. Put the extracted image folders under:

```text
data/raw/images/
```

The project resolves article images using the H&M competition folder pattern:

```text
data/raw/images/<first-3-digits-of-article-id>/<article-id>.jpg
```

Example:

```text
data/raw/images/011/0112679048.jpg
```

Once the images are present, rerun the article preparation step so the cleaned catalog includes image metadata, and the API will serve them automatically at `/catalog/images/{article_id}`.

If you do not have the Kaggle image archive on disk, the API can also fall back to public H&M-style product image URLs via `EXTERNAL_IMAGE_BASE_URL`. That keeps the frontend image-led even when `data/raw/images` is missing locally.

## Train LightGCN

```powershell
python training\train_gnn.py `
  --transactions data\transactions_cleaned.csv `
  --articles data\articles_cleaned.csv `
  --output-dir artifacts
```

For a full-data final run:

```powershell
python -m training.train_gnn `
  --transactions data\full_transactions_cleaned.csv `
  --articles data\full_articles_cleaned.csv `
  --output-dir artifacts\full `
  --test-days 7 `
  --epochs 10 `
  --samples-per-user 1
```

## Fine-tune the Encoder

```powershell
python training\finetune_encoder.py `
  --articles data\articles_cleaned.csv `
  --transactions data\transactions_cleaned.csv `
  --output-dir artifacts\fashion-minilm-finetuned
```

## Build The Semantic Search Index

```powershell
python -m training.build_semantic_index `
  --articles data\articles_cleaned.csv `
  --output-dir artifacts
```

The semantic builder prefers the sentence-transformer backend when it is available, and automatically falls back to a persisted `tfidf-svd` semantic model when the local transformer stack is unavailable. The API loads either backend from the generated artifacts.

## Run the API

```powershell
uvicorn api.main:app --reload
```

The API now includes:

- liveness and readiness endpoints at `/health` and `/health/ready`
- request tracing headers via `X-Request-ID` and `X-Process-Time-Ms`
- configurable CORS and environment-driven settings via `.env`
- richer response metadata for recommendation, search, and explanation calls

## Frontend

The frontend is scaffolded as a Next.js app. Install dependencies inside `frontend` before running it.

```powershell
cd frontend
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_BASE_URL` in `frontend/.env.example` or `frontend/.env.local` if your API is not running on `http://127.0.0.1:8000`.

## Local Quality Checks

For lightweight API and utility tests without installing the full training stack:

```powershell
pip install -r requirements-test.txt
python -m unittest discover -s tests -p "test_*.py"
```

For the frontend:

```powershell
cd frontend
npm run build
```

## Docker Compose

You can run the API and frontend together with Docker Compose:

```powershell
docker compose up --build
```

This starts:

- the API on `http://localhost:8000`
- the frontend on `http://localhost:3000`

## Notes

- The code is intentionally artifact-driven: the API starts even when trained assets are missing and falls back to deterministic demo behavior.
- Heavy dependencies such as `torch`, `torch-geometric`, `faiss`, `sentence-transformers`, and `lightgbm` are imported lazily where possible so the project can still be inspected and tested without every ML package present.
