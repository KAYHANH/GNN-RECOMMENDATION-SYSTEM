# MLOps Notes

## MLflow

The GNN trainer already logs parameters and per-epoch loss whenever `mlflow` is installed:

```powershell
python training\train_gnn.py `
  --transactions data\transactions_cleaned.csv `
  --articles data\articles_cleaned.csv `
  --output-dir artifacts
```

To browse runs locally:

```powershell
mlflow ui
```

## DVC

When this project becomes its own Git repository, initialize DVC from the project root:

```powershell
dvc init
dvc add data\articles_cleaned.csv data\transactions_cleaned.csv
dvc add artifacts\semantic_faiss_index.bin artifacts\article_ids.csv
git add .dvc .gitignore data\*.dvc artifacts\*.dvc
```

If your embeddings or indexes live in a remote bucket, configure a DVC remote and use `dvc pull` inside CI before running evaluation.

## CI

The active workflow lives in the project root at `.github/workflows/eval.yml` because GitHub only executes workflows from that location.

