$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Create .venv and install requirements first (see README.md)."
}

if (-not (Test-Path "artifacts\models\avito-minilm-retriever\config.json")) {
    & $python scripts\train_dense_retriever.py
}
& $python scripts\run_baseline.py
& $python scripts\make_submission.py
& $python scripts\build_hybrid_submission.py --sparse-answer answer.csv --output answer.csv

Write-Host "Done: answer.csv"
