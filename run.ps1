$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Create .venv and install requirements first (see README.md)."
}

if (-not (Test-Path "artifacts\models\avito-minilm-retriever\config.json")) {
    & $python scripts\train_dense_retriever.py
}
& $python scripts\run_baseline.py
& $python scripts\make_submission.py --output outputs\sparse_answer.csv
& $python scripts\build_hybrid_submission.py `
    --sparse-answer outputs\sparse_answer.csv `
    --output outputs\answer_new.csv

Move-Item -LiteralPath outputs\answer_new.csv -Destination answer.csv -Force
Remove-Item -LiteralPath outputs\sparse_answer.csv -Force
Remove-Item -LiteralPath outputs\predictions.parquet -Force
Remove-Item -LiteralPath outputs\hybrid_predictions.parquet -Force

Write-Host "Done: answer.csv"
