$ErrorActionPreference = "Stop"

# Checking if script is executed from the correct directory (backend/root)
if (-not (Test-Path ".\evaluation")) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "ERROR: Directory Mismatch!" -ForegroundColor Red
    Write-Host "Please run this script from the project root / backend folder:" -ForegroundColor Yellow
    Write-Host "   .\evaluation\run_evaluation.ps1" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    exit 1
}

$steps = @(
    "evaluation.prepare_eval_data",
    "evaluation.retrieval_eval",
    "evaluation.generation_eval"
)

foreach ($step in $steps) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Running: $step" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    python -m $step

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Error "Failed: $step"
        exit $LASTEXITCODE
    }

    Write-Host "Completed: $step" -ForegroundColor Green
}

Write-Host ""
Write-Host "All evaluation steps completed successfully." -ForegroundColor Green