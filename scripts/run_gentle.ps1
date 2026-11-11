# Run gentle GRPO training; when train_grpo.py exits with code 3 (RAM ceiling hit), resume from the last checkpoint.
# Usage: powershell -File scripts/run_gentle.ps1 [extra train_grpo.py args]
Set-Location (Split-Path $PSScriptRoot)
$env:TMP = "$PWD\.tmp"; $env:TEMP = "$PWD\.tmp"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force .tmp, logs | Out-Null
$resume = @()
if (Test-Path outputs\grpo\checkpoint-*) { $resume = @("--resume") }
do {
    # cmd redirection keeps the log plain UTF-8 (PowerShell 5.1 would write UTF-16).
    cmd /c ".venv\Scripts\python.exe -u scripts/train_grpo.py $(($args + $resume) -join ' ') >> logs\train.log 2>&1"
    $code = $LASTEXITCODE
    cmd /c "echo [run_gentle] train_grpo.py exited $code at $(Get-Date -Format s) >> logs\train.log"
    $resume = @("--resume")
} while ($code -eq 3)
