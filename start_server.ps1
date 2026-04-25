param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$condaPython = "C:/Users/Apar2/miniconda3/envs/frequency-print/python.exe"
$venvPython = Join-Path $projectRoot "venv/Scripts/python.exe"

if (Test-Path $condaPython) {
    $pythonExe = $condaPython
} elseif (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    throw "No Python runtime found. Expected '$condaPython' or '$venvPython'."
}

Write-Host "Starting server with: $pythonExe"
& $pythonExe -m uvicorn app:app --host $BindHost --port $Port --reload
