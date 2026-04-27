param(
    [string]$PythonExe = "py",
    [string]$ConfigPath = "configs/default.yaml"
)

$datasets = @("politifact", "gossipcop", "fang")
$fewshots = @(16, 32, 128)

foreach ($dataset in $datasets) {
    foreach ($shot in $fewshots) {
        Write-Host "Running dataset=$dataset few-shot=$shot"
        & $PythonExe -3 scripts/run_pmtl_disco.py `
            --config $ConfigPath `
            --dataset $dataset `
            --few-shot $shot

        if ($LASTEXITCODE -ne 0) {
            throw "Experiment failed for dataset=$dataset few-shot=$shot"
        }
    }
}

Write-Host "All PMTL-DisCo experiments finished."
