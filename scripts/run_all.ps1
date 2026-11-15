# Reproduce the whole study, in the order the experiments must run.
#
# Two rules are encoded here because getting either wrong costs hours.
#
# 1. ONE AT A TIME. Every stage saturates all cores. Running two concurrently does not halve the
#    wall time, it roughly doubles it: an earlier session had two jobs oversubscribing 32 cores
#    and neither made progress for the best part of an hour.
#
# 2. SELECTION BEFORE EVALUATION. The pre-registration requires the radius, filtration, complex
#    dimension and curve resolution to be chosen on the validation split. Stage 3 does that, and
#    the later stages should be re-run with whatever it selects if that differs from the declared
#    starting values.
#
# The transaction experiment checkpoints after each (model, family) cell and skips completed cells
# on a re-run, so it is safe to interrupt and resume. It is split into a fast pass and a slow pass
# so that a complete set of results exists early: random forest alone is roughly as expensive as
# the other three models combined.
#
# Usage:  .\scripts\run_all.ps1            (everything)
#         .\scripts\run_all.ps1 -From 4    (resume from stage 4)

param(
    [int]$From = 1,
    [int]$To = 99
)

$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$root = Join-Path $PSScriptRoot ".."
Set-Location $root

function Invoke-Stage {
    param([int]$Number, [string]$Name, [string[]]$Arguments)
    if ($Number -lt $From -or $Number -gt $To) {
        Write-Host "[$Number] skipped: $Name" -ForegroundColor DarkGray
        return
    }
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host "[$Number] $Name" -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
    $started = Get-Date
    & $py -u @Arguments
    if ($LASTEXITCODE -ne 0) { throw "stage $Number ($Name) failed with exit code $LASTEXITCODE" }
    $elapsed = (Get-Date) - $started
    Write-Host ("[{0}] done in {1:hh\:mm\:ss}" -f $Number, $elapsed) -ForegroundColor Green
}

Invoke-Stage 1 "Download datasets (skips files already present)" @(
    "data\download_scripts\get_elliptic.py")
Invoke-Stage 1 "Download Elliptic++ (Git LFS)" @(
    "data\download_scripts\get_elliptic_plus_plus.py")

Invoke-Stage 2 "Validate datasets, build Table 3" @("scripts\validate_datasets.py")

Invoke-Stage 3 "Select topological settings on validation, build Table 4" @(
    "experiments\betti\select_topology_settings.py")

Invoke-Stage 4 "Test H1: class separation with size-matched controls" @(
    "experiments\betti\test_h1_class_separation.py", "--radius", "3")

Invoke-Stage 5 "Test H4: snapshot-level topology" @(
    "experiments\temporal\snapshot_topology.py")

Invoke-Stage 6 "Test H3: wallet retrieval" @(
    "experiments\hybrid\run_wallet_similarity.py",
    "--n-queries", "2000", "--n-pool", "20000", "--exact-n", "250")

Invoke-Stage 7 "Within-T ablation, build Table 8" @(
    "experiments\ablation\run_topological_ablation.py", "--radius", "3")

Invoke-Stage 8 "Measure scalability, build Table 10" @("scripts\measure_scalability.py")

Invoke-Stage 9 "Transaction comparison, fast models (resumable)" @(
    "experiments\baseline\run_transaction_experiment.py", "--radius", "3",
    "--models", "logistic", "xgboost", "lightgbm")

Invoke-Stage 10 "Transaction comparison, random forest (resumable, slow)" @(
    "experiments\baseline\run_transaction_experiment.py", "--radius", "3",
    "--models", "random_forest")

Invoke-Stage 11 "Test H5: entity-level temporal trajectories" @(
    "experiments\temporal\entity_trajectories.py")

Invoke-Stage 12 "Statistical analysis, build Tables 5-9" @(
    "scripts\analyse_transaction_results.py")

Invoke-Stage 13 "Figures from results" @("scripts\make_figures.py")
Invoke-Stage 14 "Schematic figures" @("scripts\make_schematics.py")

Write-Host ""
Write-Host "All requested stages complete." -ForegroundColor Green
Write-Host "Tables in tables\, figures in figures\, raw results in results\." -ForegroundColor Green
