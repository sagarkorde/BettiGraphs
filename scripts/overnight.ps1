# Finish the study unattended, in the order the stages must run.
#
# Everything still outstanding is sequential and gated on the canonical transaction experiment, so
# this is one script rather than a set of jobs: stage 1 finishes the experiment, stages 2-7 bring
# the radius-dependent results into conformance with it, and stages 8-10 rebuild and check the
# manuscript. At the end the build should go green.
#
# Two rules are encoded here because getting either wrong costs hours:
#
#   ONE STAGE AT A TIME. Every stage saturates all cores. Two concurrent jobs once made no
#   progress for the best part of an hour.
#
#   NEVER EDIT A SHELL SCRIPT THAT IS EXECUTING. PowerShell is less exposed to this than bash,
#   which reads by byte offset, but the habit is worth keeping.
#
# Resumability. The transaction experiment checkpoints per (model, family) cell and skips completed
# cells, so stage 1 resumes rather than restarts. Stages 2-7 are idempotent: re-running one
# recomputes its outputs from the same inputs. So this script can be killed and restarted at any
# point, and the worst case is repeating the stage that was in flight.
#
# Why a scheduled task rather than a background process: three attempts to run the experiment
# detached from the Claude session were killed when the session went idle, each time cleanly one
# line after a checkpoint. Task Scheduler runs it under the service, outside that process group.
#
# Usage:  .\scripts\overnight.ps1
#         .\scripts\overnight.ps1 -From 3     (skip the experiment, start at the analysis)

param(
    [int]$From = 1,
    [int]$To = 99
)

$ErrorActionPreference = "Continue"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "logs\overnight.log"
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null

$script:failed = @()

function Say($text) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $text
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

function Stage($number, $name, [scriptblock]$work) {
    if ($number -lt $From -or $number -gt $To) { Say "SKIP  $number. $name"; return }
    Say "START $number. $name"
    $started = Get-Date
    # Out-File with an explicit encoding, not Tee-Object: the scheduled task runs under Windows
    # PowerShell 5.1, where Tee-Object writes UTF-16 and the morning log is unreadable as text.
    & $work *>&1 | Out-File -FilePath $log -Encoding utf8 -Append
    $minutes = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
    if ($LASTEXITCODE -ne 0) {
        Say "FAIL  $number. $name (exit $LASTEXITCODE after $minutes min)"
        $script:failed += "$number. $name"
    } else {
        Say "OK    $number. $name ($minutes min)"
    }
}

Say "================ overnight run starting ================"
Say "python: $py"

Stage 1 "Canonical transaction experiment, radius 2, 20 seeds (resumes from its checkpoint)" {
    & $py experiments\baseline\run_transaction_experiment.py --radius 2 --tag r2s20
}

Stage 2 "Preserve the radius-3 artefacts before the analysis overwrites them" {
    # The analysis writes to fixed filenames. Copy once and never again, so that re-running this
    # script cannot overwrite the preserved copies with radius-2 results.
    $pairs = @(
        @("results\statistical_tests\hypothesis_tests.json",        "results\statistical_tests\hypothesis_tests_r3.json"),
        @("results\statistical_tests\regime_shift_degradation.csv", "results\statistical_tests\regime_shift_degradation_r3.csv"),
        @("results\statistical_tests\family_selected_summary.csv",  "results\statistical_tests\family_selected_summary_r3.csv"),
        @("results\statistical_tests\family_model_summary.csv",     "results\statistical_tests\family_model_summary_r3.csv")
    )
    foreach ($pair in $pairs) {
        if ((Test-Path $pair[0]) -and -not (Test-Path $pair[1])) {
            Copy-Item $pair[0] $pair[1]
            Say "  preserved $($pair[1])"
        } else {
            Say "  already preserved or absent: $($pair[1])"
        }
    }
    $global:LASTEXITCODE = 0
}

Stage 3 "Re-analyse at radius 2" {
    & $py scripts\analyse_transaction_results.py --tag r2s20
}

Stage 4 "Ablation at radius 2" {
    & $py experiments\ablation\run_topological_ablation.py --radius 2
}

Stage 5 "H1 descriptor screen at radius 2" {
    & $py experiments\betti\test_h1_class_separation.py --radius 2
}

Stage 6 "Interpretability at radius 2" {
    & $py experiments\hybrid\run_interpretability.py --radius 2
}

Stage 7 "One-class anomaly score at radius 2, both views" {
    & $py experiments\betti\run_anomaly_score.py --radius 2 --views tx addr
}

Stage 8 "Regenerate the figures" {
    & $py scripts\make_figures.py --tag r2s20
    & $py scripts\make_schematics.py
}

Stage 9 "Verify every reported number against its artefact" {
    & $py scripts\verify_numbers.py --docs manuscript docs\DATA_FINDINGS.md docs\REVIEWER_QUESTIONS.md tables
}

Stage 10 "Build the manuscript (this is the conformance gate)" {
    & $py scripts\build_manuscript.py
}

Say "================ overnight run finished ================"
if ($script:failed.Count -eq 0) {
    Say "all stages succeeded"
} else {
    Say ("stages that failed: " + ($script:failed -join "; "))
}

# A short summary the morning can be read from without opening the log.
$summary = Join-Path $root "logs\overnight_summary.txt"
& $py -c @"
import json, pathlib, subprocess
out = []
p = pathlib.Path('results/raw/transaction_results_r2s20_r2_value_superlevel_cd2.csv')
if p.exists():
    import pandas as pd
    d = pd.read_csv(p)
    out.append('transaction experiment: %d of 32 cells, %d records, %d seeds'
               % (len(d.groupby(['model','family'])), len(d), d.seed.nunique()))
for name in ('topological_ablation', 'interpretability', 'h1_class_separation', 'anomaly_scores'):
    f = pathlib.Path('results/statistical_tests/%s.json' % name)
    if f.exists():
        try:
            radius = json.loads(f.read_text(encoding='utf-8')).get('radius')
        except Exception:
            radius = '?'
        out.append('%-22s radius %s' % (name, radius))
out.append('')
out.append('git: ' + subprocess.run(['git','rev-list','--count','HEAD'],
                                    capture_output=True, text=True).stdout.strip() + ' commits')
print('\n'.join(out))
"@ | Out-File -FilePath $summary -Encoding utf8
Say "summary written to $summary"

exit $script:failed.Count
