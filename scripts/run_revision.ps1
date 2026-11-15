# Runs the BCRA revision experiments (Amendment 5) in order, cheapest first.
# Each step is skipped if its result JSON already exists, so re-running resumes.
# Run from the repository root in a terminal you will leave open:
#     powershell -ExecutionPolicy Bypass -File scripts\run_revision.ps1
# Optional:  -Only rolling_origin,label_noise     -Jobs 8

param([string[]]$Only = @(), [int]$Jobs = -1)
$ErrorActionPreference = "Continue"
$py = ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path logs, results\revision | Out-Null

$steps = @(
  @{n="equivalence";      out="results\statistical_tests\equivalence_tests.json"; cmd=@("scripts\equivalence_tests.py","--tag","r2s20")},
  @{n="hyp5_feasibility"; out="results\revision\hyp5_feasibility.json";   cmd=@("experiments\revision\hyp5_feasibility.py")},
  @{n="screen_tx";        out="results\revision\screen_G_tx.json";        cmd=@("experiments\revision\screen.py","--view","G_tx","--predict")},
  @{n="screen_addr";      out="results\revision\screen_G_addr.json";      cmd=@("experiments\revision\screen.py","--view","G_addr","--predict")},
  @{n="synthetic";        out="results\revision\synthetic_density.json";  cmd=@("experiments\revision\synthetic_density.py")},
  @{n="feature_selected"; out="results\revision\feature_selected_T.json"; cmd=@("experiments\revision\feature_selected_T.py","--n-jobs",$Jobs)},
  @{n="label_noise";      out="results\revision\label_noise.json";        cmd=@("experiments\revision\label_noise.py","--n-jobs",$Jobs)},
  @{n="rolling_origin";   out="results\revision\rolling_origin.json";     cmd=@("experiments\revision\rolling_origin.py","--n-jobs",$Jobs)},
  @{n="gnn";              out="results\revision\gnn_baseline.json";       cmd=@("experiments\revision\gnn_baseline.py","--n-jobs",$Jobs)},
  @{n="tx_on_address";    out="results\revision\tx_on_address_view.json"; cmd=@("experiments\revision\tx_on_address_view.py","--n-jobs",$Jobs)},
  @{n="hyp3_controls";    out="results\revision\hyp3_controls.json";      cmd=@("experiments\revision\hyp3_controls.py","--n-jobs",$Jobs)}
)

foreach ($s in $steps) {
  if ($Only.Count -gt 0 -and -not ($Only -contains $s.n)) { continue }
  if (Test-Path $s.out) { Write-Host "skip $($s.n): $($s.out) exists"; continue }
  $log = "logs\revision_$($s.n).log"
  Write-Host "$(Get-Date -Format s)  running $($s.n)  -> $log"
  & $py @($s.cmd) *>&1 | Tee-Object -FilePath $log
  if (Test-Path $s.out) { Write-Host "done $($s.n)" } else { Write-Host "FAILED $($s.n); see $log" }
}
