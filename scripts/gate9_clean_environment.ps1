# Gate 9 — reproduce from a clean environment.
#
# The brief's Gate 9 asks that the project run from a clean environment. That is two separable
# questions, and it is worth separating them because they cost very different amounts:
#
#   PHASE 1 (minutes)  Does a fresh virtual environment, built only from requirements.txt, contain
#                      everything the code needs? Does the test suite pass in it? Do the verifiers
#                      agree that every reported number traces to an artefact?
#
#   PHASE 2 (hours)    Does re-running the pipeline from raw data reproduce the reported numbers?
#
# Phase 1 catches the failures that are actually common -- a missing pin, an import that only works
# because a package happens to be installed system-wide -- and it needs no CPU to speak of, so it
# can run while something else occupies the machine. Phase 2 saturates all cores and must have the
# machine to itself; scripts/run_all.ps1 is the entry point for it.
#
# The fresh environment is created OUTSIDE the repository, so it cannot be found on sys.path and
# cannot accidentally satisfy an import from the development venv.
#
# Usage:  .\scripts\gate9_clean_environment.ps1
#         .\scripts\gate9_clean_environment.ps1 -Keep     (leave the environment in place)

param(
    [switch]$Keep,
    [string]$EnvPath = (Join-Path $env:TEMP "bettigraphs-gate9")
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

function Step($n, $text) { Write-Host "`n=== $n. $text ===" -ForegroundColor Cyan }

$failures = @()
function Check($name, $block) {
    try {
        & $block
        if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
        Write-Host "  PASS  $name" -ForegroundColor Green
    } catch {
        Write-Host "  FAIL  $name -- $_" -ForegroundColor Red
        $script:failures += $name
    }
}

Step 1 "Create a clean virtual environment at $EnvPath"
if (Test-Path $EnvPath) { Remove-Item -Recurse -Force $EnvPath }
# Explicitly WITHOUT --system-site-packages: the point is to find out what requirements.txt misses.
py -3.10 -m venv $EnvPath
$py = Join-Path $EnvPath "Scripts\python.exe"
& $py -m pip install --upgrade pip --quiet

Step 2 "Install from requirements.txt alone"
& $py -m pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) { throw "installation from requirements.txt failed" }
& $py -m pip freeze | Out-File -Encoding utf8 (Join-Path $root "results\raw\gate9_freeze.txt")

Step 3 "Import every module the repository defines"
# A missing pin usually shows up here rather than in the tests, because the tests do not import
# the scripts.
Check "all modules import" { & $py -c @"
import importlib, pathlib, sys, traceback
sys.path.insert(0, '.')
bad = []
for path in sorted(pathlib.Path('src').rglob('*.py')) + sorted(pathlib.Path('scripts').rglob('*.py')):
    if '__pycache__' in path.parts: continue
    name = str(path.with_suffix('')).replace('\\', '.').replace('/', '.')
    try:
        importlib.import_module(name)
    except Exception as exc:
        bad.append(name + ': ' + type(exc).__name__ + ': ' + str(exc)[:120])
for b in bad: print('  ' + b)
print(len(bad), 'modules failed to import')
raise SystemExit(1 if bad else 0)
"@ }

Step 4 "Run the test suite in the clean environment"
Check "pytest" { & $py -m pytest tests -q }

Step 5 "Every reported number traces to an artefact"
Check "verify_numbers" { & $py scripts\verify_numbers.py --docs manuscript docs\DATA_FINDINGS.md docs\REVIEWER_QUESTIONS.md tables }

Step 6 "The manuscript builds from the artefacts"
Check "build_manuscript" { & $py scripts\build_manuscript.py --check }

Step 7 "Regenerate the derived artefacts and confirm they are unchanged"
# These are cheap and deterministic, so a clean environment must reproduce them byte for byte.
# A difference here means a result depends on something the environment does not pin.
Check "graph structure is byte-identical" { & $py -c @"
import hashlib, pathlib, subprocess, sys
target = pathlib.Path('results/processed/graph_structure.json')
before = hashlib.sha256(target.read_bytes()).hexdigest()
subprocess.run([sys.executable, 'scripts/measure_graph_structure.py',
                '--sections', 'global', 'values', 'appearances'], check=True)
after = hashlib.sha256(target.read_bytes()).hexdigest()
print('before', before[:16], 'after', after[:16])
raise SystemExit(0 if before == after else 1)
"@ }
Check "feature table is byte-identical" { & $py -c @"
import hashlib, pathlib, subprocess, sys
target = pathlib.Path('tables/table13_feature_definitions.md')
before = hashlib.sha256(target.read_bytes()).hexdigest()
subprocess.run([sys.executable, 'scripts/make_feature_table.py'], check=True)
after = hashlib.sha256(target.read_bytes()).hexdigest()
print('before', before[:16], 'after', after[:16])
raise SystemExit(0 if before == after else 1)
"@ }
Check "splits are byte-identical" { & $py -c @"
import hashlib, pathlib, subprocess, sys
target = pathlib.Path('results/processed/splits.json')
before = hashlib.sha256(target.read_bytes()).hexdigest()
subprocess.run([sys.executable, 'scripts/record_splits.py'], check=True)
after = hashlib.sha256(target.read_bytes()).hexdigest()
print('before', before[:16], 'after', after[:16])
raise SystemExit(0 if before == after else 1)
"@ }

Write-Host "`n================ Gate 9, phase 1 ================" -ForegroundColor Cyan
if ($failures.Count -eq 0) {
    Write-Host "all checks passed in a clean environment" -ForegroundColor Green
} else {
    Write-Host ("failed: " + ($failures -join ", ")) -ForegroundColor Red
}
Write-Host "environment: $EnvPath"
Write-Host "Phase 2 (full pipeline from raw data) is scripts\run_all.ps1, and needs the machine."

if (-not $Keep) { Remove-Item -Recurse -Force $EnvPath }
exit $failures.Count
