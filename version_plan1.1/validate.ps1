param(
    [string]$Python = (Join-Path $PSScriptRoot '..\.venv-hpo\Scripts\python.exe'),
    [switch]$Focused
)
$ErrorActionPreference = 'Stop'
$validationRoot = Join-Path $PSScriptRoot 'validation'
New-Item -ItemType Directory -Path $validationRoot -Force | Out-Null
$runName = 'check-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
$temporaryRoot = Join-Path $validationRoot ($runName + '-tmp')
$reportPath = Join-Path $validationRoot ($runName + '.xml')
$logPath = Join-Path $validationRoot ($runName + '.log')
$testArgs = @('-m', 'pytest', '-q', "--basetemp=$temporaryRoot", "--junitxml=$reportPath", '-o', "cache_dir=$validationRoot/pytest-cache")
if ($Focused) {
    $testArgs += @('tests/unit/test_version_plan11.py', 'tests/integration/test_version_plan11_lifecycle.py', 'tests/integration/test_complete_system.py', 'tests/unit/test_data_processing.py', 'tests/unit/test_langgraph_workflows.py', 'tests/unit/test_data_handoff.py')
}
Push-Location (Join-Path $PSScriptRoot 'project')
try {
    & $Python @testArgs 2>&1 | Tee-Object -FilePath $logPath
    $testExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $testExitCode
