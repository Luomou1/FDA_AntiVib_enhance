param(
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

python scripts\generate_app_icon.py
python -m PyInstaller --clean --noconfirm packaging\data_analysis.spec

if (-not $SkipInstaller) {
    $iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    $isccPath = $null
    if ($null -ne $iscc) {
        $isccPath = $iscc.Source
    }
    if ($null -eq $iscc) {
        $userIscc = Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
        if (Test-Path $userIscc) {
            $isccPath = $userIscc
        }
    }
    if ($null -ne $isccPath) {
        & $isccPath packaging\data_analysis.iss
    }
    else {
        Write-Warning "Inno Setup ISCC.exe was not found. Built dist\数据分析.exe only; installer was skipped."
    }
}
