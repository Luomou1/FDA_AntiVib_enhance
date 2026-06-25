param(
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

function Remove-OldInstallerExe {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallerDir
    )

    if (-not (Test-Path $InstallerDir)) {
        return
    }

    $installers = Get-ChildItem -Path $InstallerDir -Filter "*.exe" -File |
        Sort-Object LastWriteTimeUtc -Descending

    if ($installers.Count -le 1) {
        return
    }

    $installers | Select-Object -Skip 1 | ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Force
    }
}

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
        if ($LASTEXITCODE -ne 0) {
            throw "Inno Setup failed with exit code $LASTEXITCODE."
        }
        Remove-OldInstallerExe -InstallerDir (Join-Path $Root "dist\installer")
    }
    else {
        Write-Warning "Inno Setup ISCC.exe was not found. Built dist\数据分析.exe only; installer was skipped."
    }
}
