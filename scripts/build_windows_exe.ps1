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
        Sort-Object @{ Expression = "LastWriteTimeUtc"; Descending = $true },
            @{ Expression = { if ($_.Name -like "数据分析-*-setup.exe") { 0 } else { 1 } }; Ascending = $true },
            @{ Expression = "Name"; Ascending = $true }

    if ($installers.Count -le 1) {
        return
    }

    $installers | Select-Object -Skip 1 | ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Force
    }
}

python scripts\generate_app_icon.py
if ($LASTEXITCODE -ne 0) {
    throw "Icon generation failed with exit code $LASTEXITCODE."
}
python -m PyInstaller --clean --noconfirm packaging\data_analysis.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE."
}

# 构建成功不代表 DLL 能加载；必须运行最终 EXE，避免把导入失败的程序装入安装包。
$smokeProcess = Start-Process -FilePath (Join-Path $Root "dist\数据分析.exe") `
    -ArgumentList "--smoke-test-imports" -WindowStyle Hidden -PassThru
if (-not $smokeProcess.WaitForExit(120000)) {
    taskkill /PID $smokeProcess.Id /T /F | Out-Null
    throw "Packaged application import check timed out."
}
if ($smokeProcess.ExitCode -ne 0) {
    throw "Packaged application import check failed with exit code $($smokeProcess.ExitCode)."
}

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
