param(
    [Parameter(Mandatory = $true)]
    [string]$Tag,

    [string]$Repo = "Luomou1/FDA_AntiVib_enhance"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$installer = Get-ChildItem -Path "dist\installer" -Filter "*.exe" -File |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1

if ($null -eq $installer) {
    throw "No installer exe found in dist\installer."
}

$version = $Tag.TrimStart("v")
$tempAsset = Join-Path ([System.IO.Path]::GetTempPath()) "DataAnalysis-$version-setup.exe"

try {
    Copy-Item -LiteralPath $installer.FullName -Destination $tempAsset -Force
    gh release upload $Tag $tempAsset --repo $Repo --clobber
}
finally {
    if (Test-Path $tempAsset) {
        Remove-Item -LiteralPath $tempAsset -Force
    }
}
