$ErrorActionPreference = 'Continue'
$env:STRATA_A3B_Q6_GROUPED_EXACT = '1'
$env:STRATA_A3B_MTP_KV_ONLY = '0'
$exe = 'G:\Strata-Moe\build-a3b\bin\strata-moe-native-dense-parity.exe'
$folder = 'G:\Strata-Moe\bench\results\2026-10-15-a3b-iq3s-q6-reuse-exact'
foreach ($reuse in @('0', '1')) {
    $env:STRATA_A3B_Q6_REUSE_EXACT = $reuse
    & $exe --grouped-q6-hook --q6-real-shapes 2>&1 |
        Tee-Object -FilePath (Join-Path $folder "parity-reuse-$reuse.log")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
