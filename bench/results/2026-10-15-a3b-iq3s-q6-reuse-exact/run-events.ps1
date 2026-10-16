$ErrorActionPreference = 'Continue'
$env:STRATA_A3B_Q6_GROUPED_EXACT = '1'
$env:STRATA_A3B_MTP_KV_ONLY = '0'
$exe = 'G:\Strata-Moe\build-a3b\bin\strata-moe-native-dense-parity.exe'
$folder = 'G:\Strata-Moe\bench\results\2026-10-15-a3b-iq3s-q6-reuse-exact'
$arms = @('0', '1', '1', '0')
for ($i = 0; $i -lt $arms.Count; $i++) {
    $reuse = $arms[$i]
    $env:STRATA_A3B_Q6_REUSE_EXACT = $reuse
    $stdout = Join-Path $folder "events-$i-reuse-$reuse.stdout.log"
    $stderr = Join-Path $folder "events-$i-reuse-$reuse.stderr.log"
    & $exe --grouped-q6-hook --q6-real-shapes --benchmark-events 200 1> $stdout 2> $stderr
    if ($LASTEXITCODE -ne 0) { Get-Content -LiteralPath $stderr; exit $LASTEXITCODE }
    Write-Output "$i reuse=$reuse completed"
}
