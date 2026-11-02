<#
Copy sweep results from the server to this Windows PC.

  .\scripts\fetch_sweep.ps1          download the newest sweep folder now
  .\scripts\fetch_sweep.ps1 -Run     open the main.py menu on the server; when you quit
                                     it, download the sweep you just ran
  .\scripts\fetch_sweep.ps1 -Wait    wait for the sweep running on the server (e.g. in tmux)
                                     to finish, then download it

A sweep counts as finished once sweep_summary.png exists (it is written last).
If PowerShell refuses to run scripts:
  powershell -ExecutionPolicy Bypass -File .\scripts\fetch_sweep.ps1 -Wait
#>
param(
    [string]$Server = "girija@10.10.0.176",
    [string]$Repo = "/home/girija/flatland-blackbox",
    [string]$Dest = "$env:USERPROFILE\Downloads",
    [string]$Command = "python main.py",   # how main.py is started on the server (e.g. activate a venv first)
    [switch]$Run,
    [switch]$Wait,
    [int]$PollSeconds = 60
)

$sweeps = "$Repo/outputs/sweeps"

function Get-Newest {
    $out = ssh $Server "ls -dt $sweeps/*/ 2>/dev/null | head -n 1"
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
    return ([string]$out).Trim().TrimEnd('/')
}

function Test-Finished([string]$folder) {
    if (-not $folder) { return $false }
    ssh $Server "test -f '$folder/sweep_summary.png'"
    return ($LASTEXITCODE -eq 0)
}

$before = Get-Newest
$beforeDone = Test-Finished $before

if ($Run) {
    ssh -t $Server "cd $Repo && $Command"
    $target = Get-Newest
    if (-not $target -or $target -eq $before) {
        Write-Host "No new sweep was started; nothing to download."
        exit 0
    }
    if (-not (Test-Finished $target)) {
        Write-Host "The sweep $target did not finish; downloading what exists so far."
    }
}
elseif ($Wait) {
    Write-Host "Waiting for the sweep on $Server to finish (checking every $PollSeconds s, Ctrl+C to stop)..."
    while ($true) {
        $target = Get-Newest
        $isNew = ($target -ne $before) -or (-not $beforeDone)
        if ($target -and $isNew -and (Test-Finished $target)) { break }
        Start-Sleep -Seconds $PollSeconds
    }
}
else {
    $target = $before
    if (-not $target) { Write-Host "No sweeps found in $sweeps."; exit 1 }
}

$name = Split-Path $target -Leaf
Write-Host "Downloading $name to $Dest ..."
scp -r "${Server}:$target" "$Dest"
if ($LASTEXITCODE -ne 0) { Write-Host "scp failed."; exit 1 }
Write-Host "Done: $Dest\$name"
explorer "$Dest\$name"
