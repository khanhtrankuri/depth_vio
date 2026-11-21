# Windows-side helper for the WSL2 VIO environment.
# Usage (PowerShell):  .\scripts\vio.ps1 check|rviz|run <config>|play <bag_dir> [rate]|stop|shell|shutdown
param(
  [Parameter(Mandatory = $true)][ValidateSet('check', 'rviz', 'run', 'play', 'stop', 'shell', 'shutdown')][string]$Cmd,
  [string]$Arg1,
  [string]$Arg2
)
$Distro = 'Ubuntu-22.04'
# Tolerate stray trailing backslashes/quotes (they break the quoting passed to WSL).
if ($Arg1) { $Arg1 = $Arg1.Trim().TrimEnd('\', '"', "'") }
if ($Arg2) { $Arg2 = $Arg2.Trim().TrimEnd('\', '"', "'") }
$Scripts = (wsl -d $Distro -- wslpath -a ($PSScriptRoot -replace '\\', '/')).Trim()

function Invoke-Wsl([string]$bashCmd) { wsl -d $Distro -- bash -lc $bashCmd }

switch ($Cmd) {
  'check'    { Invoke-Wsl "bash $Scripts/check_env.sh" }
  'rviz'     { Invoke-Wsl "bash $Scripts/run_rviz.sh" }
  'run'      { if (-not $Arg1) { throw 'run needs a config name or path' }; Invoke-Wsl "bash $Scripts/run_openvins.sh $Arg1" }
  'play'     { if (-not $Arg1) { throw 'play needs a ROS2 bag directory' }; $r = if ($Arg2) { $Arg2 } else { '1.0' }; Invoke-Wsl "bash $Scripts/play_bag.sh $Arg1 $r" }
  'stop'     { Invoke-Wsl "bash $Scripts/stop_all.sh" }
  'shell'    { wsl -d $Distro }
  'shutdown' { wsl --terminate $Distro }   # only stops Ubuntu-22.04, leaves docker-desktop untouched
}
