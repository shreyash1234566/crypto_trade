param(
    [string]$PythonExe = "python",
    [string]$ScriptPath = "scripts/tune_hyperparameters.py",
    [string[]]$ScriptArgs = @(),
    [int]$RestoreStandbyMinutes = 30,
    [int]$RestoreMonitorMinutes = 10
)

function Get-PowerTimeoutMinutes {
    param(
        [Parameter(Mandatory = $true)][string]$Alias
    )

    $query = powercfg /query | Out-String
    $pattern = "GUID Alias:\s+$Alias\s.*?Current AC Power Setting Index:\s+0x([0-9a-fA-F]+)"
    $match = [regex]::Match($query, $pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)
    if ($match.Success) {
        return [Convert]::ToInt32($match.Groups[1].Value, 16)
    }
    return $null
}

function Set-TimeoutMinutes {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("standby", "monitor")][string]$Type,
        [Parameter(Mandatory = $true)][int]$Minutes
    )

    if ($Type -eq "standby") {
        powercfg /change standby-timeout-ac $Minutes | Out-Null
    }
    elseif ($Type -eq "monitor") {
        powercfg /change monitor-timeout-ac $Minutes | Out-Null
    }
}

$origStandby = Get-PowerTimeoutMinutes -Alias "STANDBYIDLE"
$origMonitor = Get-PowerTimeoutMinutes -Alias "VIDEOIDLE"
if ($null -eq $origStandby) { $origStandby = $RestoreStandbyMinutes }
if ($null -eq $origMonitor) { $origMonitor = $RestoreMonitorMinutes }

Write-Host "Disabling AC standby and monitor timeouts..."
Set-TimeoutMinutes -Type "standby" -Minutes 0
Set-TimeoutMinutes -Type "monitor" -Minutes 0

try {
    $command = @($ScriptPath) + $ScriptArgs
    Write-Host "Running: $PythonExe $command"
    & $PythonExe @command
}
finally {
    Write-Host "Restoring AC standby timeout to $origStandby minutes"
    Set-TimeoutMinutes -Type "standby" -Minutes $origStandby
    Write-Host "Restoring AC monitor timeout to $origMonitor minutes"
    Set-TimeoutMinutes -Type "monitor" -Minutes $origMonitor
}
