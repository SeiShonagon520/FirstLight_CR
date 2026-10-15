param(
    [string]$VmName = "",
    [int]$VmIndex = 0,
    [string]$Serial = "",
    [string]$Package = "nullsroyale.rel.free",
    [string]$Activity = "com.supercell.clashroyale.GameApp",
    [int]$ControlPort = 0,
    [int]$GuestControlPort = 0,
    [string]$MuMuManager = "",
    [string]$Adb = "",
    [switch]$RestoreOffline,
    [switch]$NoLaunch
)

# Online reconnaissance session: same dedicated VM and same patched client as the
# offline harness, but with the CR_NATIVE_OFFLINE chain removed so the client can
# reach its server.  -RestoreOffline puts the isolation rules back.

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "local_config.ps1")
if (-not $PSBoundParameters.ContainsKey("VmIndex")) { $VmIndex = Get-LocalSetting "CR_VM_INDEX" }
if (-not $PSBoundParameters.ContainsKey("VmName")) { $VmName = Get-LocalSetting "CR_VM_NAME" }
if (-not $PSBoundParameters.ContainsKey("Serial")) { $Serial = Get-LocalSetting "CR_ADB_SERIAL" }
if (-not $PSBoundParameters.ContainsKey("ControlPort")) { $ControlPort = Get-LocalSetting "CR_CONTROL_PORT" "26789" }
if (-not $PSBoundParameters.ContainsKey("GuestControlPort")) { $GuestControlPort = Get-LocalSetting "CR_GUEST_CONTROL_PORT" "26789" }
if (-not $PSBoundParameters.ContainsKey("MuMuManager")) { $MuMuManager = Get-LocalSetting "CR_MUMU_MANAGER" }
if (-not $PSBoundParameters.ContainsKey("Adb")) { $Adb = Get-LocalSetting "CR_ADB" }

if ($VmIndex -lt 0 -or -not $VmName -or $Serial -notmatch '^127\.0\.0\.1:[1-9][0-9]*$') {
    throw "Configure a dedicated VM index, name and loopback ADB serial in .env."
}
foreach ($required in @($MuMuManager, $Adb)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing required file: $required"
    }
}

function Invoke-Adb {
    param(
        [Parameter(Mandatory)][string[]]$Arguments,
        [switch]$AllowFailure
    )
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Adb @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if (-not $AllowFailure -and $exitCode -ne 0) {
        throw "adb failed: $($Arguments -join ' ')`n$($output -join "`n")"
    }
    return [pscustomobject]@{
        ExitCode = $exitCode
        Output   = ($output -join "`n").Trim()
    }
}

$infoOutput = & $MuMuManager info --vmindex "$VmIndex"
if ($LASTEXITCODE -ne 0) {
    throw "MuMuManager could not inspect VM index $VmIndex."
}
$info = ($infoOutput -join "`n") | ConvertFrom-Json
if ([int]$info.index -ne $VmIndex) {
    throw "MuMu index $($info.index) does not match requested VM index $VmIndex."
}
if ($VmName -and $info.name -and ($info.name -ne $VmName) -and ($info.name -notmatch "MuMu")) {
    throw "MuMu index $VmIndex name ($($info.name)) does not match expected $VmName."
}
if (-not [bool]$info.is_process_started -or $info.player_state -ne "start_finished") {
    Write-Host "Launching VM index $VmIndex ($VmName)..."
    & $MuMuManager control --vmindex "$VmIndex" launch | Out-Null
    $bootDeadline = [DateTime]::UtcNow.AddSeconds(90)
    do {
        Start-Sleep -Milliseconds 1000
        $infoOutput = & $MuMuManager info --vmindex "$VmIndex"
        $info = ($infoOutput -join "`n") | ConvertFrom-Json
        if ($info.is_process_started -and $info.player_state -eq "start_finished") {
            break
        }
    } while ([DateTime]::UtcNow -lt $bootDeadline)
    if (-not $info.is_process_started -or $info.player_state -ne "start_finished") {
        throw "$VmName failed to start within 90 seconds."
    }
}
if ($info.adb_host_ip -ne "127.0.0.1" -or $Serial -ne "127.0.0.1:$([int]$info.adb_port)") {
    throw "ADB serial $Serial does not match VM $VmName."
}

Invoke-Adb -Arguments @("connect", $Serial) | Out-Null
Invoke-Adb -Arguments @("-s", $Serial, "wait-for-device") | Out-Null
Invoke-Adb -Arguments @("-s", $Serial, "root") | Out-Null
Start-Sleep -Milliseconds 1500
Invoke-Adb -Arguments @("connect", $Serial) | Out-Null
Invoke-Adb -Arguments @("-s", $Serial, "wait-for-device") | Out-Null
$identity = Invoke-Adb -Arguments @("-s", $Serial, "shell", "id")
if ($identity.Output -notmatch "uid=0\(root\)") {
    throw "The VM adbd is not running as root."
}

function Set-OfflineIsolation {
    param([Parameter(Mandatory)][bool]$Enabled)
    foreach ($family in @("iptables", "ip6tables")) {
        $jump = Invoke-Adb -Arguments @(
            "-s", $Serial, "shell", $family, "-C", "OUTPUT", "-j", "CR_NATIVE_OFFLINE"
        ) -AllowFailure
        if ($Enabled) {
            $exists = Invoke-Adb -Arguments @(
                "-s", $Serial, "shell", $family, "-S", "CR_NATIVE_OFFLINE"
            ) -AllowFailure
            if ($exists.ExitCode -eq 0) {
                Invoke-Adb -Arguments @("-s", $Serial, "shell", $family, "-F", "CR_NATIVE_OFFLINE") | Out-Null
            } else {
                Invoke-Adb -Arguments @("-s", $Serial, "shell", $family, "-N", "CR_NATIVE_OFFLINE") | Out-Null
            }
            if ($jump.ExitCode -eq 0) {
                Invoke-Adb -Arguments @(
                    "-s", $Serial, "shell", $family, "-D", "OUTPUT", "-j", "CR_NATIVE_OFFLINE"
                ) | Out-Null
            }
            Invoke-Adb -Arguments @(
                "-s", $Serial, "shell", $family, "-I", "OUTPUT", "1", "-j", "CR_NATIVE_OFFLINE"
            ) | Out-Null
            Invoke-Adb -Arguments @(
                "-s", $Serial, "shell", $family, "-A", "CR_NATIVE_OFFLINE",
                "-m", "conntrack", "--ctstate", "RELATED,ESTABLISHED", "-j", "RETURN"
            ) | Out-Null
            Invoke-Adb -Arguments @(
                "-s", $Serial, "shell", $family, "-A", "CR_NATIVE_OFFLINE", "-o", "lo", "-j", "RETURN"
            ) | Out-Null
            if ($family -eq "iptables") {
                Invoke-Adb -Arguments @(
                    "-s", $Serial, "shell", $family, "-A", "CR_NATIVE_OFFLINE",
                    "-d", "10.0.2.2/32", "-j", "RETURN"
                ) | Out-Null
                Invoke-Adb -Arguments @(
                    "-s", $Serial, "shell", $family, "-A", "CR_NATIVE_OFFLINE",
                    "-j", "REJECT", "--reject-with", "icmp-port-unreachable"
                ) | Out-Null
            } else {
                Invoke-Adb -Arguments @(
                    "-s", $Serial, "shell", $family, "-A", "CR_NATIVE_OFFLINE",
                    "-j", "REJECT", "--reject-with", "icmp6-port-unreachable"
                ) | Out-Null
            }
        } else {
            if ($jump.ExitCode -eq 0) {
                Invoke-Adb -Arguments @(
                    "-s", $Serial, "shell", $family, "-D", "OUTPUT", "-j", "CR_NATIVE_OFFLINE"
                ) | Out-Null
            }
            Invoke-Adb -Arguments @(
                "-s", $Serial, "shell", $family, "-F", "CR_NATIVE_OFFLINE"
            ) -AllowFailure | Out-Null
        }
    }
}

Set-OfflineIsolation -Enabled ([bool]$RestoreOffline)

Invoke-Adb -Arguments @("-s", $Serial, "shell", "svc", "power", "stayon", "true") | Out-Null
Invoke-Adb -Arguments @(
    "-s", $Serial, "forward", "tcp:$ControlPort", "tcp:$GuestControlPort"
) | Out-Null

if (-not $NoLaunch) {
    $oldPids = @()
    $pidResult = Invoke-Adb -Arguments @("-s", $Serial, "shell", "pidof", $Package) -AllowFailure
    if ($pidResult.ExitCode -eq 0 -and $pidResult.Output) {
        $oldPids = @($pidResult.Output -split "\s+" | Where-Object { $_ })
        Invoke-Adb -Arguments @("-s", $Serial, "shell", "am", "force-stop", $Package) | Out-Null
        Start-Sleep -Seconds 2
    }
    Invoke-Adb -Arguments @(
        "-s", $Serial, "shell", "am", "start", "-n", "$Package/$Activity"
    ) | Out-Null
    Start-Sleep -Seconds 6
    $newPid = Invoke-Adb -Arguments @("-s", $Serial, "shell", "pidof", $Package) -AllowFailure
    if ($newPid.ExitCode -ne 0 -or -not $newPid.Output) {
        throw "Package $Package did not start."
    }
}

[pscustomobject]@{
    ok            = $true
    instance      = $info.name
    serial        = $Serial
    offlineMode   = [bool]$RestoreOffline
    controlPort   = $ControlPort
    pid           = (Invoke-Adb -Arguments @("-s", $Serial, "shell", "pidof", $Package) -AllowFailure).Output
} | ConvertTo-Json
