# claudelocal guard - launches llama-server inside a Windows Job Object and supervises it so that
# a 27B model at 262144 tokens of context can never take the desktop down with it.
#
# Four independent brakes, weakest-to-strongest:
#
#   1. mmap (set by the caller via --load-mode mmap). Model weights are clean file-backed pages, so
#      Windows can drop them under pressure and re-read them from disk. They are never written to
#      the pagefile. This is why 15 GB of weights cannot cause a stall on its own.
#   2. Below-normal priority + a thread count that leaves cores free. The desktop always wins a
#      scheduling fight, so the machine stays interactive even while the model is saturating RAM.
#   3. A Job Object commit cap. The KV cache and llama.cpp's host buffers ARE committed memory -
#      the only part of the footprint that can reach the pagefile. The job limit makes an overrun
#      fail the allocation inside llama-server instead of pushing the rest of Windows to disk.
#   4. A watchdog on available physical memory. It trims the server's working set first, and if
#      memory keeps falling it stops the server. A dead model beats a frozen machine.
#
# Pure ASCII, no BOM, on purpose - several AV engines on this machine hard-lock BOM'd .ps1 files.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]   $Exe,
    # PowerShell's -File flattens an array parameter into one comma-joined string, so the server
    # command line arrives as a JSON array in a file instead. It is also the easiest thing to read
    # back when a load goes wrong.
    [string]   $ArgsFile,
    [string[]] $ServerArgs = @(),
    [string] $LogFile,
    [string] $StatusFile,
    [int]    $CommitCapMB  = 12288,   # Job Object commit ceiling for llama-server
    [int]    $SoftFloorMB  = 2048,    # below this, trim the server's working set
    [int]    $HardFloorMB  = 1024,    # sustained below this, stop the server
    [int]    $HardStrikes  = 4,       # consecutive hard-floor samples before stopping
    [int]    $PollSeconds  = 2
)

$ErrorActionPreference = 'Stop'

if ($ArgsFile) {
    if (-not (Test-Path $ArgsFile)) { throw "guard: args file not found: $ArgsFile" }
    # ConvertFrom-Json emits a JSON array as ONE object rather than streaming its elements, so
    # wrapping the pipeline in @() yields a 1-element array holding the array. Assign first, then
    # cast - otherwise the whole command line reaches llama-server as a single argument.
    $parsed = Get-Content $ArgsFile -Raw | ConvertFrom-Json
    $ServerArgs = [string[]]$parsed
}
if (-not $ServerArgs -or $ServerArgs.Count -eq 0) { throw 'guard: no server arguments supplied (use -ArgsFile or -ServerArgs)' }

if (-not $LogFile)    { $LogFile    = Join-Path $PSScriptRoot '..\logs\server.log' }
if (-not $StatusFile) { $StatusFile = Join-Path $PSScriptRoot '..\logs\guard.json' }
$logDir = Split-Path -Parent $LogFile
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

# Windows PowerShell 5.1's `-Encoding utf8` writes a BOM, and JSON.parse throws on a leading
# U+FEFF - which silently blinded every Node-side reader of guard.json, including the one that is
# supposed to notice llama-server dying during a load. Write UTF-8 without a BOM instead.
$script:Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
function Write-Json([string]$path, $obj) {
    try { [System.IO.File]::WriteAllText($path, ($obj | ConvertTo-Json -Compress), $script:Utf8NoBom) } catch { }
}

$guardLog = Join-Path $logDir 'guard.log'
function Say([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $line
    try { [System.IO.File]::AppendAllText($guardLog, $line + [Environment]::NewLine, $script:Utf8NoBom) } catch { }
}

# ---------------------------------------------------------------------------------------------
# Win32 surface. Job Objects are the only way on Windows to put a hard ceiling on how much a
# process may commit; nothing in .NET exposes them.
# ---------------------------------------------------------------------------------------------
if (-not ('ClaudeLocal.Native' -as [type])) {
Add-Type -Language CSharp -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

namespace ClaudeLocal {

[StructLayout(LayoutKind.Sequential)]
public struct IO_COUNTERS {
    public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
    public ulong ReadTransferCount,  WriteTransferCount,  OtherTransferCount;
}

[StructLayout(LayoutKind.Sequential)]
public struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
    public long   PerProcessUserTimeLimit;
    public long   PerJobUserTimeLimit;
    public uint   LimitFlags;
    public UIntPtr MinimumWorkingSetSize;
    public UIntPtr MaximumWorkingSetSize;
    public uint   ActiveProcessLimit;
    public UIntPtr Affinity;
    public uint   PriorityClass;
    public uint   SchedulingClass;
}

[StructLayout(LayoutKind.Sequential)]
public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
    public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
    public IO_COUNTERS IoInfo;
    public UIntPtr ProcessMemoryLimit;
    public UIntPtr JobMemoryLimit;
    public UIntPtr PeakProcessMemoryUsed;
    public UIntPtr PeakJobMemoryUsed;
}

[StructLayout(LayoutKind.Sequential)]
public class MEMORYSTATUSEX {
    public uint  dwLength;
    public uint  dwMemoryLoad;
    public ulong ullTotalPhys, ullAvailPhys;
    public ulong ullTotalPageFile, ullAvailPageFile;
    public ulong ullTotalVirtual, ullAvailVirtual, ullAvailExtendedVirtual;
    public MEMORYSTATUSEX() { dwLength = (uint)Marshal.SizeOf(typeof(MEMORYSTATUSEX)); }
}

public static class Native {
    public const uint LIMIT_WORKINGSET        = 0x00000001;
    public const uint LIMIT_PRIORITY_CLASS    = 0x00000020;
    public const uint LIMIT_JOB_MEMORY        = 0x00000200;
    public const uint LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
    public const uint BELOW_NORMAL_PRIORITY_CLASS = 0x00004000;

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr CreateJobObject(IntPtr sa, string name);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint len);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool CloseHandle(IntPtr h);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool GlobalMemoryStatusEx([In, Out] MEMORYSTATUSEX buffer);

    [DllImport("psapi.dll", SetLastError = true)]
    public static extern bool EmptyWorkingSet(IntPtr process);

    // Creates the job, applies a commit ceiling + below-normal priority, and ties the job's
    // lifetime to this handle so killing the guard can never leave an orphaned llama-server.
    public static IntPtr CreateCappedJob(ulong commitCapBytes) {
        IntPtr job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) throw new Exception("CreateJobObject failed: " + Marshal.GetLastWin32Error());

        var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        info.BasicLimitInformation.LimitFlags =
            LIMIT_JOB_MEMORY | LIMIT_PRIORITY_CLASS | LIMIT_KILL_ON_JOB_CLOSE;
        info.BasicLimitInformation.PriorityClass = BELOW_NORMAL_PRIORITY_CLASS;
        info.JobMemoryLimit = new UIntPtr(commitCapBytes);

        int len = Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION));
        IntPtr p = Marshal.AllocHGlobal(len);
        try {
            Marshal.StructureToPtr(info, p, false);
            if (!SetInformationJobObject(job, 9 /* ExtendedLimitInformation */, p, (uint)len))
                throw new Exception("SetInformationJobObject failed: " + Marshal.GetLastWin32Error());
        } finally { Marshal.FreeHGlobal(p); }
        return job;
    }

    public static ulong AvailablePhysBytes() {
        var m = new MEMORYSTATUSEX();
        if (!GlobalMemoryStatusEx(m)) return 0;
        return m.ullAvailPhys;
    }

    public static ulong TotalPhysBytes() {
        var m = new MEMORYSTATUSEX();
        if (!GlobalMemoryStatusEx(m)) return 0;
        return m.ullTotalPhys;
    }
}
}
'@
}

# ---------------------------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------------------------
$MB = 1MB
$job = [ClaudeLocal.Native]::CreateCappedJob([uint64]($CommitCapMB * $MB))
Say ("guard up: commit cap ${CommitCapMB} MB, soft floor ${SoftFloorMB} MB, hard floor ${HardFloorMB} MB")
Say ("exec: $Exe " + ($ServerArgs -join ' '))

# ProcessStartInfo.ArgumentList is .NET Core only - Windows PowerShell 5.1 runs on .NET Framework,
# so the command line has to be quoted by hand to CommandLineToArgvW's rules.
function Quote-Arg([string]$a) {
    if ($a -notmatch '[\s"]') { return $a }
    $s = $a -replace '(\\*)"', '$1$1\"'
    $s = $s -replace '(\\+)$', '$1$1'
    return '"' + $s + '"'
}

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName               = $Exe
$psi.WorkingDirectory       = (Split-Path -Parent $Exe)
$psi.UseShellExecute        = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$psi.CreateNoWindow         = $true
$psi.Arguments              = (($ServerArgs | ForEach-Object { Quote-Arg $_ }) -join ' ')

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
$proc.EnableRaisingEvents = $true

# llama-server writes its progress to stderr; both streams are drained asynchronously so a full
# pipe buffer can never stall the model mid-load.
$sw = [System.IO.StreamWriter]::new($LogFile, $false, $script:Utf8NoBom)
$sw.AutoFlush = $true
$sink = { if ($null -ne $EventArgs.Data) { $sw.WriteLine($EventArgs.Data) } }
Register-ObjectEvent -InputObject $proc -EventName OutputDataReceived -Action $sink | Out-Null
Register-ObjectEvent -InputObject $proc -EventName ErrorDataReceived  -Action $sink | Out-Null

[void]$proc.Start()
$proc.BeginOutputReadLine()
$proc.BeginErrorReadLine()

# Assign before the model starts mapping anything substantial, so the cap covers the whole load.
if (-not [ClaudeLocal.Native]::AssignProcessToJobObject($job, $proc.Handle)) {
    Say ("WARNING AssignProcessToJobObject failed: " + [Runtime.InteropServices.Marshal]::GetLastWin32Error() + " - continuing without the commit cap")
} else {
    Say ("llama-server pid $($proc.Id) assigned to capped job")
}
try { $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal } catch { }

# ---------------------------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------------------------
$strikes  = 0
$trims    = 0
# The planner's committed-memory estimate is what sizes this cap, so recording the high-water mark
# is the only way to know whether that estimate is any good. `claudelocal status` prints both.
$peakCommit = 0
$peakWS     = 0
$total   = [ClaudeLocal.Native]::TotalPhysBytes()
$minSeen = [ClaudeLocal.Native]::AvailablePhysBytes()
$stopReason = 'exited'

while (-not $proc.HasExited) {
    Start-Sleep -Seconds $PollSeconds
    if ($proc.HasExited) { break }

    $avail = [ClaudeLocal.Native]::AvailablePhysBytes()
    if ($avail -lt $minSeen) { $minSeen = $avail }
    $availMB = [int]($avail / $MB)

    if ($availMB -lt $HardFloorMB) {
        $strikes++
        Say ("MEMORY available ${availMB} MB below hard floor ${HardFloorMB} MB (strike $strikes/$HardStrikes)")
        if ($strikes -ge $HardStrikes) {
            Say 'MEMORY sustained exhaustion - stopping llama-server to keep the desktop alive'
            $stopReason = 'oom-guard'
            try { $proc.Kill() } catch { }
            break
        }
    }
    elseif ($availMB -lt $SoftFloorMB) {
        $strikes = 0
        $trims++
        # Ask Windows to hand back this process's trimmable pages. The mmap'd weights are clean,
        # so this is nearly free and buys headroom without touching the pagefile.
        try { [void][ClaudeLocal.Native]::EmptyWorkingSet($proc.Handle) } catch { }
        Say ("MEMORY available ${availMB} MB below soft floor ${SoftFloorMB} MB - trimmed working set (trim #$trims)")
    }
    else { $strikes = 0 }

    # Process performance counters are snapshotted when the object is created and are only
    # re-read on Refresh(). Without this the guard reported llama-server's start-up working set
    # (a few hundred MB) for the whole run, so the commit cap could never be seen approaching.
    try { $proc.Refresh() } catch { }
    $commitMB = [int]($proc.PrivateMemorySize64 / $MB)
    $wsMB     = [int]($proc.WorkingSet64 / $MB)
    if ($commitMB -gt $peakCommit) { $peakCommit = $commitMB }
    if ($wsMB     -gt $peakWS)     { $peakWS     = $wsMB }

    $status = [ordered]@{
        pid          = $proc.Id
        running      = $true
        availMB      = $availMB
        minAvailMB   = [int]($minSeen / $MB)
        totalMB      = [int]($total / $MB)
        workingSetMB = $wsMB
        peakWSMB     = $peakWS
        commitMB     = $commitMB
        peakCommitMB = $peakCommit
        commitCapMB  = $CommitCapMB
        trims        = $trims
        strikes      = $strikes
        updated      = (Get-Date).ToString('o')
    }
    Write-Json $StatusFile $status
}

$code = 0
try { $code = $proc.ExitCode } catch { }
Say ("llama-server exited (code $code, reason $stopReason); lowest available memory seen was " + [int]($minSeen / $MB) + " MB")
Write-Json $StatusFile ([ordered]@{ pid = $proc.Id; running = $false; exitCode = $code; reason = $stopReason; minAvailMB = [int]($minSeen / $MB); peakCommitMB = $peakCommit; peakWSMB = $peakWS; commitCapMB = $CommitCapMB; updated = (Get-Date).ToString('o') })
try { $sw.Dispose() } catch { }
# Closing the job handle kills anything still inside it (KILL_ON_JOB_CLOSE): no orphans.
[void][ClaudeLocal.Native]::CloseHandle($job)
exit $code
