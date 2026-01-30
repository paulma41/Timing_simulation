Param()

Write-Host "[setup_env] ===== PowerShell bootstrap ====="

# Root of this project (folder containing this script)
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "[setup_env] ProjectRoot      = $ProjectRoot"

# Global location (outside OneDrive) where the real venv will live.
# You can change this path if you prefer another location.
$GlobalEnvRoot = "C:\envs\TimingSimulations"
Write-Host "[setup_env] GlobalEnvRoot    = $GlobalEnvRoot"

if (-not (Test-Path $GlobalEnvRoot)) {
    Write-Host "[setup_env] GlobalEnvRoot does not exist, creating it..."
    New-Item -ItemType Directory -Path $GlobalEnvRoot | Out-Null
} else {
    Write-Host "[setup_env] GlobalEnvRoot exists."
}

$TargetEnv = Join-Path $GlobalEnvRoot ".venv"
$LinkEnv   = Join-Path $ProjectRoot ".venv"
Write-Host "[setup_env] TargetEnv (real) = $TargetEnv"
Write-Host "[setup_env] LinkEnv (in proj)= $LinkEnv"

# If .venv exists but is not a junction, warn and stop.
if (Test-Path $LinkEnv) {
    $item = Get-Item $LinkEnv -ErrorAction SilentlyContinue
    if ($item -and -not ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        Write-Warning "[setup_env] .venv exists in project and is NOT a junction (attributes: $($item.Attributes))."
        Write-Warning "[setup_env] Delete/rename it manually, then rerun setup_env.ps1."
        return
    } else {
        Write-Host "[setup_env] .venv already exists and looks like a junction."
    }
} else {
    Write-Host "[setup_env] .venv does not exist yet in the project."
}

# Create the real venv if needed
if (-not (Test-Path $TargetEnv)) {
    Write-Host "[setup_env] TargetEnv does not exist, creating venv..."
    python -m venv $TargetEnv
    if (Test-Path $TargetEnv) {
        Write-Host "[setup_env] TargetEnv created."
    } else {
        Write-Warning "[setup_env] Failed to create venv at $TargetEnv (python -m venv)."
    }
} else {
    Write-Host "[setup_env] TargetEnv already exists."
}

# Create the junction if needed
if (-not (Test-Path $LinkEnv)) {
    Write-Host "[setup_env] Creating junction .venv -> $TargetEnv"
    cmd /c "mklink /J `"$LinkEnv`" `"$TargetEnv`""
    if (Test-Path $LinkEnv) {
        $item = Get-Item $LinkEnv -ErrorAction SilentlyContinue
        Write-Host "[setup_env] .venv created. Attributes: $($item.Attributes)"
    } else {
        Write-Warning "[setup_env] Junction .venv was not created (check mklink permissions)."
    }
} else {
    Write-Host "[setup_env] Junction .venv already present (no recreation)."
}

# Check that the junction target exists
if (-not (Test-Path $TargetEnv)) {
    Write-Warning "[setup_env] WARNING: .venv points to $TargetEnv, which does NOT exist."
} else {
    Write-Host "[setup_env] Verified: target env exists."
}

# Activate venv in current session
$ActivateScript = Join-Path $LinkEnv "Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    Write-Host "[setup_env] Activating venv via $ActivateScript"
    . $ActivateScript
    Write-Host "[setup_env] Activation done."
} else {
    Write-Warning "[setup_env] Activate.ps1 not found in $LinkEnv\Scripts"
}

Write-Host "[setup_env] ===== End bootstrap ====="

