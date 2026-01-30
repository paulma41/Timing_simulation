@echo off
setlocal

echo [setup_env] ===== CMD bootstrap =====

rem Root of this project (folder containing this script)
set "PROJECT_ROOT=%~dp0"
echo [setup_env] PROJECT_ROOT   = %PROJECT_ROOT%

rem Global location (outside OneDrive) where the real venv will live.
rem You can change this path if you prefer another location.
set "GLOBAL_ENV_ROOT=C:\envs\TimingSimulations"
echo [setup_env] GLOBAL_ENV_ROOT= %GLOBAL_ENV_ROOT%

if not exist "%GLOBAL_ENV_ROOT%" (
    echo [setup_env] GLOBAL_ENV_ROOT does not exist, creating it...
    mkdir "%GLOBAL_ENV_ROOT%"
) else (
    echo [setup_env] GLOBAL_ENV_ROOT exists.
)

set "TARGET_ENV=%GLOBAL_ENV_ROOT%\.venv"
set "LINK_ENV=%PROJECT_ROOT%\.venv"
echo [setup_env] TARGET_ENV     = %TARGET_ENV%
echo [setup_env] LINK_ENV       = %LINK_ENV%

rem If .venv exists, we don't check type here; just warn and exit to avoid overwriting.
if exist "%LINK_ENV%" (
    echo [setup_env] WARNING: .venv already exists in the project.
    echo [setup_env]          Delete/rename it manually if it's not a junction, then rerun setup_env.cmd.
    goto :EOF
)

if not exist "%TARGET_ENV%" (
    echo [setup_env] TARGET_ENV does not exist, creating venv...
    python -m venv "%TARGET_ENV%"
    if exist "%TARGET_ENV%" (
        echo [setup_env] TARGET_ENV created.
    ) else (
        echo [setup_env] WARNING: failed to create venv at %TARGET_ENV%.
    )
) else (
    echo [setup_env] TARGET_ENV already exists.
)

echo [setup_env] Creating junction .venv -> %TARGET_ENV%
mklink /J "%LINK_ENV%" "%TARGET_ENV%"

if exist "%LINK_ENV%" (
    echo [setup_env] Junction .venv created.
) else (
    echo [setup_env] WARNING: junction .venv was not created (check mklink permissions).
)

rem Check that the target exists
if not exist "%TARGET_ENV%" (
    echo [setup_env] WARNING: .venv points to %TARGET_ENV%, which does NOT exist.
) else (
    echo [setup_env] Verified: target env exists.
)

rem Activate the environment for this cmd session
if exist "%LINK_ENV%\Scripts\activate.bat" (
    echo [setup_env] Activating venv
    call "%LINK_ENV%\Scripts\activate.bat"
) else (
    echo [setup_env] WARNING: activate.bat not found in %LINK_ENV%\Scripts
)

echo [setup_env] ===== End bootstrap =====

endlocal
