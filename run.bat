@echo off
cd /d "%~dp0"
rem Keep everything (Python + packages) inside this folder so deleting the folder removes it all.
set "UV_CACHE_DIR=%~dp0.uv\cache"
set "UV_PYTHON_INSTALL_DIR=%~dp0.uv\python"
set "UVBIN=uv"
if exist "%~dp0uv.exe" set "UVBIN=%~dp0uv.exe"
"%UVBIN%" run app.py
pause
