@echo off
setlocal
for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"

set "FILE=%~1"
if not defined FILE set "FILE=%REPO_ROOT%\nb\Rules.ipynb"

jupyter nbconvert --to python "%FILE%"
exit /b %ERRORLEVEL%
