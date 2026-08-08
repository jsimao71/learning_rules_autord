@echo off
setlocal
for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"

if not "%~1"=="" (
  for %%I in ("%~1") do set "AUTORD_DATASET_NOTEBOOK=%%~fI"
)

pushd "%REPO_ROOT%\nb"
jupyter notebook Eval.ipynb
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
