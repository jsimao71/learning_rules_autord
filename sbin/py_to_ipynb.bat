@echo off
setlocal
for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"

set "PYFILE=%~1"
if not defined PYFILE set "PYFILE=%REPO_ROOT%\nb\Rules.py"
set "OUT=%~2"
if not defined OUT set "OUT=%REPO_ROOT%\nb\Rules.ipynb"

python -c "import pathlib,nbformat as nbf; source=pathlib.Path(r'%PYFILE%'); notebook=nbf.v4.new_notebook(); notebook.cells=[nbf.v4.new_markdown_cell('# Rules - generated from '+source.name),nbf.v4.new_code_cell(source.read_text(encoding='utf-8'))]; nbf.write(notebook,r'%OUT%')"
exit /b %ERRORLEVEL%
