@echo off
:: Create build directory if it doesn't exist
if not exist "build" mkdir build

:: Create a symlink to the main .tex file in the build directory
mklink "build\main.tex" "main.tex" >nul 2>&1
if errorlevel 1 (
    :: If symlink fails (e.g., already exists), just copy the file
    copy "main.tex" "build\main.tex" >nul
)

:: Compile with pdflatex → biber → pdflatex → pdflatex
echo Compiling main.tex...
pdflatex -aux-directory=build -output-directory=build "main.tex"
biber --input-directory=build --output-directory=build "main"
pdflatex -aux-directory=build -output-directory=build "main.tex"
pdflatex -aux-directory=build -output-directory=build "main.tex"

:: Move the final PDF to the main directory
move /Y "build\main.pdf" >nul

echo Compilation complete! Output: main.pdf
