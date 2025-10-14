@echo off
:: Create build directory if it doesn't exist
if not exist "build" mkdir build

:: Prompt user to select the main file
echo Select the LaTeX file to compile:
echo 1) main_draft.tex
echo 2) main_polimi_article.tex
echo 3) main_polimi_classic.tex
set /p choice="Enter your choice (1/2/3): "

:: Set the main file based on user input
if "%choice%"=="1" (
    set main_file=main_draft
) else if "%choice%"=="2" (
    set main_file=main_polimi_article
) else if "%choice%"=="3" (
    set main_file=main_polimi_classic
) else (
    echo Invalid choice. Exiting.
    exit /b 1
)

:: Create a symlink to the main .tex file in the build directory
mklink "build\%main_file%.tex" "%main_file%.tex" >nul 2>&1
if errorlevel 1 (
    :: If symlink fails (e.g., already exists), just copy the file
    copy "%main_file%.tex" "build\%main_file%.tex" >nul
)

:: Compile with pdflatex → biber → pdflatex → pdflatex
echo Compiling %main_file%.tex...
pdflatex -aux-directory=build -output-directory=build "%main_file%.tex"
echo 1/4
biber --input-directory=build --output-directory=build "%main_file%"
echo 2/4
pdflatex -aux-directory=build -output-directory=build "%main_file%.tex"
echo 3/4
pdflatex -aux-directory=build -output-directory=build "%main_file%.tex"
echo 4/4

:: Move the final PDF to the main directory
move /Y "build\%main_file%.pdf" >nul

echo Compilation complete! Output: %main_file%.pdf
