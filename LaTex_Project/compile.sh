#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build

# Create a symlink to the main .tex file in the build directory
ln -sf "../main.tex" build/

# Compile with pdflatex → biber → pdflatex → pdflatex
echo "Compiling main.tex..."
pdflatex -aux-directory=build -output-directory=build "main.tex"
biber --input-directory=build --output-directory=build "main"
pdflatex -aux-directory=build -output-directory=build "main.tex"
pdflatex -aux-directory=build -output-directory=build "main.tex"

# Move the final PDF to the main directory
mv "build/main.pdf" .

echo "Compilation complete! Output: main.pdf"
