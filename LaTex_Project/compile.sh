#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build

# Prompt user to select the main file
echo "Select the LaTeX file to compile:"
echo "1) main_draft.tex"
echo "2) main_polimi_article.tex"
echo "3) main_polimi_classic.tex"
read -p "Enter your choice (1/2/3): " choice

# Set the main file based on user input
case $choice in
    1)
        main_file="main_draft"
        ;;
    2)
        main_file="main_polimi_article"
        ;;
    3)
        main_file="main_polimi_classic"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Create a symlink to the main .tex file in the build directory
ln -sf "../${main_file}.tex" build/

# Compile with pdflatex → biber → pdflatex → pdflatex
echo "Compiling $main_file.tex..."
pdflatex -aux-directory=build -output-directory=build "${main_file}.tex"
biber --input-directory=build --output-directory=build "${main_file}"
pdflatex -aux-directory=build -output-directory=build "${main_file}.tex"
pdflatex -aux-directory=build -output-directory=build "${main_file}.tex"

# Move the final PDF to the main directory
mv "build/${main_file}.pdf" .

echo "Compilation complete! Output: ${main_file}.pdf"
