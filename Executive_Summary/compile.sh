#!/bin/bash
set -e

# Create build directory if it doesn't exist
mkdir -p build

# Clean all previous build files
rm -f build/*

# Symlink main files into build
ln -sf "../executive_summary.tex" build/
ln -sf "../main.tex" build/

echo "Compiling executive_summary.tex..."
# First LaTeX pass
pdflatex -output-directory=build "executive_summary.tex"

# Glossaries (if used)
makeglossaries --input-directory=build "executive_summary" || true

# Bibliography: prefer biber (biblatex) when a .bcf is generated, otherwise fall back to bibtex (natbib)
if [ -f build/executive_summary.bcf ]; then
	if command -v biber >/dev/null 2>&1; then
		echo "Found .bcf: running biber..."
		biber --input-directory=build "executive_summary"
	else
		echo "Error: .bcf found but 'biber' is not installed. Install biber or switch to bibtex workflow." >&2
		exit 1
	fi
else
	# No .bcf: likely using natbib/bibtex. Try bibtex if available.
	if command -v bibtex >/dev/null 2>&1; then
		echo "No .bcf found: running bibtex..."
		bibtex build/executive_summary
	else
		echo "Warning: neither .bcf nor bibtex/biber found; bibliography may be missing."
	fi
fi

# Final LaTeX passes
pdflatex -output-directory=build "executive_summary.tex"
pdflatex -output-directory=build "executive_summary.tex"

# Move the final PDF to the main directory
mv "build/executive_summary.pdf" .

# Remove symlink
rm build/executive_summary.tex

echo "Compilation complete! Output: executive_summary.pdf"
