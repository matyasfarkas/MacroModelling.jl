#!/bin/bash
# Compilation script for SurrogateNN_technical_appendix.tex

set -e

echo "==================================================================="
echo "LaTeX Compilation Script for Technical Appendix"
echo "==================================================================="

cd "$(dirname "$0")"

echo ""
echo "Step 1: First pdflatex pass..."
pdflatex -interaction=nonstopmode SurrogateNN_technical_appendix.tex

echo ""
echo "Step 2: BibTeX compilation..."
bibtex SurrogateNN_technical_appendix

echo ""
echo "Step 3: Second pdflatex pass..."
pdflatex -interaction=nonstopmode SurrogateNN_technical_appendix.tex

echo ""
echo "Step 4: Third pdflatex pass..."
pdflatex -interaction=nonstopmode SurrogateNN_technical_appendix.tex

echo ""
echo "==================================================================="
echo "Compilation complete."
echo "Output: SurrogateNN_technical_appendix.pdf"
echo "==================================================================="

if [ -f SurrogateNN_technical_appendix.pdf ]; then
    pdfinfo SurrogateNN_technical_appendix.pdf | grep "Pages:" || true
else
    echo "ERROR: PDF was not generated. Check SurrogateNN_technical_appendix.log."
    exit 1
fi
