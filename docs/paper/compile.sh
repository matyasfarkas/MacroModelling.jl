#!/bin/bash
# Compilation script for farkas_jmp_2026.tex

echo "==================================================================="
echo "LaTeX Compilation Script for Job Market Paper"
echo "==================================================================="

# Change to paper directory
cd "$(dirname "$0")"

echo ""
echo "Step 1: First pdflatex pass..."
pdflatex -interaction=nonstopmode farkas_jmp_2026.tex

echo ""
echo "Step 2: BibTeX compilation..."
bibtex farkas_jmp_2026

echo ""
echo "Step 3: Second pdflatex pass (resolve references)..."
pdflatex -interaction=nonstopmode farkas_jmp_2026.tex

echo ""
echo "Step 4: Third pdflatex pass (finalize)..."
pdflatex -interaction=nonstopmode farkas_jmp_2026.tex

echo ""
echo "==================================================================="
echo "Compilation complete!"
echo "Output: farkas_jmp_2026.pdf"
echo "==================================================================="
echo ""
echo "Auxiliary files generated:"
ls -lh farkas_jmp_2026.aux farkas_jmp_2026.log farkas_jmp_2026.bbl 2>/dev/null || echo "Some files not found (expected on first run)"

echo ""
echo "To view the PDF:"
echo "  open farkas_jmp_2026.pdf    (macOS)"
echo "  xdg-open farkas_jmp_2026.pdf    (Linux)"

# Check for errors
if [ -f farkas_jmp_2026.pdf ]; then
    echo ""
    echo "✓ PDF successfully generated"
    pdfinfo farkas_jmp_2026.pdf | grep "Pages:"
else
    echo ""
    echo "✗ ERROR: PDF was not generated. Check the log file:"
    echo "  farkas_jmp_2026.log"
fi
