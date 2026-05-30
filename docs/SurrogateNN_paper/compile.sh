#!/bin/bash
# Compilation script for SurrogateNN_paper.tex

echo "==================================================================="
echo "LaTeX Compilation Script for Job Market Paper"
echo "Default build omits the legacy in-file appendices."
echo "Use compile_full_with_appendix.sh for the archive build."
echo "==================================================================="

# Change to paper directory
cd "$(dirname "$0")"

echo ""
echo "Step 1: First pdflatex pass..."
pdflatex -interaction=nonstopmode SurrogateNN_paper.tex

echo ""
echo "Step 2: BibTeX compilation..."
bibtex SurrogateNN_paper

echo ""
echo "Step 3: Second pdflatex pass (resolve references)..."
pdflatex -interaction=nonstopmode SurrogateNN_paper.tex

echo ""
echo "Step 4: Third pdflatex pass (finalize)..."
pdflatex -interaction=nonstopmode SurrogateNN_paper.tex

echo ""
echo "==================================================================="
echo "Compilation complete!"
echo "Output: SurrogateNN_paper.pdf"
echo "==================================================================="
echo ""
echo "Auxiliary files generated:"
ls -lh SurrogateNN_paper.aux SurrogateNN_paper.log SurrogateNN_paper.bbl 2>/dev/null || echo "Some files not found (expected on first run)"

echo ""
echo "To view the PDF:"
echo "  open SurrogateNN_paper.pdf    (macOS)"
echo "  xdg-open SurrogateNN_paper.pdf    (Linux)"

# Check for errors
if [ -f SurrogateNN_paper.pdf ]; then
    echo ""
    echo "✓ PDF successfully generated"
    pdfinfo SurrogateNN_paper.pdf | grep "Pages:"
else
    echo ""
    echo "✗ ERROR: PDF was not generated. Check the log file:"
    echo "  SurrogateNN_paper.log"
fi
