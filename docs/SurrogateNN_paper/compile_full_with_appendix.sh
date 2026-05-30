#!/bin/bash
# Compile SurrogateNN_paper.tex with the legacy in-file appendices enabled.

set -e

echo "==================================================================="
echo "LaTeX Compilation Script for Full Archive Paper"
echo "==================================================================="

cd "$(dirname "$0")"

JOB=SurrogateNN_paper_full
SOURCE="\\def\\WITHPAPERAPPENDIX{1}\\input{SurrogateNN_paper.tex}"

echo ""
echo "Step 1: First pdflatex pass..."
pdflatex -jobname="$JOB" -interaction=nonstopmode "$SOURCE"

echo ""
echo "Step 2: BibTeX compilation..."
bibtex "$JOB"

echo ""
echo "Step 3: Second pdflatex pass (resolve references)..."
pdflatex -jobname="$JOB" -interaction=nonstopmode "$SOURCE"

echo ""
echo "Step 4: Third pdflatex pass (finalize)..."
pdflatex -jobname="$JOB" -interaction=nonstopmode "$SOURCE"

echo ""
echo "Step 5: Fourth pdflatex pass (settle long-appendix references)..."
pdflatex -jobname="$JOB" -interaction=nonstopmode "$SOURCE"

echo ""
echo "==================================================================="
echo "Full archive compilation complete!"
echo "Output: ${JOB}.pdf"
echo "==================================================================="

if [ -f "${JOB}.pdf" ]; then
    pdfinfo "${JOB}.pdf" | grep "Pages:"
else
    echo "ERROR: PDF was not generated. Check ${JOB}.log"
    exit 1
fi
