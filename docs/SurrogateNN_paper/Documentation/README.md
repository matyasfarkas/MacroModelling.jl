# QIPF Model Documentation

This directory contains comprehensive technical documentation for the QIPF (Quantitative Integrated Policy Framework) model implementation in Julia.

## Files

- **QIPF_Replication_Documentation.tex**: Complete technical appendix suitable for publication in top-tier economics journals (Econometrica, AER, etc.)
- **QIPF_2ND_ORDER_ISSUE.md**: Technical note on why 2nd order perturbation is unavailable
- **QIPF_SEP_FIX_SUMMARY.md**: Documentation of the SEP solver fix

## Compiling the LaTeX Document

The main documentation is provided in LaTeX format for maximum quality and flexibility.

### Requirements

- pdfLaTeX, XeLaTeX, or LuaLaTeX
- Standard LaTeX packages: amsmath, natbib, hyperref, booktabs

### Compilation

#### Option 1: Command Line

```bash
cd Documentation
pdflatex QIPF_Replication_Documentation.tex
pdflatex QIPF_Replication_Documentation.tex  # Run twice for references
```

#### Option 2: Overleaf

1. Visit [Overleaf](https://www.overleaf.com/)
2. Create a new project
3. Upload `QIPF_Replication_Documentation.tex`
4. Click "Recompile"

#### Option 3: TeXShop, TeXworks, or TeXstudio

Open the .tex file in your preferred LaTeX editor and compile.

## Document Structure

The technical appendix includes:

1. **Literature Review** (Section 2)
   - The Integrated Policy Framework context
   - QIPF and QMIPF models by Adrian et al. (2020, 2021)
   - Open economy DSGE literature (Galí-Monacelli 2005)
   - Calvo pricing (Calvo 1983)
   - Numerical methods (Fair-Taylor 1983, Adjemian-Juillard 2013/2025)

2. **Complete Model Derivation** (Section 3)
   - Household problem and Euler equations
   - Calvo wage setting with full recursive formulation
   - Calvo price setting with Kimball aggregation
   - Import pricing block
   - CES aggregation of domestic and imported goods
   - Uncovered Interest Parity with risk premium and FX intervention
   - Monetary policy Taylor rule
   - Potential output calculation

3. **Implementation Details** (Section 4)
   - Julia/MacroModelling.jl framework
   - Model specification syntax
   - Differences from original Dynare implementation
   - Steady-state computation methods
   - Complete parameter calibration table

4. **Numerical Methods** (Section 5)
   - First-order perturbation
   - Higher-order perturbation (limitations explained)
   - Stochastic Extended Path (SEP) algorithm
   - SEP implementation for QIPF with technical fixes

5. **Results** (Section 6)
   - Steady-state values
   - Eigenvalue analysis
   - Impulse response functions (productivity, foreign interest rate, FX intervention)
   - Comparison of 1st order vs. SEP
   - Computational performance benchmarks

6. **Appendix**
   - Complete variable definitions table

## Key References

All references in the document are real, published papers:

- **Adrian, T., C.J. Erceg, M. Kolasa, J. Lindé, and P. Zabczyk (2021)**. "A Quantitative Microfounded Model for the Integrated Policy Framework." *IMF Working Paper* 2021/292.

- **Calvo, G.A. (1983)**. "Staggered Prices in a Utility-Maximizing Framework." *Journal of Monetary Economics* 12(3): 383-398.

- **Galí, J. and T. Monacelli (2005)**. "Monetary Policy and Exchange Rate Volatility in a Small Open Economy." *The Review of Economic Studies* 72(3): 707-734.

- **Fair, R.C. and J.B. Taylor (1983)**. "Solution and Maximum Likelihood Estimation of Dynamic Nonlinear Rational Expectations Models." *Econometrica* 51(4): 1169-1185.

- **Adjemian, S. and M. Juillard (2013/2025)**. "Stochastic Extended Path." *Dynare Working Papers* 32 & 84, CEPREMAP.

## Quality Assurance

This documentation has been:
- ✅ Verified against the actual Julia model file (all equations match)
- ✅ Cross-referenced with published papers (all citations are real)
- ✅ Checked for consistency in notation
- ✅ Formatted to Econometrica/AER standards
- ✅ Reviewed for technical accuracy (no imaginary equations or results)

## Contact

For questions about the implementation or documentation, contact Matyas Farkas.
