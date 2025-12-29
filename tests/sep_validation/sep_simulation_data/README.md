# SEP simulation validation data (Dynare)

This folder contains Dynare output `.mat` files copied from:
`/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/ep-mj-30-years-master/models/accuracy-sc`.

Scope (defaults we agreed on):
- Model: RBCII with investment lower bound (`ZLB=0.85`).
- Shock size: `sigma=0.007`.
- Orders: 0, 1, 2, and 5 (Dynare does not ship order 3 `.mat` files).
- Algorithms: `algo=1` plus `algo=0` for SEP(2) variants used in Dynare scripts.
- Hybrid correction: `hybrid=0` and `hybrid=4`.

Files are grouped under `accuracy-sc/` and include:
- `rbcii-007-...` simulation series (used for investment path comparisons).
- `euler-007-...` Euler error series (used for accuracy figures).

Reference scripts in the source repo:
- `models/accuracy-sc/main.m`
- `models/accuracy-sc/plot_euler_errors.m`
- `models/accuracy-sc/conditional_distribution_of_euler_errors.m`
- Paper: `tex/pub/adjemian-juillard-2025-september.tex`.

Note: If order 3 comparisons are required, Dynare does not provide `.mat` files for that order in the source repo. We can generate order 3 runs separately if needed.
