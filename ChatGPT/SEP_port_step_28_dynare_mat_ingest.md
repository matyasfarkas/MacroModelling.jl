# SEP porting step 28: ingest Dynare RBCII `.mat` validation data

## Goal
Stage the Dynare RBCII SEP simulation outputs needed for validation, and add `.mat` reading support to MacroModelling.

## Inputs and provenance
Source repo (Dynare outputs):
- `/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/ep-mj-30-years-master/models/accuracy-sc`

Paper references:
- `tex/pub/adjemian-juillard-2025-september.tex`
- `models/accuracy-sc/main.m`
- `models/accuracy-sc/plot_euler_errors.m`
- `models/accuracy-sc/conditional_distribution_of_euler_errors.m`

Defaults applied:
- ZLB only (`ZLB=0.85`), no `.nozlb` files.
- Shock size `sigma=0.007`.
- Orders 0/1/2/5 (Dynare does not ship order 3 `.mat` files).
- `algo=1` plus `algo=0` for SEP(2) variants used in Dynare scripts.
- `hybrid=0` and `hybrid=4`.

## Changes made
1) Added MAT.jl dependency
- `Project.toml`: added `MAT = "23992714-dd62-5051-b70f-ba57cb901cac"`.
- `Manifest.toml`: updated to include MAT + dependencies.

2) Staged Dynare `.mat` files for validation
- Created `tests/sep_validation/sep_simulation_data/accuracy-sc/`.
- Copied the following files from the Dynare repo:
  - `rbcii-007-sep-0-algo-1-hybrid-0.mat`
  - `rbcii-007-sep-1-algo-1-hybrid-0.mat`
  - `rbcii-007-sep-2-algo-1-hybrid-0.mat`
  - `rbcii-007-sep-2-algo-0-hybrid-0.mat`
  - `rbcii-007-sep-5-algo-1-hybrid-0.mat`
  - `rbcii-007-sep-1-algo-1-hybrid-4.mat`
  - `rbcii-007-sep-2-algo-1-hybrid-4.mat`
  - `rbcii-007-sep-2-algo-0-hybrid-4.mat`
  - `rbcii-007-sep-5-algo-1-hybrid-4.mat`
  - `euler-007-sep-0-algo-1-hybrid-0.mat`
  - `euler-007-sep-1-algo-1-hybrid-0.mat`
  - `euler-007-sep-2-algo-1-hybrid-0.mat`
  - `euler-007-sep-2-algo-0-hybrid-0.mat`
  - `euler-007-sep-5-algo-1-hybrid-0.mat`
  - `euler-007-sep-1-algo-1-hybrid-4.mat`
  - `euler-007-sep-2-algo-1-hybrid-4.mat`
  - `euler-007-sep-2-algo-0-hybrid-4.mat`
  - `euler-007-sep-5-algo-1-hybrid-4.mat`

3) Documented provenance and constraints
- Added `tests/sep_validation/sep_simulation_data/README.md` with source paths, scope, and the missing order 3 note.

## Verification
- Loaded representative files with MAT.jl:
  - `rbcii-007-sep-0-algo-1-hybrid-0.mat`
  - `euler-007-sep-0-algo-1-hybrid-0.mat`
- Confirmed dseries structure with keys: `DATA__`, `NAMES__`, `FREQ__`, `INIT__`, `TEX__`, `OPS__`, `TAGS__`.

## Open items
- Order 3 `.mat` outputs are not present in the Dynare repo. If a direct order-3 Dynare comparison is required, we need to generate new Dynare runs or accept order 5 as the closest available reference.

## Next step
Implement the MAT-based importer to extract `Investment`, `efficiency`, and the implied shock sequence from the staged `.mat` files, then wire this into the RBCII SEP validation script and plotting pipeline.
