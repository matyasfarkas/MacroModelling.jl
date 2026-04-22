# Paper Reproducibility Protocol (Updated US Data, HLT OBC)

This document is the replication protocol for the updated real-data application in:

- `/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/docs/paper/farkas_jmp_2026.tex`

The protocol is strict: if an output is not present, related claims must be marked `TBA, to be added`.

## 1. Environment

From repository root:

```bash
cd /Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## 2. Full End-to-End Pipeline

Primary command (requested submission depth):

```bash
cd /Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl
julia --project=. scripts/hlt_real_data_update_pipeline.jl \
  --run-dir=.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023
```

This creates/updates:

- `.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/dataset/`
- `.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/real_data/`
- `.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/manifests/`
- `.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/diagnostics/`
- `.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/tables/`

Important run settings in this protocol:

- Model: `Smets_Wouters_2007_HLT_obc`
- Data: `/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/test/data/usmodel_update.csv`
- Observables/order: `[:dy,:dc,:dinve,:labobs,:pinfobs,:dwobs,:robs]`
- Sample window: `47:290`
- Prefix for `s0`: `1:46`
- Estimation depth target: `2000` samples, `4` chains
- Switching settings: hard gate, inversion likelihood branches, bounded gate share `(0.01, 0.99)`
- Sampler settings: `--sampler=mh --mh-rw-cprobp=1e-6 --mh-rw-cindp=1e-6 --mh-rw-curvp=1e-4`

## 3. Restart Only the Estimation Step

If surrogate/payload/gate artifacts already exist, run only switching estimation:

```bash
cd /Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl
env JULIA_NUM_THREADS=4 julia --project=. scripts/hlt_sep_surrogate_synthetic_estimation.jl \
  .local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/dataset/hlt_sep_surrogate_trained.jls \
  .local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/real_data/hlt_real_data_payload.jls \
  --out=.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/real_data/hlt_sep_surrogate_estimation_chain.jls \
  --gate-calibration=.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/real_data/gate_calibration.jls \
  --samples=2000 --chains=4 --use-obc \
  --sampler=mh --mh-rw-cprobp=1e-6 --mh-rw-cindp=1e-6 --mh-rw-curvp=1e-4 \
  --shock-filter=inversion --linear-filter=inversion --post-mean-filter=inversion \
  --gate-mode=hard --gate-k-pre=0 --gate-k-post=0 --gate-min-len=1 \
  --gate-share-min=0.01 --gate-share-max=0.99 --fail-degenerate-gate=true \
  --checkpoint-path=.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/real_data/hlt_sep_surrogate_estimation_checkpoint.jls
```

Note: for `--sampler=mh`, the script forces serial multi-chain execution (no `MCMCThreads`) due a DynamicPPL thread-safety bug.

## 4. Extract Paper Tables from Chain Output

Use the chain payload (full run preferred; smoke chain only for plumbing checks):

```bash
cd /Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl
julia --project=. scripts/extract_hlt_real_data_tables.jl \
  .local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/real_data/hlt_sep_surrogate_estimation_chain.jls \
  --out-dir=.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/tables \
  --run-manifest=.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/manifests/run_manifest.toml \
  --generated-dir=docs/paper/generated
```

Expected generated files in `docs/paper/generated/`:

- `table_hlt_realdata_posterior.tex`
- `table_hlt_realdata_mcmc.tex`
- `table_hlt_realdata_gate.tex`
- `table_hlt_realdata_runmeta.tex`
- `hlt_real_data_summary.toml`
- `hlt_real_data_summary.json`

## 5. Compile Manuscript

```bash
cd /Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/docs/paper
./compile.sh
```

## 6. Verification Tests

```bash
cd /Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl
julia --project=. test/test_hlt_real_data_payload.jl
julia --project=. test/test_extract_hlt_real_data_tables.jl
julia --project=. test/test_hlt_acceptance_smoke.jl
```

## 7. Claim Discipline Rule

Only claims backed by run artifacts are allowed. If backing evidence is absent or incomplete, the manuscript must use:

- `TBA, to be added`

The current claim registry is:

- `/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/docs/paper/CLAIM_SUPPORT_MATRIX.md`
