# Staging Externalization Manifest

Generated: 2026-02-24 (local move after provenance audit snapshot)

## Audit Snapshot Reference
- Provenance snapshot docs were generated before this move:
  - `/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/docs/consolidation/PROVENANCE_MAP.md`
  - `/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/docs/consolidation/MIGRATION_DECISION_MATRIX.md`
- Raw inventory snapshot:
  - `/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/.local_artifacts/consolidation/20260224_221422/`

## Moved Paths
- `Claude/` -> `/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/.local_artifacts/staging/Claude`
- `Cleanup - Exporting functionality/` -> `/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/.local_artifacts/staging/Cleanup - Exporting functionality`

## Purpose
- Remove non-canonical staging bundles from the repository root.
- Keep all files locally accessible for provenance and recovery.
- Preserve a clean canonical implementation root for active HLT switching estimator development.

## Policy
- Files in `.local_artifacts/` are local-only and gitignored.
- If any item is later promoted to canonical project content, it must be reintroduced via the migration matrix (`adopt` / `merge` / `wrap`) with provenance recorded.
