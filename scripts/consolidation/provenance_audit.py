#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

AUDIT_EXTS = {".jl", ".md", ".toml", ".jls", ".mat", ".pdf", ".log", ".zip"}
TEXT_HASH_EXTS = {".jl", ".md", ".toml"}
SMALL_BINARY_HASH_LIMIT = 10 * 1024 * 1024
KEYWORD_RE = re.compile(r"(hlt|surrogate|regime|sep)", re.IGNORECASE)


@dataclass
class RootSpec:
    source_repo: str
    root: Path
    label: str


def sha256_file(path: Path, force: bool = False) -> str:
    ext = path.suffix.lower()
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    if ext not in TEXT_HASH_EXTS and not force and size > SMALL_BINARY_HASH_LIMIT:
        return ""
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # Reduce noise and scan time.
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {".git", "node_modules", ".venv", "venv", "__pycache__", ".pixi"}
        ]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in AUDIT_EXTS:
                yield p


def classify(path: Path, source_repo: str) -> str:
    p = str(path)
    if source_repo == "SurrogateNN_Estimation.jl":
        if "/src/" in p:
            return "active_source"
        if "/scripts/" in p:
            return "active_script"
        if "/models/" in p:
            return "active_model"
        if "/test/" in p:
            return "active_test"
        if "/tests/" in p:
            return "legacy_tests_or_fixtures"
        if "/docs/regime_switching_estimation/" in p:
            return "active_regime_docs"
        if "/archive/research/regime_switching/" in p:
            return "historical_regime_docs"
        if "/Claude/" in p:
            return "staging_notes"
        if "/Cleanup - Exporting functionality/" in p:
            return "staging_export_bundle"
        if "/archive/" in p:
            return "local_archive"
        if "/docs/" in p:
            return "active_docs"
        return "other_target_repo"

    # Non_Linear_DSGE external repo
    if "/SW07_development/" in p:
        return "external_hlt_sep_dev"
    if "/clean_validation_package/" in p:
        return "external_validation_package"
    if "/development code/" in p:
        return "external_general_dev"
    return "external_other"


def candidate_target(path: Path, source_repo: str, classification: str, repo_relative_path: str | None = None) -> str:
    p = str(path)
    name = path.name
    if source_repo == "SurrogateNN_Estimation.jl":
        if classification in {"active_source", "active_script", "active_model", "active_test", "active_docs", "active_regime_docs"}:
            return repo_relative_path or name
        if classification in {"staging_notes", "staging_export_bundle"}:
            return f".local_artifacts/staging/{path.parent.name}/{name}"
        if classification == "legacy_tests_or_fixtures":
            if repo_relative_path and repo_relative_path.startswith("tests/"):
                return "test/fixtures/" + repo_relative_path[len("tests/"):]
            return "test/fixtures/"
        if classification == "historical_regime_docs":
            return "archive/research/regime_switching/..."
        return "(no move)"

    # External repo heuristics (selective migration targets)
    if classification == "external_hlt_sep_dev":
        if name.startswith("Smets_Wouters_2007_HLT") and path.suffix == ".jl":
            return f"models/{name}"
        if "HLT" in name and path.suffix == ".jl":
            return f"scripts/{name}"
        if path.suffix == ".md":
            return f"archive/research/regime_switching/historical_import/{name}"
        if path.suffix in {".log", ".pdf", ".zip", ".mat"}:
            return f".local_artifacts/external_imports/SW07_development/{name}"
        return f"archive/research/regime_switching/historical_import/{name}"
    if classification == "external_validation_package":
        if path.suffix == ".jl":
            return f"archive/research/regime_switching/historical_import/{name}"
        return f".local_artifacts/external_imports/clean_validation_package/{name}"
    if classification == "external_general_dev":
        if path.suffix == ".jl" and KEYWORD_RE.search(name):
            return f"archive/research/regime_switching/historical_import/{name}"
        return f".local_artifacts/external_imports/development_code/{name}"
    return "(review)"


def action_for(path: Path, source_repo: str, classification: str) -> str:
    ext = path.suffix.lower()
    p = str(path)
    if classification in {"staging_notes", "staging_export_bundle"}:
        return "archive-local"
    if classification == "historical_regime_docs":
        return "keep-historical"
    if classification == "legacy_tests_or_fixtures":
        return "merge"
    if source_repo == "SurrogateNN_Estimation.jl":
        if classification in {"active_script"} and KEYWORD_RE.search(path.name):
            return "wrap"
        if classification.startswith("active_"):
            return "merge"
        if ext in {".log", ".pdf", ".zip", ".mat", ".jls"}:
            return "archive-local"
        return "keep-historical"

    # External repo
    if classification == "external_hlt_sep_dev":
        if ext == ".jl" and KEYWORD_RE.search(path.name):
            return "adopt"
        if ext == ".md":
            return "keep-historical"
        return "archive-local"
    if classification == "external_validation_package":
        if ext == ".jl":
            return "adopt"
        return "archive-local"
    if classification == "external_general_dev":
        if ext == ".jl" and KEYWORD_RE.search(path.name):
            return "adopt"
        if ext in {".log", ".mat", ".jls", ".pdf", ".zip"}:
            return "drop"
        return "keep-historical"
    return "keep-historical"


def build_roots(target_repo: Path, external_repo: Path) -> list[RootSpec]:
    roots: list[RootSpec] = []
    target_paths = [
        ("src", target_repo / "src"),
        ("scripts", target_repo / "scripts"),
        ("models", target_repo / "models"),
        ("test", target_repo / "test"),
        ("tests", target_repo / "tests"),
        ("docs/regime_switching_estimation", target_repo / "docs" / "regime_switching_estimation"),
        ("archive/research/regime_switching", target_repo / "archive" / "research" / "regime_switching"),
        ("Claude", target_repo / "Claude"),
        ("Cleanup - Exporting functionality", target_repo / "Cleanup - Exporting functionality"),
    ]
    for label, root in target_paths:
        if root.exists():
            roots.append(RootSpec("SurrogateNN_Estimation.jl", root, label))

    external_paths = [
        ("SW07_development", external_repo / "SW07_development"),
        ("development code", external_repo / "development code"),
        ("clean_validation_package", external_repo / "clean_validation_package"),
    ]
    for label, root in external_paths:
        if root.exists():
            roots.append(RootSpec("Non_Linear_DSGE", root, label))
    return roots


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Generate cross-repo provenance inventory and migration matrix.")
    ap.add_argument("--target-repo", default="/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl")
    ap.add_argument("--external-repo", default="/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE")
    ap.add_argument("--out-root", default="/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/.local_artifacts/consolidation")
    ap.add_argument("--docs-root", default="/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/docs/consolidation")
    args = ap.parse_args()

    target_repo = Path(args.target_repo).resolve()
    external_repo = Path(args.external_repo).resolve()
    out_root = Path(args.out_root).resolve()
    docs_root = Path(args.docs_root).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    roots = build_roots(target_repo, external_repo)
    rows: list[dict] = []
    for spec in roots:
        for p in iter_files(spec.root):
            try:
                st = p.stat()
                rel_to_repo = str(p.relative_to(target_repo if spec.source_repo == "SurrogateNN_Estimation.jl" else external_repo))
            except Exception:
                continue
            cls = classify(p, spec.source_repo)
            act = action_for(p, spec.source_repo, cls)
            cand = candidate_target(p, spec.source_repo, cls, rel_to_repo)
            rows.append(
                {
                    "source_repo": spec.source_repo,
                    "root_label": spec.label,
                    "path": str(p),
                    "repo_relative_path": rel_to_repo,
                    "ext": p.suffix.lower(),
                    "size_bytes": st.st_size,
                    "mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                    "sha256": sha256_file(p),
                    "classification": cls,
                    "candidate_target": cand,
                    "action": act,
                }
            )

    rows.sort(key=lambda r: (r["source_repo"], r["repo_relative_path"]))
    write_csv(out_dir / "provenance_inventory.csv", rows)
    (out_dir / "provenance_inventory.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # HLT/SEP/surrogate overlap report across repos for text sources/docs
    overlap_candidates = [
        r for r in rows
        if r["ext"] in {".jl", ".md", ".toml"}
        and KEYWORD_RE.search(Path(r["repo_relative_path"]).name)
    ]
    by_basename: dict[str, list[dict]] = defaultdict(list)
    for r in overlap_candidates:
        by_basename[Path(r["repo_relative_path"]).name].append(r)

    overlap_rows: list[dict] = []
    for basename, grp in sorted(by_basename.items()):
        repos = sorted({g["source_repo"] for g in grp})
        if len(repos) < 2:
            continue
        hashes = {g.get("sha256", "") for g in grp if g.get("sha256", "")}
        overlap_rows.append(
            {
                "basename": basename,
                "repos": ",".join(repos),
                "exact_hash_match": "yes" if len(hashes) == 1 and hashes else "no",
                "paths": " | ".join(g["repo_relative_path"] for g in grp),
                "actions": " | ".join(g["action"] for g in grp),
            }
        )
    write_csv(out_dir / "hlt_overlap_report.csv", overlap_rows)

    # Summaries for docs
    by_source = Counter(r["source_repo"] for r in rows)
    by_class = Counter(r["classification"] for r in rows)
    by_action = Counter(r["action"] for r in rows)
    staging_rows = [r for r in rows if r["classification"] in {"staging_notes", "staging_export_bundle"}]

    prov_md = []
    prov_md.append("# Provenance Map")
    prov_md.append("")
    prov_md.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    prov_md.append("")
    prov_md.append("## Audit Scope")
    for spec in roots:
        prov_md.append(f"- `{spec.source_repo}` :: `{spec.label}` -> `{spec.root}`")
    prov_md.append("")
    prov_md.append("## Raw Outputs")
    prov_md.append(f"- Inventory CSV: `{out_dir / 'provenance_inventory.csv'}`")
    prov_md.append(f"- Inventory JSON: `{out_dir / 'provenance_inventory.json'}`")
    prov_md.append(f"- HLT overlap CSV: `{out_dir / 'hlt_overlap_report.csv'}`")
    prov_md.append("")
    prov_md.append("## Counts by Source Repo")
    for k, v in sorted(by_source.items()):
        prov_md.append(f"- `{k}`: {v}")
    prov_md.append("")
    prov_md.append("## Counts by Classification")
    for k, v in by_class.most_common():
        prov_md.append(f"- `{k}`: {v}")
    prov_md.append("")
    prov_md.append("## Counts by Proposed Action")
    for k, v in by_action.most_common():
        prov_md.append(f"- `{k}`: {v}")
    prov_md.append("")
    prov_md.append("## Key HLT/SEP/Surrogate Overlaps (Cross-Repo)")
    if overlap_rows:
        prov_md.append("| Basename | Exact Hash Match | Repos |")
        prov_md.append("|---|---:|---|")
        for r in overlap_rows[:40]:
            prov_md.append(f"| `{r['basename']}` | {r['exact_hash_match']} | `{r['repos']}` |")
    else:
        prov_md.append("- No cross-repo overlaps found for the keyword filter.")
    prov_md.append("")
    prov_md.append("## Staging Folders (Planned Externalization)")
    if staging_rows:
        for r in staging_rows[:20]:
            prov_md.append(f"- `{r['repo_relative_path']}` -> `{r['candidate_target']}` (`{r['action']}`)")
        if len(staging_rows) > 20:
            prov_md.append(f"- ... plus {len(staging_rows)-20} additional staged files in raw inventory.")
    else:
        prov_md.append("- No staged files detected under `Claude/` or `Cleanup - Exporting functionality/`.")
    (docs_root / "PROVENANCE_MAP.md").write_text("\n".join(prov_md) + "\n", encoding="utf-8")

    # Migration decision matrix: HLT/SEP/surrogate + staging + testing/docs structure hotspots
    matrix_rows = [
        r for r in rows
        if KEYWORD_RE.search(r["repo_relative_path"]) or r["classification"] in {
            "staging_notes", "staging_export_bundle", "legacy_tests_or_fixtures", "active_regime_docs", "historical_regime_docs"
        }
    ]
    matrix_rows.sort(key=lambda r: (r["action"], r["source_repo"], r["repo_relative_path"]))

    md = []
    md.append("# Migration Decision Matrix")
    md.append("")
    md.append(f"Generated from provenance inventory: `{out_dir / 'provenance_inventory.csv'}`")
    md.append("")
    md.append("Allowed actions used: `adopt`, `merge`, `wrap`, `archive-local`, `drop`, `keep-historical`")
    md.append("")
    md.append("| Action | Source Repo | Path | Classification | Candidate Target |")
    md.append("|---|---|---|---|---|")
    for r in matrix_rows[:400]:
        md.append(
            f"| `{r['action']}` | `{r['source_repo']}` | `{r['repo_relative_path']}` | `{r['classification']}` | `{r['candidate_target']}` |"
        )
    if len(matrix_rows) > 400:
        md.append("")
        md.append(f"_Truncated in Markdown view ({len(matrix_rows)-400} additional rows). Use the raw inventory CSV/JSON for the full matrix source._")
    (docs_root / "MIGRATION_DECISION_MATRIX.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    latest = out_root / "LATEST_RUN.txt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(out_dir) + "\n", encoding="utf-8")
    print(f"Wrote inventory to {out_dir}")
    print(f"Updated docs in {docs_root}")


if __name__ == "__main__":
    main()
