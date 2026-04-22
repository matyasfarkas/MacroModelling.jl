#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  chunk_compile_sections.sh [tex_file] [all|N]

Examples:
  chunk_compile_sections.sh
  chunk_compile_sections.sh farkas_jmp_2026.tex all
  chunk_compile_sections.sh farkas_jmp_2026.tex 5

Behavior:
  - Builds incremental "prefix" chunks by section.
  - Each chunk contains the original preamble and content up to a section boundary.
  - Helps isolate which section introduces compile failures or stalls.
EOF
}

tex_file="${1:-farkas_jmp_2026.tex}"
mode="${2:-all}"

if [[ "${tex_file}" == "-h" || "${tex_file}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${tex_file}" ]]; then
  echo "ERROR: TeX file not found: ${tex_file}" >&2
  exit 1
fi

tex_dir="$(cd "$(dirname "${tex_file}")" && pwd)"
tex_abs="${tex_dir}/$(basename "${tex_file}")"
base="$(basename "${tex_abs}" .tex)"
build_dir="${tex_dir}/.chunk_build"
mkdir -p "${build_dir}"

section_lines=()
while IFS=: read -r line_num _rest; do
  section_lines+=("${line_num}")
done < <(rg -n '^[[:space:]]*\\section\{' "${tex_abs}")

section_titles=()
while IFS= read -r line; do
  section_titles+=("${line}")
done < <(rg -n '^[[:space:]]*\\section\{' "${tex_abs}" | sed -E 's/^[0-9]+:[[:space:]]*\\section\{(.*)\}/\1/')

if [[ "${#section_lines[@]}" -eq 0 ]]; then
  echo "ERROR: No \\section{...} lines found in ${tex_abs}" >&2
  exit 1
fi

end_document_line="$(rg -n '\\end\{document\}' "${tex_abs}" | head -n1 | cut -d: -f1)"
if [[ -z "${end_document_line}" ]]; then
  echo "ERROR: Missing \\end{document} in ${tex_abs}" >&2
  exit 1
fi

compile_prefix() {
  local idx="$1"
  local n_sections="${#section_lines[@]}"
  local sec_end
  if (( idx < n_sections )); then
    sec_end=$(( section_lines[idx] - 1 ))
  else
    sec_end=$(( end_document_line - 1 ))
  fi

  local chunk_tex="${build_dir}/${base}.chunk_${idx}.tex"
  local chunk_job="${base}.chunk_${idx}"
  local chunk_log="${build_dir}/${chunk_job}.log"
  local title="${section_titles[idx-1]}"

  sed -n "1,${sec_end}p" "${tex_abs}" > "${chunk_tex}"
  printf '\n\\end{document}\n' >> "${chunk_tex}"

  echo "==> Compiling chunk ${idx}/${n_sections}: ${title}"

  (
    cd "${tex_dir}"
    latexmk -C -silent "${chunk_tex}" >/dev/null 2>&1 || true
    latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname="${chunk_job}" "${chunk_tex}" >"${chunk_log}" 2>&1
  )

  echo "    OK: ${build_dir}/${chunk_job}.pdf"
}

if [[ "${mode}" == "all" ]]; then
  for i in "${!section_lines[@]}"; do
    idx=$(( i + 1 ))
    if ! compile_prefix "${idx}"; then
      echo "FAILED at chunk ${idx}. See log: ${build_dir}/${base}.chunk_${idx}.log" >&2
      tail -n 80 "${build_dir}/${base}.chunk_${idx}.log" >&2 || true
      exit 1
    fi
  done
  echo "All section-prefix chunks compiled."
else
  if ! [[ "${mode}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: mode must be 'all' or a section index (N)." >&2
    usage
    exit 1
  fi
  idx="${mode}"
  if (( idx < 1 || idx > ${#section_lines[@]} )); then
    echo "ERROR: section index out of range: ${idx} (1..${#section_lines[@]})." >&2
    exit 1
  fi
  compile_prefix "${idx}"
fi
