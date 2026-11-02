#!/bin/bash
# Static release checks (no GPU, no data): syntax of every Python / shell file,
# no leftover cluster paths or credentials, CLIs start (`--help`) in the env.
#
#   set -a; source .env; set +a
#   bash tests/static_checks.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO}"
PY="${PERSONADRIVE_CONDA_ENV:+${PERSONADRIVE_CONDA_ENV}/bin/}python"
fail=0
ok()  { echo "  ok   $*"; }
bad() { echo "  FAIL $*"; fail=1; }

echo "[1] Python syntax (all first-party .py files)"
out=$(PYTHONDONTWRITEBYTECODE=1 python3 - <<'EOF'
import ast, pathlib
skip = {'.git', 'leaderboard', 'leaderboard_autopilot', 'scenario_runner', 'scenario_runner_autopilot',
        'Bench2Drive', 'database', 'outputs', 'eval_results', 'software', 'pretrained', 'data_sample'}
n, bad = 0, []
for p in pathlib.Path('.').rglob('*.py'):
    if p.parts[0] in skip or 'tests/_work' in str(p):
        continue
    n += 1
    try:
        ast.parse(p.read_text(), str(p))
    except SyntaxError as e:
        bad.append(f"{p}:{e.lineno}: {e.msg}")
print("\n".join(bad) if bad else f"{n} files parse")
EOF
)
if echo "$out" | grep -q "files parse"; then ok "$out"; else bad "$out"; fi

echo "[2] Shell syntax"
nsh=0; shbad=""
while IFS= read -r -d '' f; do
    nsh=$((nsh + 1)); bash -n "$f" 2>/dev/null || shbad+=" $f"
done < <(find . -name '*.sh' -not -path './.git/*' -not -path './tests/_work/*' -print0)
[ -z "$shbad" ] && ok "$nsh shell scripts" || bad "bash -n failed:$shbad"

echo "[3] No cluster paths, user names or secrets in tracked text"
# Cluster paths and stale script names (data/augmented_templates is VQA template text).
# Add your own user names or paths as an extended regex in PD_EXTRA_IDENTIFIERS
# (e.g. in your .env).
pattern='/share/crsp|/pub/[a-z][a-z0-9_]+/|/dfs[0-9]|/data/homezvol|hpc3\.|SELMALAK|selmalak_lab|SIMLINGO_|faies'
pattern="${pattern}${PD_EXTRA_IDENTIFIERS:+|${PD_EXTRA_IDENTIFIERS}}"
# Secret-shaped strings: OpenAI / HF tokens, private keys, literal password assignments in code.
secrets='sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|(password|passwd) *[=:] *["'"'"'][^"'"'"' ]{4,}'
files=$(git ls-files -co --exclude-standard 2>/dev/null | grep -vE '^(tests/static_checks\.sh|data/augmented_templates/)')
# SIMLINGO_ROOT is the documented variable of human_data_collection/ (its bundled SimLingo tree), not a stale name.
hits=$(echo "$files" | tr '\n' '\0' | xargs -0 grep -nIE "$pattern" 2>/dev/null | grep -v 'SIMLINGO_ROOT' || true)
shits=$(echo "$files" | grep -E '\.(py|sh|yaml|yml|json|txt|cfg|ini|env)$|^\.env' | tr '\n' '\0' | xargs -0 grep -nIE "$secrets" 2>/dev/null || true)
[ -z "$hits" ]  && ok "no cluster paths / identifiers" || bad "cluster paths:"$'\n'"$hits"
[ -z "$shits" ] && ok "no secret-shaped strings"      || bad "possible secrets:"$'\n'"$shits"

echo "[4] Files that must never ship"
for f in persona_drive_rag/pull_data.py arial.ttf data/simlingo.zip Bench2Drive/assets/overview.mp4 .env; do
    if git ls-files -co --exclude-standard --error-unmatch "$f" >/dev/null 2>&1; then bad "$f would be committed"; fi
done
[ $fail -eq 0 ] && ok "none tracked"

echo "[5] CLIs start (--help)"
cli_log="$(mktemp)"; trap 'rm -f "${cli_log}"' EXIT
show_log() { tail -n 3 "${cli_log}" | sed 's/^/         /'; }
for s in rag_data_processor_v2.py embed_mine.py embed_single.py embed_merge.py embed_training.py build_faiss_index.py rag_app.py logs.py rag_pivot_runner.py; do
    (cd persona_drive_rag && timeout 600 "${PY}" "$s" --help >"${cli_log}" 2>&1) && ok "[retrieval] persona_drive_rag/$s" || { bad "[retrieval] persona_drive_rag/$s --help (exit $?)"; show_log; }
done
for s in scripts/training/generate_icl_examples.py tests/make_mini_database.py scripts/data_processing/records_integrating.py scripts/data_processing/merge_measurements.py \
         scripts/analysis/style_conditioning_eval.py scripts/analysis/style_retrieval_probe.py scripts/analysis/waypoint_accuracy_sweep.py scripts/analysis/qa_accuracy_sweep.py \
         scripts/analysis/benchmark_icl_timing.py scripts/analysis/style_speed_stats.py scripts/eval/paper_tables.py scripts/eval/aggregate_b2d_metrics.py scripts/eval/aggregate_b2d_metrics_k.py scripts/data_processing/run_pipeline.py; do
    PYTHONPATH="${REPO}:${PYTHONPATH:-}" timeout 600 "${PY}" "$s" --help >"${cli_log}" 2>&1 && ok "[main] $s" || { bad "[main] $s --help (exit $?)"; show_log; }
done

echo "[6] Training config resolves"
# On a GPU node, importing DeepSpeed needs CUDA_HOME; use the env's own nvcc if
# no toolkit is set (envs built by environment/setup_env.sh ship one).
_main_prefix="$(cd "$(dirname "$(command -v "${PY}")")/.." && pwd)"
if [ -z "${CUDA_HOME:-}" ] && [ -x "${_main_prefix}/bin/nvcc" ]; then export CUDA_HOME="${_main_prefix}"; fi
PYTHONPATH="${REPO}:${PYTHONPATH:-}" timeout 600 "${PY}" simlingo_training/train.py experiment=personadrive_icl_sft --cfg job \
    > /tmp/pd_cfg_$$.txt 2>&1 && grep -q "use_icl: true" /tmp/pd_cfg_$$.txt && grep -q "num_icl_examples: 2" /tmp/pd_cfg_$$.txt \
    && ok "experiment=personadrive_icl_sft (use_icl, K=2)" || { bad "hydra config"; tail -5 /tmp/pd_cfg_$$.txt; }
rm -f /tmp/pd_cfg_$$.txt

echo
[ $fail -eq 0 ] && echo "STATIC CHECKS PASSED" || echo "STATIC CHECKS FAILED"
exit $fail
