#!/bin/bash
# ============================================================
# Tower Anime Pipeline — Remediation Deployment
# ============================================================
# Applies all fixes from the gap analysis:
#   1. Schema alignment (generation_validation + metrics tables)
#   2. Workflow fixes (add SaveImage nodes for output tracking)
#   3. Updated scripts (validator, monitor, smoke test patch)
#   4. Verification (smoke test + monitor)
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh              # Full deployment
#   ./deploy.sh --dry-run    # Preview only
#   ./deploy.sh --skip-db    # Skip database changes
#   ./deploy.sh --skip-wf    # Skip workflow fixes
#
# Run from: wherever you extracted the tower-deploy package
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_SRC="$SCRIPT_DIR/scripts"
SQL_SRC="$SCRIPT_DIR/sql"

# Target directories on Tower
TARGET_SCRIPTS="/opt/tower-echo-brain/scripts"
TARGET_WORKFLOWS="/opt/tower-anime-production/workflows/comfyui"

# DB config
PG_HOST="localhost"
PG_USER="patrick"
PG_DB="anime_production"
export PGPASSWORD="RP78eIrW7cI2jYvL5akt1yurE"

DRY_RUN=false
SKIP_DB=false
SKIP_WF=false

for arg in "$@"; do
    case $arg in
        --dry-run)  DRY_RUN=true ;;
        --skip-db)  SKIP_DB=true ;;
        --skip-wf)  SKIP_WF=true ;;
    esac
done

echo ""
echo "============================================================"
echo "  TOWER ANIME PIPELINE — REMEDIATION DEPLOYMENT"
echo "============================================================"
echo ""

# -------------------------------------------------------------------
# Step 1: Deploy scripts
# -------------------------------------------------------------------
echo "Step 1: Deploying scripts to $TARGET_SCRIPTS"

DEPLOY_SCRIPTS=(
    "generation_output_validator.py"
    "fix_workflow_outputs.py"
    "generation_monitor.py"
    "smoke_test_e2e_patch.py"
)

for script in "${DEPLOY_SCRIPTS[@]}"; do
    src="$SCRIPTS_SRC/$script"
    dst="$TARGET_SCRIPTS/$script"

    if [ ! -f "$src" ]; then
        echo "  ⚠ Missing: $script"
        continue
    fi

    if $DRY_RUN; then
        echo "  [dry-run] Would copy $script → $dst"
    else
        cp "$src" "$dst"
        chmod +x "$dst"
        echo "  ✅ $script"
    fi
done

echo ""

# -------------------------------------------------------------------
# Step 2: Apply database schema
# -------------------------------------------------------------------
if $SKIP_DB; then
    echo "Step 2: Database schema — SKIPPED (--skip-db)"
else
    echo "Step 2: Applying database schema alignment"

    if $DRY_RUN; then
        echo "  [dry-run] Would apply $SQL_SRC/schema_alignment.sql"
    else
        if psql -h "$PG_HOST" -U "$PG_USER" -d "$PG_DB" \
            -f "$SQL_SRC/schema_alignment.sql" 2>&1 | tail -5; then
            echo "  ✅ Schema applied"
        else
            echo "  ❌ Schema failed — check errors above"
        fi
    fi
fi

echo ""

# -------------------------------------------------------------------
# Step 3: Fix workflows (add SaveImage nodes)
# -------------------------------------------------------------------
if $SKIP_WF; then
    echo "Step 3: Workflow fixes — SKIPPED (--skip-wf)"
else
    echo "Step 3: Fixing workflows (adding SaveImage nodes)"

    if [ ! -d "$TARGET_WORKFLOWS" ]; then
        echo "  ⚠ Workflow dir not found: $TARGET_WORKFLOWS"
    else
        if $DRY_RUN; then
            python3 "$TARGET_SCRIPTS/fix_workflow_outputs.py" --dry-run
        else
            python3 "$TARGET_SCRIPTS/fix_workflow_outputs.py"
        fi
    fi
fi

echo ""

# -------------------------------------------------------------------
# Step 4: Apply smoke test E2E patch
# -------------------------------------------------------------------
echo "Step 4: Patching smoke test E2E function"

SMOKE_TEST="$TARGET_SCRIPTS/tower_anime_smoke_test.py"

if [ ! -f "$SMOKE_TEST" ]; then
    echo "  ⚠ Smoke test not found: $SMOKE_TEST"
else
    # Check if already patched (look for our signature)
    if grep -q "CACHE BUST" "$SMOKE_TEST" 2>/dev/null; then
        echo "  ✅ Already patched (cache bust present)"
    else
        if $DRY_RUN; then
            echo "  [dry-run] Would patch test_end_to_end_generation()"
            echo "  Note: Manual patch required — replace the function with"
            echo "  the version from smoke_test_e2e_patch.py"
        else
            echo "  ⚠ MANUAL STEP REQUIRED:"
            echo "  Replace test_end_to_end_generation() in"
            echo "    $SMOKE_TEST"
            echo "  with the version from"
            echo "    $TARGET_SCRIPTS/smoke_test_e2e_patch.py"
            echo ""
            echo "  Also add 'import random, struct' to the top of the file."
            echo ""
            echo "  (Automated patching skipped to avoid breaking the file)"
        fi
    fi
fi

echo ""

# -------------------------------------------------------------------
# Step 5: Verify
# -------------------------------------------------------------------
echo "Step 5: Verification"

if $DRY_RUN; then
    echo "  [dry-run] Would run smoke test and monitor"
else
    echo ""
    echo "  --- Workflow Audit ---"
    python3 "$TARGET_SCRIPTS/fix_workflow_outputs.py" --audit 2>&1 || true

    echo ""
    echo "  --- Smoke Test (quick) ---"
    if python3 "$TARGET_SCRIPTS/tower_anime_smoke_test.py" --quick 2>&1 | \
        grep -E "(PASS|FAIL|WARN|Total)"; then
        true
    else
        echo "  ⚠ Smoke test had issues — check output above"
    fi

    echo ""
    echo "  --- Generation Monitor ---"
    python3 "$TARGET_SCRIPTS/generation_monitor.py" --hours 48 2>&1 || true
fi

echo ""
echo "============================================================"
echo "  DEPLOYMENT COMPLETE"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "    1. Run full smoke test:   python3 $TARGET_SCRIPTS/tower_anime_smoke_test.py --verbose"
echo "    2. Test a generation:     python3 $TARGET_SCRIPTS/ssot_generation_orchestrator.py 'Generate Kai in cyberpunk alley'"
echo "    3. Validate output:       python3 $TARGET_SCRIPTS/generation_output_validator.py --latest"
echo "    4. Check monitor:         python3 $TARGET_SCRIPTS/generation_monitor.py"
echo ""
