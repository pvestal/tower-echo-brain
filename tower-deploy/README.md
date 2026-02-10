# Tower Anime Pipeline — Remediation Package

## What This Fixes

| Problem | Severity | Fix |
|---------|----------|-----|
| 75% of workflows produce 0 trackable outputs | 🔴 Critical | `fix_workflow_outputs.py` adds SaveImage nodes |
| E2E test marks "0 images" as PASS | 🔴 Critical | `smoke_test_e2e_patch.py` with real validation |
| ComfyUI caching returns stale empty results | 🔴 Critical | Random seed injection in smoke test |
| VHS_VideoCombine files invisible to validator | 🟡 Major | Disk-scan fallback in `generation_output_validator.py` |
| No validation tracking in SSOT | 🟡 Major | `generation_validation` table + recording |
| No pipeline monitoring | 🟡 Major | `generation_monitor.py` + SQL views |
| `generation_history` schema conflicts | 🟡 Major | `schema_alignment.sql` links tables |

## Quick Deploy

```bash
cd tower-deploy
chmod +x deploy.sh
./deploy.sh --dry-run    # Preview first
./deploy.sh              # Apply everything
```

## Files

```
tower-deploy/
├── deploy.sh                              # One-command deployment
├── README.md                              # This file
├── scripts/
│   ├── generation_output_validator.py     # Core: validate outputs + VHS fallback
│   ├── fix_workflow_outputs.py            # Fix: add SaveImage to VHS-only workflows
│   ├── smoke_test_e2e_patch.py            # Fix: replacement E2E test function
│   └── generation_monitor.py             # Monitor: pipeline stats + failures
└── sql/
    └── schema_alignment.sql               # DB: validation tables + monitoring views
```

## Manual Steps

The deploy script handles most things automatically, but the smoke test
patch requires manual integration:

1. Open `/opt/tower-echo-brain/scripts/tower_anime_smoke_test.py`
2. Add `import random, struct` at the top
3. Replace `test_end_to_end_generation()` with the version from
   `smoke_test_e2e_patch.py`

## How the VHS Fallback Works

The root cause of the 75% failure rate:

```
Normal node (SaveImage):
  ComfyUI runs → saves file → registers in history outputs → validator finds it ✅

VHS_VideoCombine:
  ComfyUI runs → saves file → returns VHS_FILENAMES type → NOT in history outputs → validator finds nothing ❌
```

The validator now does:
1. Check ComfyUI history (standard path)
2. If 0 outputs AND workflow has VHS_VideoCombine:
   - Get submission timestamp from execution_start message
   - Scan `/opt/ComfyUI/output/` for files created in that time window
   - Validate those files instead
3. Log that VHS fallback was used (for monitoring)

The proper fix is `fix_workflow_outputs.py` which adds a SaveImage node
alongside VHS_VideoCombine so both video AND trackable frame output exist.

## Monitoring

After deployment, check pipeline health with:

```bash
# Quick stats
python3 /opt/tower-echo-brain/scripts/generation_monitor.py

# Or via SQL
psql -d anime_production -c "SELECT * FROM v_generation_stats_24h;"
psql -d anime_production -c "SELECT * FROM v_recent_failures;"
```

## Verification

```bash
# 1. Check workflows are fixed
python3 /opt/tower-echo-brain/scripts/fix_workflow_outputs.py --audit

# 2. Run smoke test (should show validated image count, not "0 images")
python3 /opt/tower-echo-brain/scripts/tower_anime_smoke_test.py --verbose

# 3. Validate a specific generation
python3 /opt/tower-echo-brain/scripts/generation_output_validator.py --latest

# 4. Check SSOT recording
psql -d anime_production -c "SELECT * FROM generation_validation ORDER BY validated_at DESC LIMIT 5;"
```
