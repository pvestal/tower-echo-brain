# Diagnosis Report: Tower Anime Production Issues

## Issue #1: Service Not Running (FIXED)
**Documentation claimed:** "API Service: Port 8328 (NOT RUNNING - service doesn't exist)"
**Reality:** Service exists but was just stopped
- Service file: `/etc/systemd/system/tower-anime-production.service` ✅
- API code: `/opt/tower-anime-production/api/main.py` ✅
- Virtual env: `/opt/tower-anime-production/venv/` ✅
- **Status:** NOW RUNNING after `systemctl start tower-anime-production`

## Issue #2: Video Generation Broken (ROOT CAUSE IDENTIFIED)
**Problem:** Smoke test produces 1-frame "videos" (50KB files)
**Cause:** Test optimization overrides `batch_size=1` to speed up testing

### Evidence:
```python
# tower_anime_smoke_test.py line 879
node["inputs"]["batch_size"] = 2 if has_rife else 1  # BREAKS ANIMATEDIFF
```

### Test Results:
**With batch_size=1 (smoke test):**
- Frame count: 1
- Duration: 0.083s
- File size: 50KB
- **Result:** Not a video, just a still image

**With batch_size=16 (proper):**
- Frame count: 16 ✅
- Duration: 1.33s ✅
- File size: 562KB ✅
- **Result:** Actual animated video with motion

## Issue #3: Frontend Directory (EXISTS)
**Documentation claimed:** No frontend directory
**Reality:** `/opt/tower-anime-production/frontend/` exists with:
- Vue.js app with node_modules
- package.json, vite.config.js
- src/ directory with components

## Issue #4: Cache Problem (ALREADY FIXED)
**Problem:** ComfyUI cached all executions, returned 0 outputs
**Solution:** Random seed injection in smoke test
```python
node["inputs"]["seed"] = random.randint(0, 2**31)
```
**Status:** FIXED ✅

## Issue #5: VHS_VideoCombine Tracking (PARTIALLY FIXED)
**Problem:** VHS saves to disk but not in ComfyUI history
**Solutions Applied:**
1. Added SaveImage nodes to all workflows ✅
2. VHS disk-scan fallback in validator ✅
**Status:** Workaround in place, core issue remains

## Summary of Real vs Documented State

| Component | Documentation Said | Reality |
|-----------|-------------------|---------|
| API Service (8328) | Not running, doesn't exist | Exists, just needed starting |
| Video Generation | Broken (1-frame) | Works fine with batch_size≥16 |
| Frontend | Doesn't exist | Exists in frontend/ |
| E2E Tests | 100% pass | Pass but with broken videos |
| AnimateDiff | Working | Works only with proper batch_size |

## Root Causes

1. **Test Optimization Breaking Production:** Smoke test optimizes for speed by setting batch_size=1, which completely breaks AnimateDiff video generation

2. **Service Management:** Service was created but not enabled/started

3. **Documentation Drift:** README wasn't updated as components were added

## Recommended Fixes

### Immediate:
```python
# Fix smoke test to detect AnimateDiff
if has_animatediff:
    node["inputs"]["batch_size"] = max(16, original_batch_size)
```

### Short Term:
- Enable service: `systemctl enable tower-anime-production`
- Separate smoke tests for images vs videos
- Add video validation (check frame count > 1)

### Long Term:
- CI/CD pipeline to keep docs in sync
- Automated service health monitoring
- Performance vs correctness trade-off configuration