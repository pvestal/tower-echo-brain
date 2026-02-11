# Video Output Quality Analysis

## THE PROBLEM: Videos are 1-frame garbage

### What's Happening
**The smoke test is destroying AnimateDiff video generation:**

1. **Original workflow:** `batch_size=24` → 24 frames → 2-second video at 12fps
2. **Smoke test override:** `batch_size=1` → 1 frame → 0.083s "video"
3. **Result:** 50KB MP4 files containing a single still frame

### Root Cause
```python
# In tower_anime_smoke_test.py line 879:
node["inputs"]["batch_size"] = 2 if has_rife else 1
```

The smoke test sets batch_size to 1 for non-RIFE workflows to speed up testing. But AnimateDiff REQUIRES multiple frames to generate motion. With 1 frame, it produces:
- No temporal coherence
- No motion
- Just a single generated image wrapped in an MP4 container
- "Garbage" video that's actually a still image

### Evidence
```bash
ffprobe GOBLIN_ACTION_COMBAT_00031.mp4:
  nb_frames=1      # Only 1 frame!
  duration=0.083s  # Less than 1/10th second
  width=512
  height=512
  frame_rate=12/1
```

### Why It Seemed to Work
- E2E test passes because files exist and have valid headers
- Validator confirms they're valid MP4s
- But nobody checked if they're actual VIDEOS with motion

## THE FIX

### Option 1: Proper AnimateDiff Testing
```python
# Detect AnimateDiff workflows and preserve minimum frames
if ct == "EmptyLatentImage":
    has_animatediff = any(
        "AnimateDiff" in n.get("class_type", "")
        for n in workflow.values() if isinstance(n, dict)
    )

    if has_animatediff:
        # AnimateDiff needs 16+ frames for coherent motion
        node["inputs"]["batch_size"] = max(16, node["inputs"].get("batch_size", 16))
    elif has_rife:
        node["inputs"]["batch_size"] = 2
    else:
        node["inputs"]["batch_size"] = 1
```

### Option 2: Skip Video Workflows in Smoke Test
```python
# Prefer static image workflows for smoke testing
candidates = [
    "anime_character_simple.json",       # Static image
    "cyberpunk_character_production.json", # Static image
    "simple_image_test.json",            # Static image
    # Skip AnimateDiff workflows in smoke test
]
```

### Option 3: Separate Video Testing
Create `test_video_generation()` that:
- Uses appropriate batch_size (16-24 frames)
- Accepts longer timeout (30-60s)
- Validates frame count > 1
- Checks actual video duration

## Impact Analysis

### Current State
- **Videos:** 1-frame garbage (not videos at all)
- **File size:** ~50KB (should be 500KB-2MB for real videos)
- **Duration:** 0.083s (should be 1-2s minimum)
- **Motion:** None (AnimateDiff disabled by low frame count)

### After Fix
- **Videos:** Proper multi-frame animations
- **File size:** 500KB-2MB
- **Duration:** 1.3s (16 frames) to 2s (24 frames) at 12fps
- **Motion:** Actual animation with temporal coherence

## Recommendations

1. **Immediate:** Fix smoke test to detect AnimateDiff and set `batch_size >= 16`
2. **Short term:** Add video-specific validation (frame count, duration)
3. **Long term:** Separate test suites for images vs videos with appropriate timeouts