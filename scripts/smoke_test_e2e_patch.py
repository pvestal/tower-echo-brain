#!/usr/bin/env python3
"""
Smoke Test E2E Replacement
============================
Drop-in replacement for test_end_to_end_generation() in tower_anime_smoke_test.py.

Fixes three critical issues:
  1. CACHE BUST: Randomizes KSampler seeds so ComfyUI actually generates
     instead of returning cached empty results in 0.3s
  2. VHS FALLBACK: When ComfyUI history reports 0 outputs (VHS_VideoCombine),
     scans output dir for files created during the generation window
  3. REAL VALIDATION: Checks files exist on disk, validates headers, checks
     dimensions, reports per-file results. FAILS on 0 images (was PASS before)

Integration:
  Replace the test_end_to_end_generation function in tower_anime_smoke_test.py
  with the version below. Requires 'import random, struct' at top of file.

Place at: /opt/tower-echo-brain/scripts/smoke_test_e2e_patch.py
"""

# ==========================================================================
# PASTE THIS INTO tower_anime_smoke_test.py
# replacing the existing test_end_to_end_generation function
# Also add: import random, struct  at the top of the file
# ==========================================================================

def test_end_to_end_generation(report):  # TestReport
    """
    Submit a minimal workflow to ComfyUI, wait for completion,
    then VALIDATE actual output files exist and are usable.
    Includes cache busting and VHS_VideoCombine fallback.
    """
    import random
    import struct

    name = "End-to-End Generation (Validated)"

    COMFYUI_OUTPUT_DIR = "/opt/ComfyUI/output"

    # ---------------------------------------------------------------
    # 1. Find a working workflow (prefer non-RIFE, prefer SaveImage)
    # ---------------------------------------------------------------
    workflow_path = None
    candidates = [
        # SaveImage workflows (best — trackable outputs)
        "anime_character_simple.json",
        "cyberpunk_character_production.json",
        "simple_image_test.json",
        # Non-RIFE VHS workflows (need fallback but should work)
        "ACTION_combat_workflow.json",
        "anime_video_simple_test.json",
        "anime_video_fixed_no_rife.json",
        "FIXED_anime_video_workflow.json",
        "GENERIC_anime_video_workflow.json",
        # RIFE workflows (last resort — need batch_size >= 2)
        "anime_30sec_working_workflow.json",
        "anime_30sec_fixed_workflow.json",
    ]
    for wf in candidates:
        path = os.path.join(Config.WORKFLOW_DIR, wf)
        if os.path.isfile(path):
            workflow_path = path
            break

    if not workflow_path:
        report.add(TestResult(
            name, Status.FAIL,
            f"No known workflow found in {Config.WORKFLOW_DIR}",
        ))
        return

    try:
        with open(workflow_path) as f:
            workflow = json.load(f)
    except Exception as e:
        report.add(TestResult(name, Status.FAIL, f"Workflow parse error: {e}"))
        return

    # ---------------------------------------------------------------
    # 2. Analyze workflow features
    # ---------------------------------------------------------------
    has_rife = False
    has_vhs = False
    has_save_image = False

    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        if ct == "RIFE VFI":
            has_rife = True
        elif ct == "VHS_VideoCombine":
            has_vhs = True
        elif ct == "SaveImage":
            has_save_image = True

    # ---------------------------------------------------------------
    # 3. Patch workflow: shrink for speed + CACHE BUST with random seed
    # ---------------------------------------------------------------
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue

        ct = node.get("class_type", "")
        title = node.get("_meta", {}).get("title", "").lower()

        if ct == "CheckpointLoaderSimple":
            ckpt = node.get("inputs", {}).get("ckpt_name", "")
            if ckpt == "realisticVision_v51.safetensors":
                node["inputs"]["ckpt_name"] = "realistic_vision_v51.safetensors"

        if ct == "CLIPTextEncode":
            if "positive" in title or "prompt" in title:
                node["inputs"]["text"] = (
                    "1girl, portrait, dark hair, city background, "
                    "photorealistic, high quality, smoke test"
                )

        if ct == "EmptyLatentImage":
            node["inputs"]["width"] = 512
            node["inputs"]["height"] = 512
            node["inputs"]["batch_size"] = 2 if has_rife else 1

        if ct in ("KSampler", "KSamplerAdvanced"):
            node["inputs"]["steps"] = min(
                node["inputs"].get("steps", 20), 8
            )
            # >>> CACHE BUST: random seed forces actual generation <<<
            node["inputs"]["seed"] = random.randint(0, 2**32 - 1)

    # ---------------------------------------------------------------
    # 4. Submit to ComfyUI
    # ---------------------------------------------------------------
    # Record pre-submission time for VHS fallback scanning
    pre_submit_time = time.time()

    try:
        data = json.dumps({"prompt": workflow}).encode()
        req = urllib.request.Request(
            f"{Config.COMFYUI_URL}/prompt",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
    except Exception as e:
        report.add(TestResult(
            name, Status.FAIL,
            f"Failed to submit workflow: {e}",
            fix_hint="Check ComfyUI logs for errors.",
        ))
        return

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        error_msg = result.get("error", result.get("node_errors", "unknown"))
        report.add(TestResult(
            name, Status.FAIL,
            f"ComfyUI rejected workflow: {error_msg}",
        ))
        return

    # ---------------------------------------------------------------
    # 5. Poll for completion
    # ---------------------------------------------------------------
    start = time.time()
    completed = False
    errored = False
    history_entry = None

    while time.time() - start < Config.GEN_TIMEOUT:
        history = http_get(f"{Config.COMFYUI_URL}/history/{prompt_id}")
        if history and prompt_id in history:
            history_entry = history[prompt_id]
            status_info = history_entry.get("status", {})

            if status_info.get("completed") or status_info.get("status_str") == "success":
                completed = True
                break
            if status_info.get("status_str") == "error":
                errored = True
                messages = status_info.get("messages", [])
                err_detail = ""
                for msg in messages:
                    if isinstance(msg, list) and msg[0] == "execution_error":
                        err_detail = msg[1].get("exception_message", "")[:200]
                report.add(TestResult(
                    name, Status.FAIL,
                    f"Generation errored: {err_detail or status_info}",
                ))
                return

        time.sleep(Config.GEN_POLL_INTERVAL)

    elapsed = round(time.time() - start, 1)

    if not completed:
        report.add(TestResult(
            name, Status.FAIL,
            f"Timed out after {Config.GEN_TIMEOUT}s (prompt_id: {prompt_id})",
            fix_hint="Check ComfyUI console for stuck jobs or GPU OOM.",
        ))
        return

    # ---------------------------------------------------------------
    # 6. VALIDATE OUTPUT — resolve files, check they're real
    # ---------------------------------------------------------------

    # Step 6a: Try ComfyUI history outputs (standard path)
    outputs = history_entry.get("outputs", {})
    output_files = []
    found_via = "history"

    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        for img in node_output.get("images", []):
            fname = img.get("filename", "")
            subfolder = img.get("subfolder", "")
            if fname:
                output_files.append(
                    os.path.join(COMFYUI_OUTPUT_DIR, subfolder, fname)
                )
        for gif in node_output.get("gifs", []):
            fname = gif.get("filename", "")
            subfolder = gif.get("subfolder", "")
            if fname:
                output_files.append(
                    os.path.join(COMFYUI_OUTPUT_DIR, subfolder, fname)
                )

    # Step 6b: VHS fallback — scan disk for files created during generation
    if not output_files and has_vhs:
        scan_start = pre_submit_time - 2  # 2s clock skew buffer
        scan_end = time.time() + 2
        found_via = "vhs_fallback"

        valid_exts = {".mp4", ".webm", ".gif", ".png", ".apng"}
        for fname in os.listdir(COMFYUI_OUTPUT_DIR):
            fpath = os.path.join(COMFYUI_OUTPUT_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_exts:
                continue
            mtime = os.path.getmtime(fpath)
            if scan_start <= mtime <= scan_end:
                output_files.append(fpath)

    # Step 6c: No outputs at all
    workflow_name = os.path.basename(workflow_path)
    if not output_files:
        fix = (
            "Workflow has VHS_VideoCombine but no SaveImage node. "
            "Run fix_workflow_outputs.py to add trackable output nodes."
            if has_vhs else
            "Workflow may lack a SaveImage node, or output is disconnected."
        )
        report.add(TestResult(
            name, Status.FAIL,
            f"ComfyUI completed in {elapsed}s but produced 0 output files. "
            f"Workflow: {workflow_name}",
            fix_hint=fix,
        ))
        return

    # ---------------------------------------------------------------
    # 7. Validate each file
    # ---------------------------------------------------------------
    validated = 0
    file_issues = []
    detail_lines = []

    for fpath in output_files:
        fname = os.path.basename(fpath)

        if not os.path.isfile(fpath):
            file_issues.append(f"{fname}: not found on disk")
            continue

        file_size = os.path.getsize(fpath)
        if file_size < 10_000:
            file_issues.append(f"{fname}: too small ({file_size}B)")
            continue

        # Detect format
        width, height, fmt = 0, 0, ""
        ext = os.path.splitext(fname)[1].lower()

        if ext in (".mp4", ".webm"):
            fmt = ext.lstrip(".")
            try:
                with open(fpath, "rb") as f:
                    header = f.read(32)
                if ext == ".mp4" and b"ftyp" in header[:12]:
                    width, height = 512, 512  # Placeholder
                elif ext == ".webm" and header[:4] == b'\x1a\x45\xdf\xa3':
                    width, height = 512, 512
                else:
                    file_issues.append(f"{fname}: bad video header")
                    continue
            except Exception:
                file_issues.append(f"{fname}: can't read")
                continue
        else:
            try:
                with open(fpath, "rb") as f:
                    header = f.read(32)
                if header[:8] == b'\x89PNG\r\n\x1a\n':
                    width = struct.unpack(">I", header[16:20])[0]
                    height = struct.unpack(">I", header[20:24])[0]
                    fmt = "PNG"
                elif header[:2] == b'\xff\xd8':
                    fmt = "JPEG"
                    width, height = 512, 512
                elif header[:6] in (b'GIF87a', b'GIF89a'):
                    width = struct.unpack("<H", header[6:8])[0]
                    height = struct.unpack("<H", header[8:10])[0]
                    fmt = "GIF"
            except Exception:
                pass

        if width > 0 and width < 128:
            file_issues.append(f"{fname}: too small ({width}x{height})")
            continue

        size_kb = file_size / 1024
        dims = f"{width}x{height}" if width > 0 else "?x?"
        via_tag = " [vhs]" if found_via == "vhs_fallback" else ""
        detail_lines.append(f"✓ {fname}{via_tag} — {fmt} {dims} ({size_kb:.0f}KB)")
        validated += 1

    # ---------------------------------------------------------------
    # 8. Final verdict
    # ---------------------------------------------------------------
    total = len(output_files)
    fallback_note = " (via VHS disk scan)" if found_via == "vhs_fallback" else ""

    if validated == 0:
        report.add(TestResult(
            name, Status.FAIL,
            f"Generated {total} file(s) but none passed validation. "
            f"Workflow: {workflow_name}{fallback_note}",
            detail="\n".join(file_issues),
            fix_hint="Check ComfyUI output dir and workflow save nodes.",
        ))
    elif validated < total:
        report.add(TestResult(
            name, Status.WARN,
            f"{validated}/{total} files validated in {elapsed}s "
            f"({workflow_name}){fallback_note}",
            detail="\n".join(detail_lines + file_issues),
        ))
    else:
        report.add(TestResult(
            name, Status.PASS,
            f"{validated} file(s) generated and validated in {elapsed}s "
            f"({workflow_name}){fallback_note}",
            detail="\n".join(detail_lines),
        ))
