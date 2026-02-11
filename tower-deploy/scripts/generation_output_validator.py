#!/usr/bin/env python3
"""
Generation Output Validator (v2 — VHS Fallback)
=================================================
The missing link in the E2E pipeline. After ComfyUI reports "completed",
this module:

  1. Locates output files via ComfyUI history API
  2. FALLBACK: If history reports 0 outputs (VHS_VideoCombine issue),
     scans output directory for files created during the generation window
  3. Validates files are real images/videos (not corrupt, not zero-byte)
  4. Runs quality gates (dimensions, file size, blank detection)
  5. Records results to SSOT (generation_validation table)

VHS_VideoCombine Problem:
  ComfyUI's history API only tracks outputs from nodes that return
  OUTPUT_NODE = True (like SaveImage). VHS_VideoCombine saves to disk
  but returns VHS_FILENAMES type, which ComfyUI ignores in history.
  This causes 75% of workflows to report "0 outputs" despite generating
  real files. The fallback disk scan solves this.

Dependencies: Pillow (pip install Pillow --break-system-packages)
              psycopg2 (for SSOT recording)

Usage:
    python3 generation_output_validator.py <prompt_id>
    python3 generation_output_validator.py --latest
    python3 generation_output_validator.py --scan
    python3 generation_output_validator.py --scan --scan-hours 48

Place at: /opt/tower-echo-brain/scripts/generation_output_validator.py
"""

import argparse
import json
import logging
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    COMFYUI_URL = os.getenv("COMFYUI_URL", "http://localhost:8188")
    COMFYUI_OUTPUT_DIR = os.getenv("COMFYUI_OUTPUT_DIR", "/opt/ComfyUI/output")

    PG_HOST = os.getenv("PG_HOST", "localhost")
    PG_PORT = int(os.getenv("PG_PORT", "5432"))
    PG_USER = os.getenv("PG_USER", "patrick")
    PG_PASSWORD = os.getenv("PG_PASSWORD", "")
    PG_DATABASE = os.getenv("PG_DATABASE", "anime_production")

    # Quality gates
    MIN_FILE_SIZE_BYTES = 10_000          # 10KB — anything smaller is junk
    MAX_FILE_SIZE_BYTES = 50_000_000      # 50MB — sanity cap
    MIN_WIDTH = 256
    MIN_HEIGHT = 256
    MAX_BLANK_RATIO = 0.95                # >95% same color → blank
    MIN_STDDEV = 5.0                      # Minimum pixel std dev

    # VHS fallback: how many seconds before/after submission to look for files
    VHS_SCAN_WINDOW_SECONDS = 300         # 5 minutes

    # Known VHS output patterns
    VHS_OUTPUT_EXTENSIONS = {".mp4", ".webm", ".gif", ".png", ".apng"}
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov"}
    ALL_OUTPUT_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("output_validator")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImageVerdict:
    """Result of validating a single output file (image or video)."""
    path: str
    exists: bool = False
    valid_file: bool = False
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    format: str = ""
    is_blank: bool = False
    pixel_stddev: float = 0.0
    is_video: bool = False
    passed: bool = False
    found_via: str = ""           # "history" or "vhs_fallback"
    issues: list = field(default_factory=list)


@dataclass
class GenerationVerdict:
    """Result of validating an entire generation job."""
    prompt_id: str
    status: str = ""               # passed, failed, no_output, partial, error
    images: list = field(default_factory=list)
    total_files: int = 0
    passed_files: int = 0
    failed_files: int = 0
    workflow_used: str = ""
    model_used: str = ""
    prompt_text: str = ""
    generation_time_s: float = 0.0
    recorded_to_ssot: bool = False
    used_vhs_fallback: bool = False
    issues: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def http_json(method: str, url: str, payload: dict = None, timeout: int = 30):
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.error(f"HTTP {method} {url} failed: {e}")
        return None


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

def validate_output_file(filepath: str, found_via: str = "history") -> ImageVerdict:
    """
    Validate a single output file — image or video.
    Uses Pillow for images if available, header parsing as fallback.
    """
    v = ImageVerdict(path=filepath, found_via=found_via)
    ext = Path(filepath).suffix.lower()
    v.is_video = ext in Config.VIDEO_EXTENSIONS

    # --- File exists? ---
    if not os.path.isfile(filepath):
        v.issues.append("File not found")
        return v
    v.exists = True

    # --- File size ---
    v.file_size_bytes = os.path.getsize(filepath)
    if v.file_size_bytes < Config.MIN_FILE_SIZE_BYTES:
        v.issues.append(f"Too small: {v.file_size_bytes}B (min {Config.MIN_FILE_SIZE_BYTES}B)")
    if v.file_size_bytes > Config.MAX_FILE_SIZE_BYTES:
        v.issues.append(f"Too large: {v.file_size_bytes}B (max {Config.MAX_FILE_SIZE_BYTES}B)")

    # --- Video files ---
    if v.is_video:
        try:
            with open(filepath, "rb") as f:
                header = f.read(32)

            if ext == ".mp4" and b"ftyp" in header[:12]:
                v.valid_file = True
                v.format = "MP4"
                # Try ffprobe for real dimensions
                v.width, v.height = _get_video_dimensions(filepath)
            elif ext == ".webm" and header[:4] == b'\x1a\x45\xdf\xa3':
                v.valid_file = True
                v.format = "WebM"
                v.width, v.height = _get_video_dimensions(filepath)
            elif ext == ".avi" and header[:4] == b'RIFF' and header[8:12] == b'AVI ':
                v.valid_file = True
                v.format = "AVI"
            elif ext == ".mov" and b"moov" in header[:32]:
                v.valid_file = True
                v.format = "MOV"
            else:
                v.issues.append(f"Unrecognized video format for {ext}")
        except Exception as e:
            v.issues.append(f"Video validation error: {e}")

        v.passed = (
            v.exists and v.valid_file
            and v.file_size_bytes >= Config.MIN_FILE_SIZE_BYTES
            and len(v.issues) == 0
        )
        return v

    # --- Image files (Pillow or header parsing) ---
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(filepath)
        img.verify()
        img = Image.open(filepath)  # Re-open after verify
        v.width, v.height = img.size
        v.format = img.format or ext.lstrip(".")
        v.valid_file = True

        if v.width < Config.MIN_WIDTH:
            v.issues.append(f"Width {v.width} < {Config.MIN_WIDTH}")
        if v.height < Config.MIN_HEIGHT:
            v.issues.append(f"Height {v.height} < {Config.MIN_HEIGHT}")

        # Blank detection
        try:
            arr = np.array(img.convert("RGB"), dtype=np.float32)
            v.pixel_stddev = float(np.std(arr))
            if v.pixel_stddev < Config.MIN_STDDEV:
                v.is_blank = True
                v.issues.append(f"Blank/flat image (stddev={v.pixel_stddev:.1f})")

            pixels = arr.reshape(-1, 3)
            quantized = (pixels // 32).astype(int)
            unique, counts = np.unique(quantized, axis=0, return_counts=True)
            max_ratio = counts.max() / counts.sum()
            if max_ratio > Config.MAX_BLANK_RATIO:
                v.is_blank = True
                v.issues.append(f"Single color dominates {max_ratio:.1%}")
        except Exception:
            pass

        img.close()

    except ImportError:
        v.valid_file, v.width, v.height, v.format = _parse_image_header(filepath)
        if not v.valid_file:
            v.issues.append("Can't parse header (install Pillow for full validation)")

    except Exception as e:
        v.issues.append(f"Image error: {e}")
        v.valid_file = False

    v.passed = (
        v.exists and v.valid_file and not v.is_blank
        and v.file_size_bytes >= Config.MIN_FILE_SIZE_BYTES
        and v.width >= Config.MIN_WIDTH
        and v.height >= Config.MIN_HEIGHT
        and len(v.issues) == 0
    )
    return v


def _get_video_dimensions(filepath: str) -> tuple:
    """Try ffprobe for video dimensions, return (width, height) or (0, 0)."""
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x", filepath],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and "x" in result.stdout:
            w, h = result.stdout.strip().split("x")
            return int(w), int(h)
    except Exception:
        pass
    return 0, 0


def _parse_image_header(filepath: str) -> tuple:
    """Minimal header parsing without Pillow. Returns (valid, w, h, format)."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(32)

        if header[:8] == b'\x89PNG\r\n\x1a\n':
            w = struct.unpack(">I", header[16:20])[0]
            h = struct.unpack(">I", header[20:24])[0]
            return True, w, h, "PNG"

        if header[:2] == b'\xff\xd8':
            w, h = _parse_jpeg_dimensions(filepath)
            return True, w, h, "JPEG"

        if header[:6] in (b'GIF87a', b'GIF89a'):
            w = struct.unpack("<H", header[6:8])[0]
            h = struct.unpack("<H", header[8:10])[0]
            return True, w, h, "GIF"

        if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
            return True, 0, 0, "WEBP"

    except Exception:
        pass
    return False, 0, 0, ""


def _parse_jpeg_dimensions(filepath: str) -> tuple:
    try:
        with open(filepath, "rb") as f:
            f.read(2)
            while True:
                marker = f.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    break
                if marker[1] in (0xC0, 0xC1, 0xC2):
                    f.read(3)
                    h = struct.unpack(">H", f.read(2))[0]
                    w = struct.unpack(">H", f.read(2))[0]
                    return w, h
                else:
                    length = struct.unpack(">H", f.read(2))[0]
                    f.read(length - 2)
    except Exception:
        pass
    return 0, 0


# ---------------------------------------------------------------------------
# ComfyUI output resolution — with VHS fallback
# ---------------------------------------------------------------------------

def resolve_outputs_from_history(prompt_id: str) -> List[str]:
    """Resolve output files from ComfyUI history API (standard path)."""
    history = http_json("GET", f"{Config.COMFYUI_URL}/history/{prompt_id}")
    if not history or prompt_id not in history:
        return []

    entry = history[prompt_id]
    outputs = entry.get("outputs", {})
    paths = []

    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        for img in node_output.get("images", []):
            fname = img.get("filename", "")
            subfolder = img.get("subfolder", "")
            if fname:
                paths.append(os.path.join(Config.COMFYUI_OUTPUT_DIR, subfolder, fname))
        for gif in node_output.get("gifs", []):
            fname = gif.get("filename", "")
            subfolder = gif.get("subfolder", "")
            if fname:
                paths.append(os.path.join(Config.COMFYUI_OUTPUT_DIR, subfolder, fname))

    return paths


def resolve_outputs_vhs_fallback(prompt_id: str) -> List[str]:
    """
    VHS_VideoCombine fallback: scan output directory for files created
    during the generation window.

    VHS_VideoCombine writes directly to disk with filenames like:
      anime_frame_00001.mp4, AnimateDiff_00001.gif, etc.

    We identify these by:
      1. Getting the submission timestamp from ComfyUI history
      2. Scanning the output dir for files modified within the time window
      3. Filtering to known output extensions
    """
    # Get submission timestamp from history
    history = http_json("GET", f"{Config.COMFYUI_URL}/history/{prompt_id}")
    if not history or prompt_id not in history:
        logger.warning(f"No history for {prompt_id}, can't do VHS fallback")
        return []

    entry = history[prompt_id]

    # Extract timestamp from execution messages
    submit_time = None
    status = entry.get("status", {})
    messages = status.get("messages", [])
    for msg in messages:
        if isinstance(msg, list) and len(msg) >= 2:
            if msg[0] == "execution_start":
                ts = msg[1].get("timestamp", 0)
                if ts > 0:
                    # ComfyUI timestamps are milliseconds
                    submit_time = ts / 1000.0
                    break

    if not submit_time:
        # Fall back to "recent files" — last 60 seconds
        submit_time = time.time() - 60
        logger.debug("No execution_start timestamp, using last 60s")

    # Scan output directory
    scan_start = submit_time - 5  # 5s before submission (clock skew buffer)
    scan_end = submit_time + Config.VHS_SCAN_WINDOW_SECONDS

    found = []
    output_dir = Config.COMFYUI_OUTPUT_DIR

    if not os.path.isdir(output_dir):
        return []

    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        if not os.path.isfile(fpath):
            continue

        ext = Path(fname).suffix.lower()
        if ext not in Config.ALL_OUTPUT_EXTENSIONS:
            continue

        mtime = os.path.getmtime(fpath)
        if scan_start <= mtime <= scan_end:
            found.append(fpath)

    if found:
        logger.info(
            f"VHS fallback found {len(found)} files in output dir "
            f"(window: {scan_end - scan_start:.0f}s)"
        )

    return found


def extract_generation_metadata(prompt_id: str) -> dict:
    """Extract workflow, model, and prompt info from ComfyUI history."""
    history = http_json("GET", f"{Config.COMFYUI_URL}/history/{prompt_id}")
    if not history or prompt_id not in history:
        return {}

    entry = history[prompt_id]
    prompt_data = entry.get("prompt", [])

    meta = {
        "workflow": "",
        "model": "",
        "positive_prompt": "",
        "negative_prompt": "",
        "loras": [],
        "has_vhs": False,
        "has_save_image": False,
    }

    workflow = {}
    if isinstance(prompt_data, list) and len(prompt_data) >= 3:
        workflow = prompt_data[2] if isinstance(prompt_data[2], dict) else {}
    elif isinstance(prompt_data, dict):
        workflow = prompt_data

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        inputs = node.get("inputs", {})
        title = node.get("_meta", {}).get("title", "").lower()

        if ct == "CheckpointLoaderSimple":
            meta["model"] = inputs.get("ckpt_name", "")
        elif ct == "LoraLoader":
            meta["loras"].append(inputs.get("lora_name", ""))
        elif ct == "CLIPTextEncode":
            text = inputs.get("text", "")
            if "positive" in title or "prompt" in title:
                meta["positive_prompt"] = text[:500]
            elif "negative" in title:
                meta["negative_prompt"] = text[:500]
        elif ct == "VHS_VideoCombine":
            meta["has_vhs"] = True
        elif ct == "SaveImage":
            meta["has_save_image"] = True

    return meta


# ---------------------------------------------------------------------------
# SSOT recording
# ---------------------------------------------------------------------------

def record_to_ssot(verdict: GenerationVerdict) -> bool:
    """
    Record to generation_validation table in the SSOT.
    Creates the table if it doesn't exist (idempotent).
    Links to generation_history via comfyui_prompt_id when possible.
    """
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=Config.PG_HOST, port=Config.PG_PORT,
            user=Config.PG_USER, password=Config.PG_PASSWORD,
            dbname=Config.PG_DATABASE, connect_timeout=10,
        )
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS generation_validation (
                id SERIAL PRIMARY KEY,
                comfyui_prompt_id VARCHAR(255) UNIQUE,
                prompt_text TEXT,
                model_name VARCHAR(255),
                loras JSONB DEFAULT '[]',
                validation_status VARCHAR(50) NOT NULL,
                quality_score FLOAT,
                output_paths JSONB DEFAULT '[]',
                total_images INTEGER DEFAULT 0,
                passed_images INTEGER DEFAULT 0,
                failed_images INTEGER DEFAULT 0,
                issues JSONB DEFAULT '[]',
                generation_time_s FLOAT,
                workflow_used TEXT,
                image_details JSONB DEFAULT '[]',
                used_vhs_fallback BOOLEAN DEFAULT FALSE,
                validated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        # Index for monitoring queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_gv_status
            ON generation_validation(validation_status)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_gv_validated_at
            ON generation_validation(validated_at)
        """)
        conn.commit()

        # Build image details
        image_details = []
        for img in verdict.images:
            image_details.append({
                "path": img.path,
                "passed": img.passed,
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "size_bytes": img.file_size_bytes,
                "is_blank": img.is_blank,
                "is_video": img.is_video,
                "pixel_stddev": round(img.pixel_stddev, 2),
                "found_via": img.found_via,
                "issues": img.issues,
            })

        quality_score = (
            verdict.passed_files / verdict.total_files
            if verdict.total_files > 0 else 0.0
        )

        cur.execute("""
            INSERT INTO generation_validation (
                comfyui_prompt_id, prompt_text, model_name,
                validation_status, quality_score, output_paths,
                total_images, passed_images, failed_images,
                issues, generation_time_s, workflow_used,
                image_details, used_vhs_fallback
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (comfyui_prompt_id)
            DO UPDATE SET
                validation_status = EXCLUDED.validation_status,
                quality_score = EXCLUDED.quality_score,
                output_paths = EXCLUDED.output_paths,
                total_images = EXCLUDED.total_images,
                passed_images = EXCLUDED.passed_images,
                failed_images = EXCLUDED.failed_images,
                issues = EXCLUDED.issues,
                image_details = EXCLUDED.image_details,
                used_vhs_fallback = EXCLUDED.used_vhs_fallback,
                validated_at = NOW()
        """, (
            verdict.prompt_id,
            verdict.prompt_text[:1000] if verdict.prompt_text else None,
            verdict.model_used,
            verdict.status,
            quality_score,
            json.dumps([img.path for img in verdict.images]),
            verdict.total_files,
            verdict.passed_files,
            verdict.failed_files,
            json.dumps(verdict.issues),
            verdict.generation_time_s,
            verdict.workflow_used,
            json.dumps(image_details),
            verdict.used_vhs_fallback,
        ))

        conn.commit()
        conn.close()
        logger.info(f"Recorded validation for {verdict.prompt_id}")
        return True

    except ImportError:
        logger.warning("psycopg2 not available — skip SSOT recording")
        return False
    except Exception as e:
        logger.error(f"SSOT recording failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

class OutputValidator:

    def validate_generation(self, prompt_id: str,
                            record_ssot: bool = True) -> GenerationVerdict:
        """
        Full validation pipeline:
          1. Try ComfyUI history API for output files
          2. If 0 found → VHS fallback disk scan
          3. Validate each file
          4. Record to SSOT
        """
        verdict = GenerationVerdict(prompt_id=prompt_id)

        # --- Step 1: Resolve via history ---
        file_paths = resolve_outputs_from_history(prompt_id)

        # --- Step 2: VHS fallback if history empty ---
        if not file_paths:
            meta = extract_generation_metadata(prompt_id)
            if meta.get("has_vhs"):
                logger.info(
                    "History reports 0 outputs but workflow has VHS_VideoCombine. "
                    "Scanning output directory..."
                )
                file_paths = resolve_outputs_vhs_fallback(prompt_id)
                if file_paths:
                    verdict.used_vhs_fallback = True
                    verdict.issues.append(
                        f"Outputs found via VHS disk scan ({len(file_paths)} files). "
                        "Consider adding SaveImage node for proper tracking."
                    )
            else:
                # Genuinely no output
                verdict.status = "no_output"
                verdict.issues.append(
                    "ComfyUI completed but no output files found. "
                    "Check SaveImage/SaveAnimatedWEBP nodes."
                )
                if record_ssot:
                    verdict.recorded_to_ssot = record_to_ssot(verdict)
                return verdict

        if not file_paths:
            verdict.status = "no_output"
            verdict.issues.append(
                "No outputs in history AND no recent files in output dir."
            )
            if record_ssot:
                verdict.recorded_to_ssot = record_to_ssot(verdict)
            return verdict

        # --- Extract metadata ---
        meta = extract_generation_metadata(prompt_id)
        verdict.model_used = meta.get("model", "")
        verdict.prompt_text = meta.get("positive_prompt", "")

        # --- Validate each file ---
        found_via = "vhs_fallback" if verdict.used_vhs_fallback else "history"
        for fpath in file_paths:
            file_verdict = validate_output_file(fpath, found_via=found_via)
            verdict.images.append(file_verdict)
            if file_verdict.passed:
                verdict.passed_files += 1
            else:
                verdict.failed_files += 1
                for issue in file_verdict.issues:
                    verdict.issues.append(f"{os.path.basename(fpath)}: {issue}")

        verdict.total_files = len(verdict.images)

        # --- Overall status ---
        if verdict.passed_files == verdict.total_files and verdict.total_files > 0:
            verdict.status = "passed"
        elif verdict.passed_files > 0:
            verdict.status = "partial"
        else:
            verdict.status = "failed"

        # --- Record to SSOT ---
        if record_ssot:
            verdict.recorded_to_ssot = record_to_ssot(verdict)

        return verdict

    def validate_latest(self, n: int = 1, record_ssot: bool = True) -> list:
        """Validate the N most recent ComfyUI generations."""
        history = http_json("GET", f"{Config.COMFYUI_URL}/history")
        if not history:
            logger.error("Cannot fetch ComfyUI history")
            return []

        # Sort by recency
        sorted_ids = sorted(
            history.keys(),
            key=lambda pid: _get_timestamp(history[pid]),
            reverse=True,
        )[:n]

        return [
            self.validate_generation(pid, record_ssot=record_ssot)
            for pid in sorted_ids
        ]

    def scan_output_directory(self, max_age_hours: int = 24) -> list:
        """Scan output dir for recent files, validate without prompt_id."""
        output_dir = Config.COMFYUI_OUTPUT_DIR
        if not os.path.isdir(output_dir):
            logger.error(f"Output dir not found: {output_dir}")
            return []

        cutoff = time.time() - (max_age_hours * 3600)
        verdicts = []

        for fname in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if os.path.getmtime(fpath) < cutoff:
                continue
            ext = Path(fname).suffix.lower()
            if ext not in Config.ALL_OUTPUT_EXTENSIONS:
                continue

            verdicts.append(validate_output_file(fpath, found_via="disk_scan"))

        return verdicts


def _get_timestamp(history_entry: dict) -> float:
    """Extract a sortable timestamp from a history entry."""
    messages = history_entry.get("status", {}).get("messages", [])
    for msg in messages:
        if isinstance(msg, list) and len(msg) >= 2 and msg[0] == "execution_start":
            return msg[1].get("timestamp", 0)
    return 0


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_verdict(verdict: GenerationVerdict):
    print("\n" + "=" * 60)
    print("  GENERATION OUTPUT VALIDATION")
    print("=" * 60)

    icons = {"passed": "✅", "partial": "⚠️", "failed": "❌",
             "no_output": "🚫", "error": "💥"}
    print(f"\n  {icons.get(verdict.status, '❓')} Status: {verdict.status.upper()}")
    print(f"  Prompt ID: {verdict.prompt_id}")
    if verdict.model_used:
        print(f"  Model: {verdict.model_used}")
    if verdict.prompt_text:
        print(f"  Prompt: {verdict.prompt_text[:80]}...")
    if verdict.used_vhs_fallback:
        print(f"  ⚠ Found via VHS disk scan (not ComfyUI history)")

    print(f"\n  Files: {verdict.total_files} total, "
          f"{verdict.passed_files} passed, {verdict.failed_files} failed")

    for img in verdict.images:
        icon = "✅" if img.passed else "❌"
        name = os.path.basename(img.path)
        kind = "video" if img.is_video else "image"
        size_kb = img.file_size_bytes / 1024
        dims = f"{img.width}x{img.height}" if img.width else "?x?"
        via = f" [{img.found_via}]" if img.found_via == "vhs_fallback" else ""
        print(f"\n    {icon} {name}{via}")
        if img.valid_file:
            print(f"       {dims} {img.format} {kind} ({size_kb:.0f}KB)")
            if not img.is_video and img.pixel_stddev > 0:
                print(f"       stddev={img.pixel_stddev:.1f}")
        for issue in img.issues:
            print(f"       ⚠ {issue}")

    if verdict.issues:
        print(f"\n  Issues:")
        for issue in verdict.issues[:10]:
            print(f"    • {issue}")

    print(f"\n  SSOT Recorded: {'✅' if verdict.recorded_to_ssot else '❌'}")
    print("=" * 60)


def print_scan_results(verdicts: list):
    print("\n" + "=" * 60)
    print("  OUTPUT DIRECTORY SCAN")
    print("=" * 60)

    passed = sum(1 for v in verdicts if v.passed)
    failed = len(verdicts) - passed
    print(f"\n  Found {len(verdicts)} recent files: {passed} valid, {failed} issues\n")

    for v in verdicts:
        icon = "✅" if v.passed else "❌"
        name = os.path.basename(v.path)
        size_kb = v.file_size_bytes / 1024
        dims = f"{v.width}x{v.height}" if v.width else "?x?"
        kind = v.format or "?"
        print(f"  {icon} {name:<40} {kind:>5} {dims:>10} {size_kb:>8.0f}KB")
        for issue in v.issues:
            print(f"     ⚠ {issue}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation Output Validator v2")
    parser.add_argument("prompt_id", nargs="?", help="ComfyUI prompt_id")
    parser.add_argument("--latest", type=int, nargs="?", const=1)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--scan-hours", type=int, default=24)
    parser.add_argument("--no-ssot", action="store_true")

    args = parser.parse_args()
    validator = OutputValidator()

    if args.scan:
        verdicts = validator.scan_output_directory(args.scan_hours)
        print_scan_results(verdicts)
    elif args.latest is not None:
        for v in validator.validate_latest(args.latest, not args.no_ssot):
            print_verdict(v)
    elif args.prompt_id:
        print_verdict(
            validator.validate_generation(args.prompt_id, not args.no_ssot)
        )
    else:
        parser.print_help()
