#!/usr/bin/env python3
"""
SSOT Compliance Validator
=========================
Verifies the story_bible collection is properly SSOT-compliant:
  ✓ Payloads contain source_table + source_id (references)
  ✗ Payloads do NOT contain full content/text/description duplicates
  ✓ References resolve to real PostgreSQL rows
  ✓ Fresh fetch returns more data than the Qdrant payload

Run after rebuild_story_bible_ssot.py to confirm compliance.

Usage:
    python3 validate_ssot_compliance.py
    python3 validate_ssot_compliance.py --verbose
    python3 validate_ssot_compliance.py --fix    # Report what needs re-indexing

Place at: /opt/tower-echo-brain/scripts/validate_ssot_compliance.py
"""

import argparse
import json
import logging
import os
import sys
import urllib.request
import urllib.error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ssot_validator")


# ---------------------------------------------------------------------------
# Config (same as other scripts)
# ---------------------------------------------------------------------------

class Config:
    PG_HOST = os.getenv("PG_HOST", "localhost")
    PG_PORT = int(os.getenv("PG_PORT", "5432"))
    PG_USER = os.getenv("PG_USER", "patrick")
    PG_PASSWORD = os.getenv("PG_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")
    PG_DATABASE = os.getenv("PG_DATABASE", "anime_production")

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION = "story_bible"

    # Maximum payload size (bytes) for an SSOT-compliant entry.
    # If a payload is bigger than this, it's probably duplicating data.
    MAX_PAYLOAD_BYTES = 2000

    # Required payload fields for SSOT compliance
    REQUIRED_FIELDS = {"type", "source_table", "source_id", "indexed_at"}

    # Fields that indicate data duplication (should NOT be present)
    DUPLICATION_FIELDS = {
        "content", "text", "description", "personality", "appearance",
        "backstory", "dialogue", "narrative_text", "synopsis",
        "full_text", "raw_content",
    }


def http_json(method, url, payload=None, timeout=30):
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
        return None


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def validate(verbose=False, fix_mode=False):
    results = {"pass": 0, "fail": 0, "warn": 0}

    print("\n" + "=" * 60)
    print("  SSOT COMPLIANCE VALIDATION")
    print("=" * 60)

    # --- Check 1: Collection exists ---
    print("\n  [1/6] Collection exists...")
    coll = http_json("GET", f"{Config.QDRANT_URL}/collections/{Config.COLLECTION}")
    if not coll or "result" not in coll:
        print("  ❌ FAIL: story_bible collection not found")
        results["fail"] += 1
        print_summary(results)
        return results

    points_count = coll["result"].get("points_count", 0)
    print(f"  ✅ PASS: {points_count} points in {Config.COLLECTION}")
    results["pass"] += 1

    # --- Check 2: Scroll all points and inspect payloads ---
    print("\n  [2/6] Payload structure compliance...")
    scroll_resp = http_json("POST",
        f"{Config.QDRANT_URL}/collections/{Config.COLLECTION}/points/scroll",
        {"limit": 200, "with_payload": True, "with_vectors": False},
    )

    if not scroll_resp or "result" not in scroll_resp:
        print("  ❌ FAIL: Cannot scroll points")
        results["fail"] += 1
        print_summary(results)
        return results

    points = scroll_resp["result"].get("points", [])

    missing_required = []   # Points missing source_table/source_id
    has_duplication = []     # Points with full content fields
    oversized = []           # Points with payloads > threshold
    compliant = 0
    workflow_points = 0      # Workflows get a pass (not in PG)

    for pt in points:
        payload = pt.get("payload", {})
        pt_id = pt.get("id", "?")
        pt_type = payload.get("type", "unknown")

        # Workflows are file-based, not in PG — different rules
        if pt_type == "workflow" or payload.get("source_type") == "file":
            workflow_points += 1
            continue

        # Check required fields
        missing = Config.REQUIRED_FIELDS - set(payload.keys())
        if missing:
            missing_required.append({
                "id": pt_id,
                "type": pt_type,
                "missing": missing,
            })

        # Check for data duplication
        duped_fields = Config.DUPLICATION_FIELDS & set(payload.keys())
        if duped_fields:
            has_duplication.append({
                "id": pt_id,
                "type": pt_type,
                "duplicated": duped_fields,
            })

        # Check payload size
        payload_bytes = len(json.dumps(payload).encode())
        if payload_bytes > Config.MAX_PAYLOAD_BYTES:
            oversized.append({
                "id": pt_id,
                "type": pt_type,
                "bytes": payload_bytes,
            })

        if not missing and not duped_fields and payload_bytes <= Config.MAX_PAYLOAD_BYTES:
            compliant += 1

    total_db_points = len(points) - workflow_points

    if missing_required:
        print(f"  ❌ FAIL: {len(missing_required)}/{total_db_points} points "
              f"missing SSOT reference fields")
        results["fail"] += 1
        if verbose:
            for item in missing_required[:5]:
                print(f"         id={item['id']} type={item['type']} "
                      f"missing={item['missing']}")
    else:
        print(f"  ✅ PASS: All {total_db_points} DB points have "
              f"source_table + source_id")
        results["pass"] += 1

    # --- Check 3: No data duplication ---
    print("\n  [3/6] No data duplication in payloads...")
    if has_duplication:
        print(f"  ❌ FAIL: {len(has_duplication)}/{total_db_points} points "
              f"contain duplicated content fields")
        results["fail"] += 1
        if verbose:
            for item in has_duplication[:5]:
                print(f"         id={item['id']} type={item['type']} "
                      f"fields={item['duplicated']}")
        if fix_mode:
            print(f"\n  💡 FIX: Run rebuild_story_bible_ssot.py to rebuild "
                  f"as references-only")
    else:
        print(f"  ✅ PASS: No content duplication detected")
        results["pass"] += 1

    # --- Check 4: Payload sizes ---
    print("\n  [4/6] Payload size check (max {:.0f}KB)...".format(
        Config.MAX_PAYLOAD_BYTES / 1024))
    if oversized:
        print(f"  ⚠️  WARN: {len(oversized)} points exceed "
              f"{Config.MAX_PAYLOAD_BYTES}B")
        results["warn"] += 1
        if verbose:
            for item in oversized[:5]:
                print(f"         id={item['id']} type={item['type']} "
                      f"size={item['bytes']}B")
    else:
        print(f"  ✅ PASS: All payloads within size limits")
        results["pass"] += 1

    # --- Check 5: References resolve to PostgreSQL ---
    print("\n  [5/6] Reference resolution (Qdrant → PostgreSQL)...")
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=Config.PG_HOST, port=Config.PG_PORT,
            user=Config.PG_USER, password=Config.PG_PASSWORD,
            dbname=Config.PG_DATABASE, connect_timeout=10,
        )
        cur = conn.cursor()

        # Sample 10 points and verify they resolve
        sample_points = [
            pt for pt in points
            if pt.get("payload", {}).get("source_table")
            and pt.get("payload", {}).get("type") != "workflow"
        ][:10]

        resolved = 0
        dangling = []

        for pt in sample_points:
            p = pt["payload"]
            table = p["source_table"]
            src_id = p["source_id"]

            try:
                # Check table exists
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    LIMIT 1
                """, (table,))
                if not cur.fetchone():
                    dangling.append(f"{table}.{src_id} (table not found)")
                    continue

                # Check row exists
                cur.execute(f"SELECT 1 FROM {table} WHERE id = %s", (src_id,))
                if cur.fetchone():
                    resolved += 1
                else:
                    dangling.append(f"{table}.{src_id} (row not found)")
            except Exception as e:
                dangling.append(f"{table}.{src_id} ({e})")

        conn.close()

        if dangling:
            print(f"  ⚠️  WARN: {len(dangling)}/{len(sample_points)} "
                  f"references are dangling")
            results["warn"] += 1
            for d in dangling:
                print(f"         → {d}")
        else:
            print(f"  ✅ PASS: {resolved}/{len(sample_points)} sampled "
                  f"references resolve to SSOT")
            results["pass"] += 1

    except ImportError:
        print("  ⏭  SKIP: psycopg2 not installed")
    except Exception as e:
        print(f"  ❌ FAIL: PostgreSQL check failed: {e}")
        results["fail"] += 1

    # --- Check 6: Fresh fetch returns more data than payload ---
    print("\n  [6/6] Fresh fetch enrichment check...")
    try:
        conn = psycopg2.connect(
            host=Config.PG_HOST, port=Config.PG_PORT,
            user=Config.PG_USER, password=Config.PG_PASSWORD,
            dbname=Config.PG_DATABASE, connect_timeout=10,
        )
        cur = conn.cursor()

        # Pick a character point to test
        char_points = [
            pt for pt in points
            if pt.get("payload", {}).get("type") == "character"
            and pt.get("payload", {}).get("source_table")
        ]

        if char_points:
            pt = char_points[0]
            p = pt["payload"]
            payload_bytes = len(json.dumps(p).encode())

            cur.execute(f"SELECT * FROM {p['source_table']} WHERE id = %s",
                        (p["source_id"],))
            cols = [desc[0] for desc in cur.description]
            row = cur.fetchone()

            if row:
                row_dict = dict(zip(cols, row))
                # Rough size comparison
                fresh_bytes = sum(
                    len(str(v).encode()) for v in row_dict.values()
                    if v is not None
                )

                if fresh_bytes > payload_bytes:
                    print(f"  ✅ PASS: Fresh fetch ({fresh_bytes}B) > "
                          f"payload ({payload_bytes}B) — "
                          f"SSOT has richer data")
                    results["pass"] += 1
                else:
                    print(f"  ⚠️  WARN: Payload ({payload_bytes}B) ≥ "
                          f"fresh fetch ({fresh_bytes}B) — "
                          f"may be duplicating data")
                    results["warn"] += 1
            else:
                print("  ⚠️  WARN: Test character not found in SSOT")
                results["warn"] += 1
        else:
            print("  ⏭  SKIP: No character points to test")

        conn.close()

    except Exception as e:
        print(f"  ⚠️  WARN: Enrichment check failed: {e}")
        results["warn"] += 1

    print_summary(results)
    return results


def print_summary(results):
    print("\n" + "-" * 60)
    total = results["pass"] + results["fail"] + results["warn"]
    print(f"  Total: {total} | "
          f"Pass: {results['pass']} | "
          f"Fail: {results['fail']} | "
          f"Warn: {results['warn']}")

    if results["fail"] == 0:
        print("\n  🎉 SSOT COMPLIANT — Qdrant is a search index,")
        print("     PostgreSQL is the Single Source of Truth.")
    else:
        print("\n  🔧 NOT SSOT COMPLIANT — Run rebuild_story_bible_ssot.py")
        print("     to fix data duplication issues.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSOT Compliance Validator")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fix", action="store_true",
                        help="Show remediation steps")
    args = parser.parse_args()

    results = validate(verbose=args.verbose, fix_mode=args.fix)
    sys.exit(1 if results["fail"] > 0 else 0)