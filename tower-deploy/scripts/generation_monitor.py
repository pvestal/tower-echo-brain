#!/usr/bin/env python3
"""
Generation Pipeline Monitor
==============================
Quick stats and failure analysis from the generation_validation table.

Usage:
    python3 generation_monitor.py              # Last 24h stats
    python3 generation_monitor.py --hours 72   # Last 72h
    python3 generation_monitor.py --failures   # Recent failures only

Place at: /opt/tower-echo-brain/scripts/generation_monitor.py
"""

import argparse
import json
import os
import sys

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "patrick")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_DATABASE = os.getenv("PG_DATABASE", "anime_production")


def get_connection():
    import psycopg2
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        user=PG_USER, password=PG_PASSWORD,
        dbname=PG_DATABASE, connect_timeout=10,
    )


def show_stats(hours: int = 24):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN validation_status = 'passed' THEN 1 ELSE 0 END) AS passed,
            SUM(CASE WHEN validation_status = 'failed' THEN 1 ELSE 0 END) AS failed,
            SUM(CASE WHEN validation_status = 'partial' THEN 1 ELSE 0 END) AS partial,
            SUM(CASE WHEN validation_status = 'no_output' THEN 1 ELSE 0 END) AS no_output,
            ROUND(AVG(quality_score)::numeric, 3) AS avg_quality,
            ROUND(AVG(generation_time_s)::numeric, 1) AS avg_time,
            SUM(CASE WHEN used_vhs_fallback THEN 1 ELSE 0 END) AS vhs_fallback
        FROM generation_validation
        WHERE validated_at > NOW() - INTERVAL '%s hours'
    """, (hours,))

    row = cur.fetchone()
    total, passed, failed, partial, no_output, avg_q, avg_t, vhs = row

    print(f"\n{'=' * 50}")
    print(f"  GENERATION STATS (Last {hours}h)")
    print(f"{'=' * 50}")

    if total == 0:
        print(f"\n  No generations recorded in the last {hours} hours.")
        conn.close()
        return

    pass_rate = (passed / total * 100) if total else 0

    print(f"\n  Total:      {total}")
    print(f"  ✅ Passed:   {passed} ({pass_rate:.0f}%)")
    print(f"  ❌ Failed:   {failed}")
    print(f"  ⚠️  Partial:  {partial}")
    print(f"  🚫 No output: {no_output}")
    print(f"  📊 Avg quality: {avg_q or 'N/A'}")
    print(f"  ⏱  Avg time:    {avg_t or 'N/A'}s")
    if vhs and vhs > 0:
        print(f"  🔄 VHS fallback: {vhs} ({vhs/total*100:.0f}%)")

    # Model breakdown
    cur.execute("""
        SELECT model_name, COUNT(*), 
               ROUND(AVG(quality_score)::numeric, 2)
        FROM generation_validation
        WHERE validated_at > NOW() - INTERVAL '%s hours'
          AND model_name IS NOT NULL
        GROUP BY model_name
        ORDER BY COUNT(*) DESC
    """, (hours,))

    models = cur.fetchall()
    if models:
        print(f"\n  Models:")
        for name, count, quality in models:
            print(f"    {name}: {count} gens (avg quality: {quality})")

    conn.close()
    print(f"{'=' * 50}")


def show_failures(limit: int = 10):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            comfyui_prompt_id,
            validation_status,
            model_name,
            workflow_used,
            total_images,
            passed_images,
            issues,
            used_vhs_fallback,
            validated_at
        FROM generation_validation
        WHERE validation_status IN ('failed', 'no_output')
        ORDER BY validated_at DESC
        LIMIT %s
    """, (limit,))

    rows = cur.fetchall()

    print(f"\n{'=' * 50}")
    print(f"  RECENT FAILURES ({len(rows)})")
    print(f"{'=' * 50}")

    if not rows:
        print(f"\n  No failures found. 🎉")
    else:
        for row in rows:
            pid, status, model, wf, total, passed, issues_json, vhs, ts = row
            issues = json.loads(issues_json) if issues_json else []
            print(f"\n  ❌ {pid[:12]}... ({status})")
            print(f"     Time: {ts}")
            print(f"     Model: {model or '?'}")
            if wf:
                print(f"     Workflow: {os.path.basename(wf)}")
            print(f"     Output: {passed}/{total} passed")
            if vhs:
                print(f"     VHS fallback: yes")
            if issues:
                for iss in issues[:3]:
                    print(f"     • {iss[:100]}")

    conn.close()
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation Pipeline Monitor")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--failures", action="store_true")
    parser.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    try:
        if args.failures:
            show_failures(args.limit)
        else:
            show_stats(args.hours)
    except ImportError:
        print("psycopg2 required: pip install psycopg2-binary --break-system-packages")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
