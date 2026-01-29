#!/usr/bin/env python3
"""
Conversation ingestion for Echo Brain learning pipeline
Processes conversation files and ingests them into Qdrant
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import subprocess

def main():
    """Run the conversation ingestion script"""
    print("🔄 Starting Claude conversation indexing...")

    # Call the conversation ingestion script
    ingestion_script = Path("/opt/tower-echo-brain/scripts/conversation_ingestion.py")

    if ingestion_script.exists():
        try:
            # Run the conversation ingestion script using the venv Python
            result = subprocess.run(
                ["/opt/tower-echo-brain/venv/bin/python", str(ingestion_script)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            # Print output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

            # Check result
            if result.returncode == 0:
                print("✅ Ingestion completed successfully")

                # Log successful run
                log_dir = Path("/opt/tower-echo-brain/logs")
                log_dir.mkdir(exist_ok=True)
                with open(log_dir / "last_ingestion.log", "w") as f:
                    f.write(f"Ingestion completed at {datetime.now()}\n")
                    f.write(f"Exit code: {result.returncode}\n")
                    if result.stdout:
                        f.write(f"Output:\n{result.stdout}\n")
            else:
                print(f"❌ Ingestion failed with code {result.returncode}")
                sys.exit(result.returncode)

        except subprocess.TimeoutExpired:
            print("❌ Ingestion timed out after 10 minutes")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error running ingestion: {e}")
            sys.exit(1)
    else:
        # Fallback to checking for conversation files
        conversation_dir = Path("/home/patrick/.claude/projects")

        if conversation_dir.exists():
            # Find all JSONL files recursively
            conversation_files = list(conversation_dir.rglob("*.jsonl"))
            print(f"📁 Found {len(conversation_files)} conversation files")
            print(f"⚠️  Conversation ingestion script not found at {ingestion_script}")
            print("⚠️  Files found but not processed - please check conversation_ingestion.py")
        else:
            print(f"⚠️  Conversation directory not found: {conversation_dir}")

if __name__ == "__main__":
    main()