# TASK: Personal Data Indexing — Ingest Patrick's Digital History

## PREREQUISITES

- Document ingestion pipeline (Prompt 02) must be deployed
- Web search (Prompt 01) should be deployed

## CONTEXT

Echo Brain has 509K+ vectors but most are from automated ingestion of conversations and code. Patrick's broader digital life — Google data, important documents, reference manuals — isn't indexed. This prompt creates targeted ingesters for high-value personal data sources.

## PHASE 1: DISCOVER WHAT'S AVAILABLE

```bash
echo "=== Google Takeout Data ==="
find /home/patrick -maxdepth 4 -name "Takeout" -type d 2>/dev/null
find /home/patrick -maxdepth 4 -name "*.mbox" -type f 2>/dev/null | head -5
find /home/patrick -maxdepth 4 -name "MyActivity*" -type f 2>/dev/null | head -5

echo ""
echo "=== Documents & Manuals ==="
find /home/patrick/Documents -type f \( -name "*.pdf" -o -name "*.docx" \) 2>/dev/null | head -20
find /home/patrick/Downloads -type f \( -name "*.pdf" -o -name "*.docx" \) 2>/dev/null | head -20

echo ""
echo "=== Claude Conversations ==="
find /home/patrick -maxdepth 4 -name "*claude*" -type f 2>/dev/null | head -10
ls /opt/tower-echo-brain/scripts/*claude* 2>/dev/null
ls /opt/tower-echo-brain/scripts/*ingest* 2>/dev/null

echo ""
echo "=== Existing Ingestion Scripts ==="
ls /opt/tower-echo-brain/scripts/
grep -l "ingest\|import\|index" /opt/tower-echo-brain/scripts/*.py 2>/dev/null

echo ""
echo "=== RV / Victron Documentation ==="
find /home/patrick -maxdepth 5 -type f \( -iname "*victron*" -o -iname "*sundowner*" -o -iname "*tundra*" -o -iname "*rv*" -o -iname "*trailblazer*" \) 2>/dev/null | head -20

echo ""
echo "=== Current Qdrant Stats ==="
curl -s http://localhost:6333/collections/echo_memory | python3 -c "
import sys, json; d=json.load(sys.stdin)
r = d.get('result', {})
print(f'Vectors: {r.get(\"vectors_count\", \"?\")}')
print(f'Points: {r.get(\"points_count\", \"?\")}')
"
```

## PHASE 2: BATCH DOCUMENT INGESTION SCRIPT

Create `/opt/tower-echo-brain/scripts/ingest_documents.py`

This script uses the Document Service (from Prompt 02) to batch-ingest files:

```python
#!/usr/bin/env python3
"""
Batch document ingester for Echo Brain.
Scans specified directories for documents and ingests them via the API.

Usage:
    python ingest_documents.py /path/to/documents
    python ingest_documents.py /home/patrick/Documents --extensions pdf,docx
    python ingest_documents.py --dry-run /path/to/check
"""

import argparse
import asyncio
import hashlib
import httpx
from pathlib import Path

ECHO_BRAIN_URL = "http://localhost:8309"
SUPPORTED_EXTENSIONS = {
    '.pdf', '.docx', '.doc', '.xlsx', '.csv', 
    '.txt', '.md', '.html', '.json', '.yaml', '.yml'
}

async def ingest_directory(directory: str, extensions: set = None, dry_run: bool = False):
    """Scan directory and ingest all supported documents."""
    path = Path(directory)
    if not path.exists():
        print(f"ERROR: {directory} does not exist")
        return
    
    exts = extensions or SUPPORTED_EXTENSIONS
    files = []
    for ext in exts:
        files.extend(path.rglob(f"*{ext}"))
    
    print(f"Found {len(files)} documents in {directory}")
    
    if dry_run:
        for f in files:
            print(f"  [DRY RUN] Would ingest: {f} ({f.stat().st_size / 1024:.1f}KB)")
        return
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        success = 0
        skipped = 0
        failed = 0
        
        for f in files:
            try:
                resp = await client.post(
                    f"{ECHO_BRAIN_URL}/api/echo/documents/ingest-path",
                    json={"path": str(f.absolute())}
                )
                data = resp.json()
                
                if resp.status_code == 200:
                    chunks = data.get('chunk_count', '?')
                    print(f"  ✓ {f.name} — {chunks} chunks")
                    success += 1
                elif 'duplicate' in str(data).lower() or 'already' in str(data).lower():
                    print(f"  ⊘ {f.name} — already ingested")
                    skipped += 1
                else:
                    print(f"  ✗ {f.name} — {data}")
                    failed += 1
            except Exception as e:
                print(f"  ✗ {f.name} — ERROR: {e}")
                failed += 1
        
        print(f"\nResults: {success} ingested, {skipped} skipped, {failed} failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ingest documents into Echo Brain")
    parser.add_argument("directory", help="Directory to scan for documents")
    parser.add_argument("--extensions", help="Comma-separated extensions (e.g. pdf,docx)", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested without doing it")
    args = parser.parse_args()
    
    exts = {f".{e.strip('.')}" for e in args.extensions.split(",")} if args.extensions else None
    asyncio.run(ingest_directory(args.directory, exts, args.dry_run))
```

## PHASE 3: GOOGLE TAKEOUT PARSER (if data exists)

**Only build this if Phase 1 discovery finds Google Takeout data.**

Create `/opt/tower-echo-brain/scripts/ingest_google_takeout.py`

Handles:
- **Gmail (mbox):** Parse emails, extract subject + body + date, chunk by email
- **Chrome bookmarks:** Extract title + URL + folder hierarchy  
- **YouTube history:** Watch history with titles and timestamps
- **Drive metadata:** File names, shared status, last modified

Each item gets a payload with `type: "google_takeout"`, `subtype: "email|bookmark|youtube|drive"`, and appropriate metadata.

## PHASE 4: PRIORITY DOCUMENTS TO INGEST

After the tooling is built, ingest these high-value document categories:

```bash
# 1. RV manuals and Victron documentation (if found in Phase 1)
python scripts/ingest_documents.py /path/to/rv-docs --extensions pdf

# 2. Any technical reference PDFs
python scripts/ingest_documents.py /home/patrick/Documents --extensions pdf,docx --dry-run
# Review the dry run, then run for real

# 3. Project documentation
python scripts/ingest_documents.py /opt/tower-echo-brain/docs --extensions md
```

## PHASE 5: VERIFY

```bash
echo "=== Qdrant Before vs After ==="
curl -s http://localhost:6333/collections/echo_memory | python3 -c "
import sys, json; d=json.load(sys.stdin)
print(f'Total vectors: {d[\"result\"][\"vectors_count\"]}')
"

echo ""
echo "=== Ingested Documents ==="
curl -s http://localhost:8309/api/echo/documents | python3 -c "
import sys, json; docs=json.load(sys.stdin)
print(f'Total documents: {len(docs)}')
for d in docs[:10]:
    print(f'  {d[\"filename\"]} — {d[\"chunk_count\"]} chunks, {d[\"document_type\"]}')
"

echo ""
echo "=== Test Query Against Ingested Content ==="
curl -s -X POST http://localhost:8309/api/echo/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What documents have been ingested into Echo Brain?"}' | python3 -c "
import sys, json; d=json.load(sys.stdin)
print(d.get('response', '')[:500])
"
```

## CONSTRAINTS

- Use the Document Service API from Prompt 02 — do NOT directly write to Qdrant
- Respect file permissions — only ingest files the echo user can read
- Skip files larger than 50MB
- Log everything for audit trail
- Handle errors gracefully — one bad file should not kill the whole batch

## DONE WHEN

- [ ] Batch ingestion script works for directories of documents
- [ ] Google Takeout parser created (if data exists)
- [ ] At least one batch of real documents successfully ingested
- [ ] Vector count in Qdrant has increased
- [ ] Queries against ingested content return relevant results
