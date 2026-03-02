# TASK: Build Document Ingestion Pipeline for Echo Brain

## PREREQUISITE

Web search (Prompt 01) should be deployed first, but this can run in parallel if needed.

## CONTEXT

Echo Brain has 509K+ vectors in Qdrant but NO ability to ingest new documents. Patrick has RV manuals, datasheets, PDFs, reference documents that should be queryable. This prompt builds the document ingestion pipeline.

## SYSTEM FACTS

```
Echo Brain:     /opt/tower-echo-brain/
Qdrant:         localhost:6333, collection=echo_memory, 768D nomic-embed-text
Embeddings:     nomic-embed-text via Ollama at localhost:11434/api/embed
Embedding svc:  /opt/tower-echo-brain/src/services/embedding_service.py
Python venv:    /opt/tower-echo-brain/venv/
Max context:    8192 tokens (nomic-embed-text)
```

## PHASE 1: INSTALL DEPENDENCIES

```bash
# Install document parsing libraries
/opt/tower-echo-brain/venv/bin/pip install \
  python-docx \
  pymupdf \
  python-pptx \
  openpyxl \
  markdown \
  beautifulsoup4 \
  chardet

# Verify
/opt/tower-echo-brain/venv/bin/python -c "
import fitz; print(f'PyMuPDF {fitz.version}')  # PDF parsing
import docx; print('python-docx OK')
import openpyxl; print('openpyxl OK')
import bs4; print('beautifulsoup4 OK')
"
```

**NOTE:** We're using `pymupdf` (fitz) instead of `unstructured` because it's lighter (~20MB vs ~500MB+), has no system dependencies, and handles PDFs well. python-docx for Word, openpyxl for Excel.

## PHASE 2: BUILD DOCUMENT SERVICE

### Step 1: Study existing embedding service

```bash
cat /opt/tower-echo-brain/src/services/embedding_service.py
# Understand how embeddings are generated and stored
grep -rn "embed\|qdrant\|upsert\|QdrantClient" /opt/tower-echo-brain/src/services/ --include="*.py" | head -20
```

### Step 2: Create document service

Create `/opt/tower-echo-brain/src/services/document_service.py`

Requirements:

**File type parsers:**
- PDF → text via PyMuPDF (preserves page numbers)
- DOCX → text via python-docx (preserves headings)
- XLSX/CSV → text via openpyxl/csv (preserves column headers)
- Markdown → text (strip formatting)
- HTML → text via BeautifulSoup
- Plain text (.txt, .log)
- Code files (.py, .js, .ts, .yaml, .json, .sh) — preserve structure

**Chunking strategy:**
- Semantic chunking: split on headings/sections first, then by paragraph if chunks are too large
- Target chunk size: ~1200 words (fits well within nomic-embed-text 8192 token context)
- Overlap: 150 words between chunks
- Preserve metadata per chunk: source filename, page number, section heading, chunk index

**Embedding and storage:**
- Use the existing embedding service to generate 768D vectors
- Store in `echo_memory` collection (or a new `documents` collection if the codebase uses multiple collections)
- Payload must include: `text`, `type: "document"`, `source` (filename), `page`, `section`, `document_type` (extension), `created_at`

**Document registry:**
- Track ingested documents in PostgreSQL (table: `ingested_documents`)
- Schema: id, filename, file_hash (SHA256), file_size, document_type, chunk_count, pages, ingested_at, status
- Prevent duplicate ingestion by checking file hash

```python
# Pseudocode structure — adapt to existing patterns
class DocumentService:
    async def ingest_file(self, file_path: str, collection: str = "echo_memory") -> IngestResult
    async def ingest_upload(self, upload: UploadFile, collection: str = "echo_memory") -> IngestResult
    async def list_documents(self) -> list[DocumentInfo]
    async def delete_document(self, doc_id: int) -> bool
    
    def _parse_pdf(self, path: str) -> list[DocumentChunk]
    def _parse_docx(self, path: str) -> list[DocumentChunk]
    def _parse_xlsx(self, path: str) -> list[DocumentChunk]
    def _parse_text(self, path: str) -> list[DocumentChunk]
    def _parse_code(self, path: str) -> list[DocumentChunk]
    
    def _semantic_chunk(self, text: str, max_words: int = 1200, overlap: int = 150) -> list[str]
```

## PHASE 3: DATABASE TABLE

### Create the ingested_documents table

```bash
sudo -u postgres psql echo_brain -c "
CREATE TABLE IF NOT EXISTS ingested_documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(500) NOT NULL,
    file_hash VARCHAR(64) NOT NULL UNIQUE,
    file_size BIGINT,
    document_type VARCHAR(20),
    chunk_count INTEGER DEFAULT 0,
    pages INTEGER DEFAULT 0,
    vector_ids TEXT[],  -- Array of Qdrant point IDs for cleanup
    status VARCHAR(20) DEFAULT 'processing',
    error_message TEXT,
    ingested_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ingested_docs_hash ON ingested_documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_ingested_docs_status ON ingested_documents(status);
"
```

## PHASE 4: API ENDPOINTS

Create router following existing patterns. Endpoints:

```
POST   /api/echo/documents/upload          — Upload and ingest a single file (multipart/form-data)
POST   /api/echo/documents/ingest-path     — Ingest a file from a local path on Tower
POST   /api/echo/documents/ingest-batch    — Ingest multiple local files (array of paths)
GET    /api/echo/documents                 — List all ingested documents
GET    /api/echo/documents/{doc_id}        — Get document details
DELETE /api/echo/documents/{doc_id}        — Remove document and its vectors from Qdrant
GET    /api/echo/documents/search?q=...    — Search within ingested documents only
```

**The `ingest-path` endpoint is critical** — it allows Claude Code to ingest files directly from the Tower filesystem without uploading through HTTP. Example:

```bash
curl -X POST http://localhost:8309/api/echo/documents/ingest-path \
  -H "Content-Type: application/json" \
  -d '{"path": "/home/patrick/Documents/rv-manual.pdf"}'
```

## PHASE 5: MCP TOOL

Add to the existing MCP server:

```python
@mcp.tool()
async def ingest_document(file_path: str) -> str:
    """Ingest a local file into Echo Brain's memory.
    
    Supports: PDF, DOCX, XLSX, CSV, TXT, MD, HTML, Python, JS, TS, YAML, JSON
    
    Args:
        file_path: Absolute path to the file on Tower
    """
    # Call /api/echo/documents/ingest-path
    # Return summary: "Ingested {filename}: {chunk_count} chunks, {pages} pages"
```

## PHASE 6: VERIFY

```bash
echo "=== TEST 1: Create test document ==="
cat > /tmp/test-doc.md << 'EOF'
# Echo Brain Test Document

## Section One: Overview
This is a test document for verifying the document ingestion pipeline.
It contains multiple sections with different content to test semantic chunking.

## Section Two: Technical Details
The Echo Brain system uses nomic-embed-text for embeddings with 768 dimensions.
Vectors are stored in Qdrant at localhost:6333 in the echo_memory collection.

## Section Three: RV Systems
The 2022 Toyota Tundra 1794 Edition tows a 2021 Sundowner Trailblazer 2286TB.
The RV uses LiFePO4 batteries with Victron MultiPlus inverter and SmartShunt monitoring.
EOF

echo "=== TEST 2: Ingest via API ==="
curl -s -X POST http://localhost:8309/api/echo/documents/ingest-path \
  -H "Content-Type: application/json" \
  -d '{"path": "/tmp/test-doc.md"}' | python3 -m json.tool

echo ""
echo "=== TEST 3: List documents ==="
curl -s http://localhost:8309/api/echo/documents | python3 -m json.tool

echo ""
echo "=== TEST 4: Search within documents ==="
curl -s "http://localhost:8309/api/echo/documents/search?q=Victron+battery" | python3 -c "
import sys, json; d=json.load(sys.stdin)
results = d.get('results', [])
print(f'PASS: {len(results)} results found') if results else print('FAIL: no results')
for r in results[:3]:
    print(f'  Score: {r.get(\"score\",0):.3f} | {r.get(\"source\",\"?\")} | {r.get(\"text\",\"\")[:100]}')
"

echo ""
echo "=== TEST 5: Ask about ingested content ==="
curl -s -X POST http://localhost:8309/api/echo/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What embedding model does the test document say Echo Brain uses?"}' | python3 -c "
import sys, json; d=json.load(sys.stdin)
resp = d.get('response', '')
print(f'PASS') if 'nomic' in resp.lower() or '768' in resp else print(f'CHECK: {resp[:200]}')
"

echo ""
echo "=== TEST 6: Duplicate prevention ==="
curl -s -X POST http://localhost:8309/api/echo/documents/ingest-path \
  -H "Content-Type: application/json" \
  -d '{"path": "/tmp/test-doc.md"}' | python3 -c "
import sys, json; d=json.load(sys.stdin)
print(f'PASS: duplicate detected') if 'duplicate' in str(d).lower() or 'already' in str(d).lower() else print(f'CHECK: {d}')
"

echo ""
echo "=== TEST 7: Delete document ==="
DOC_ID=$(curl -s http://localhost:8309/api/echo/documents | python3 -c "import sys,json; docs=json.load(sys.stdin); print(docs[0]['id'] if docs else '')" 2>/dev/null)
if [ -n "$DOC_ID" ]; then
  curl -s -X DELETE "http://localhost:8309/api/echo/documents/$DOC_ID" | python3 -m json.tool
  echo "PASS: Document deleted"
else
  echo "SKIP: No document to delete"
fi

# Clean up
rm -f /tmp/test-doc.md
```

## CONSTRAINTS

- Use PyMuPDF for PDF (not pypdf2, not unstructured)
- Use the EXISTING embedding service — do not create a new one
- Store vectors in the EXISTING echo_memory collection unless the codebase already uses multiple collections
- Match existing code patterns (imports, logging, error handling)
- All new files go in the existing directory structure
- Do NOT change the embedding model or vector dimensions

## DONE WHEN

- [ ] Document service created with parsers for PDF, DOCX, XLSX, CSV, TXT, MD, code
- [ ] `ingested_documents` table created in PostgreSQL
- [ ] Upload endpoint works (multipart file upload)
- [ ] Local path ingestion works (`ingest-path`)
- [ ] Duplicate prevention by file hash
- [ ] Document deletion removes vectors from Qdrant
- [ ] MCP `ingest_document` tool registered
- [ ] All 7 verification tests pass
- [ ] Service restarted and healthy
