from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import asyncpg
import os
import hashlib
import subprocess
import yaml
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])

# Database connection configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "echo_brain",
    "user": "patrick",
    "password": os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")
}

MANIFEST_DIR = Path("/opt/tower-echo-brain/model-manifests")

# Pydantic models
class ModelInfo(BaseModel):
    name: str
    display_name: Optional[str]
    category: Optional[str]
    file_path: str
    file_size_bytes: Optional[int]
    status: str
    last_verified: Optional[datetime]
    metadata: Optional[dict]

class ModelDownloadRequest(BaseModel):
    name: str
    force: bool = False

class ModelVerifyRequest(BaseModel):
    names: Optional[List[str]] = None  # None = verify all

async def get_db():
    """Get database connection"""
    return await asyncpg.connect(**DB_CONFIG)

async def log_operation(model_name: str, operation: str, status: str, message: str, details: dict = None):
    """Log model operations to database"""
    conn = await get_db()
    try:
        await conn.execute("""
            INSERT INTO tower_model_logs (model_name, operation, status, message, details)
            VALUES ($1, $2, $3, $4, $5)
        """, model_name, operation, status, message, json.dumps(details or {}))
    except Exception as e:
        logger.error(f"Failed to log operation: {e}")
    finally:
        await conn.close()

@router.get("/logs")
async def get_model_logs(model_name: str = None, limit: int = 50):
    """Get model operation logs"""
    conn = await get_db()
    try:
        if model_name:
            rows = await conn.fetch("""
                SELECT * FROM tower_model_logs
                WHERE model_name = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, model_name, limit)
        else:
            rows = await conn.fetch("""
                SELECT * FROM tower_model_logs
                ORDER BY created_at DESC
                LIMIT $1
            """, limit)
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get model logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()

@router.get("/downloads/status")
async def get_download_status():
    """Get status of all downloads"""
    conn = await get_db()
    try:
        rows = await conn.fetch("""
            SELECT * FROM tower_model_downloads
            ORDER BY created_at DESC
            LIMIT 20
        """)
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get download status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()

@router.get("")
async def list_models(category: str = None, status: str = None):
    """List all registered models with optional filters"""
    conn = await get_db()
    try:
        query = "SELECT * FROM tower_models WHERE 1=1"
        params = []

        if category:
            params.append(category)
            query += f" AND category = ${len(params)}"
        if status:
            params.append(status)
            query += f" AND status = ${len(params)}"

        query += " ORDER BY category, name"

        rows = await conn.fetch(query, *params)
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()

@router.get("/{name}/path")
async def get_model_path(name: str):
    """Get just the file path for a model - simple endpoint for scripts"""
    conn = await get_db()
    try:
        row = await conn.fetchrow(
            "SELECT file_path, status FROM tower_models WHERE name = $1", name
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
        if row['status'] == 'missing':
            raise HTTPException(status_code=404, detail=f"Model '{name}' is missing from disk")
        return {"path": row['file_path']}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model path for {name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()

@router.get("/{name}")
async def get_model(name: str):
    """Get model info by name - returns path for scripts/Claude Code"""
    conn = await get_db()
    try:
        row = await conn.fetchrow(
            "SELECT * FROM tower_models WHERE name = $1", name
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
        return dict(row)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model {name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()

@router.post("/verify")
async def verify_models(request: ModelVerifyRequest, background_tasks: BackgroundTasks):
    """Verify model files exist and match expected sizes"""
    conn = await get_db()
    try:
        if request.names:
            rows = await conn.fetch(
                "SELECT * FROM tower_models WHERE name = ANY($1)", request.names
            )
        else:
            rows = await conn.fetch("SELECT * FROM tower_models")

        results = []
        for row in rows:
            model_name = row['name']
            file_path = row['file_path']
            expected_size = row['file_size_bytes']

            if not os.path.exists(file_path):
                status = "missing"
                message = "File not found"
            else:
                actual_size = os.path.getsize(file_path)
                if expected_size and actual_size < expected_size * 0.95:
                    status = "corrupted"
                    message = f"Size mismatch: {actual_size} vs expected {expected_size}"
                else:
                    status = "healthy"
                    message = f"OK ({actual_size} bytes)"

            # Update status in database
            await conn.execute("""
                UPDATE tower_models
                SET status = $1, last_verified = NOW(), updated_at = NOW()
                WHERE name = $2
            """, status, model_name)

            await log_operation(model_name, "verify", status, message)

            results.append({
                "name": model_name,
                "status": status,
                "message": message
            })

        return {"results": results}
    except Exception as e:
        logger.error(f"Failed to verify models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()

@router.post("/download")
async def queue_download(request: ModelDownloadRequest, background_tasks: BackgroundTasks):
    """Queue a model for download"""
    conn = await get_db()
    try:
        row = await conn.fetchrow(
            "SELECT * FROM tower_models WHERE name = $1", request.name
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"Model '{request.name}' not found in registry")

        if not row['source_url']:
            raise HTTPException(status_code=400, detail=f"Model '{request.name}' has no source URL")

        if row['status'] == 'healthy' and not request.force:
            return {"status": "skipped", "message": "Model already healthy. Use force=true to re-download."}

        # Queue the download
        await conn.execute("""
            INSERT INTO tower_model_downloads (model_name, status, total_bytes)
            VALUES ($1, 'pending', $2)
            ON CONFLICT DO NOTHING
        """, request.name, row['file_size_bytes'])

        await log_operation(request.name, "download_queued", "pending", f"Download queued from {row['source_url']}")

        # Start download in background
        background_tasks.add_task(download_model, request.name, row['source_url'], row['file_path'])

        return {"status": "queued", "message": f"Download started for {request.name}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue download for {request.name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()

async def download_model(name: str, url: str, dest_path: str):
    """Background task to download a model"""
    conn = await get_db()
    try:
        await conn.execute("""
            UPDATE tower_model_downloads
            SET status = 'downloading', started_at = NOW()
            WHERE model_name = $1 AND status = 'pending'
        """, name)

        await conn.execute("""
            UPDATE tower_models SET status = 'downloading' WHERE name = $1
        """, name)

        await log_operation(name, "download_start", "downloading", f"Downloading from {url}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Use wget with resume support
        result = subprocess.run(
            ["wget", "-c", "-O", dest_path, url],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )

        if result.returncode == 0:
            await conn.execute("""
                UPDATE tower_model_downloads
                SET status = 'complete', completed_at = NOW()
                WHERE model_name = $1
            """, name)

            await conn.execute("""
                UPDATE tower_models SET status = 'healthy', last_verified = NOW() WHERE name = $1
            """, name)

            await log_operation(name, "download_complete", "healthy", f"Downloaded to {dest_path}")
        else:
            error_msg = result.stderr[:500] if result.stderr else "Download failed"
            await conn.execute("""
                UPDATE tower_model_downloads
                SET status = 'failed', error_message = $2
                WHERE model_name = $1
            """, name, error_msg)

            await conn.execute("""
                UPDATE tower_models SET status = 'failed' WHERE name = $1
            """, name)

            await log_operation(name, "download_fail", "failed", error_msg)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Download failed for {name}: {error_msg}")
        await log_operation(name, "download_fail", "failed", error_msg)
        try:
            await conn.execute("""
                UPDATE tower_models SET status = 'failed' WHERE name = $1
            """, name)
        except:
            pass
    finally:
        await conn.close()

@router.get("/downloads/status")
async def get_download_status():
    """Get status of all downloads"""
    conn = await get_db()
    try:
        rows = await conn.fetch("""
            SELECT * FROM tower_model_downloads
            ORDER BY created_at DESC
            LIMIT 20
        """)
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get download status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()

@router.get("/logs")
async def get_model_logs(model_name: str = None, limit: int = 50):
    """Get model operation logs"""
    conn = await get_db()
    try:
        if model_name:
            rows = await conn.fetch("""
                SELECT * FROM tower_model_logs
                WHERE model_name = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, model_name, limit)
        else:
            rows = await conn.fetch("""
                SELECT * FROM tower_model_logs
                ORDER BY created_at DESC
                LIMIT $1
            """, limit)
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get model logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()

@router.post("/sync-manifests")
async def sync_manifests():
    """Load all YAML manifests into database"""
    conn = await get_db()
    try:
        results = []

        for manifest_file in MANIFEST_DIR.glob("*.yaml"):
            with open(manifest_file) as f:
                manifest = yaml.safe_load(f)

            for model_name, model_info in manifest.get('models', {}).items():
                # Check if file exists
                file_path = model_info.get('path', '')
                if os.path.exists(file_path):
                    actual_size = os.path.getsize(file_path)
                    expected_size = model_info.get('size_bytes')
                    if expected_size and actual_size < expected_size * 0.95:
                        status = 'corrupted'
                    else:
                        status = 'healthy'
                else:
                    status = 'missing'

                # Upsert model
                await conn.execute("""
                    INSERT INTO tower_models (name, display_name, category, file_path, file_size_bytes, source_url, status, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (name) DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        category = EXCLUDED.category,
                        file_path = EXCLUDED.file_path,
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        source_url = EXCLUDED.source_url,
                        status = EXCLUDED.status,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                """,
                    model_name,
                    model_info.get('display_name'),
                    model_info.get('category'),
                    file_path,
                    model_info.get('size_bytes'),
                    model_info.get('source_url'),
                    status,
                    json.dumps(model_info.get('metadata', {}))
                )

                results.append({"name": model_name, "status": status})

        # Handle dependencies
        for manifest_file in MANIFEST_DIR.glob("*.yaml"):
            with open(manifest_file) as f:
                manifest = yaml.safe_load(f)

            for model_name, deps in manifest.get('dependencies', {}).items():
                for dep in deps:
                    await conn.execute("""
                        INSERT INTO tower_model_dependencies (model_name, depends_on)
                        VALUES ($1, $2)
                        ON CONFLICT DO NOTHING
                    """, model_name, dep)

        return {"synced": len(results), "models": results}
    except Exception as e:
        logger.error(f"Failed to sync manifests: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()

@router.post("/sync-ollama")
async def sync_ollama():
    """Sync Ollama models into registry"""
    conn = await get_db()
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="Failed to list Ollama models")

        synced = []
        lines = result.stdout.strip().split('\n')[1:]  # Skip header

        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                size_str = parts[2] if len(parts) > 2 else "0"

                # Convert size string (e.g., "4.7 GB") to bytes
                size_bytes = 0
                if "GB" in size_str:
                    size_bytes = int(float(size_str.replace("GB", "").strip()) * 1024 * 1024 * 1024)
                elif "MB" in size_str:
                    size_bytes = int(float(size_str.replace("MB", "").strip()) * 1024 * 1024)

                model_name = f"ollama/{name}"

                await conn.execute("""
                    INSERT INTO tower_models (name, display_name, category, file_path, file_size_bytes, status)
                    VALUES ($1, $2, 'ollama', $3, $4, 'healthy')
                    ON CONFLICT (name) DO UPDATE SET
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        status = 'healthy',
                        last_verified = NOW(),
                        updated_at = NOW()
                """, model_name, name, f"ollama://{name}", size_bytes)

                synced.append(model_name)

        return {"synced": len(synced), "models": synced}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to sync Ollama models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()