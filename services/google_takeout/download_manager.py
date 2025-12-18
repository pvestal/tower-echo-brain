#!/usr/bin/env python3
"""
Download Manager for Google Takeout Files
Handles downloads with resume capability, integrity checking, and progress tracking
"""

import os
import hashlib
import logging
import asyncio
import aiohttp
import aiofiles
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

@dataclass
class DownloadJob:
    """Represents a download job with metadata"""
    url: str
    local_path: str
    expected_size: Optional[int] = None
    expected_checksum: Optional[str] = None
    checksum_algorithm: str = 'sha256'
    archive_id: str = ''
    job_id: str = ''
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    bytes_downloaded: int = 0
    status: str = 'pending'  # pending, downloading, paused, completed, failed
    error_message: Optional[str] = None
    retry_count: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'url': self.url,
            'local_path': self.local_path,
            'expected_size': self.expected_size,
            'expected_checksum': self.expected_checksum,
            'checksum_algorithm': self.checksum_algorithm,
            'archive_id': self.archive_id,
            'job_id': self.job_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'bytes_downloaded': self.bytes_downloaded,
            'status': self.status,
            'error_message': self.error_message,
            'retry_count': self.retry_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DownloadJob':
        """Create from dictionary"""
        job = cls(
            url=data['url'],
            local_path=data['local_path'],
            expected_size=data.get('expected_size'),
            expected_checksum=data.get('expected_checksum'),
            checksum_algorithm=data.get('checksum_algorithm', 'sha256'),
            archive_id=data.get('archive_id', ''),
            job_id=data.get('job_id', ''),
            bytes_downloaded=data.get('bytes_downloaded', 0),
            status=data.get('status', 'pending'),
            error_message=data.get('error_message'),
            retry_count=data.get('retry_count', 0)
        )

        # Parse datetime fields
        for field in ['created_at', 'started_at', 'completed_at']:
            if data.get(field):
                setattr(job, field, datetime.fromisoformat(data[field]))

        return job


class DownloadManager:
    """Manages download operations with resume capability and integrity checking"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize download manager

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.base_path = Path(config['download']['base_path'])
        self.downloads_path = self.base_path / 'downloads'
        self.temp_path = self.base_path / 'temp'
        self.checksums_path = self.base_path / 'checksums'

        # Download settings
        self.chunk_size = config['download']['chunk_size']
        self.max_concurrent = config['download']['max_concurrent']
        self.temp_suffix = config['download']['temp_suffix']

        # Create directories
        for path in [self.downloads_path, self.temp_path, self.checksums_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Job tracking
        self.active_jobs: Dict[str, DownloadJob] = {}
        self.job_file = self.base_path / 'active_downloads.json'
        self.progress_callbacks: List[Callable] = []

        # Load existing jobs
        self._load_jobs()

        logger.info(f"Download manager initialized with base path: {self.base_path}")

    def _load_jobs(self) -> None:
        """Load active download jobs from disk"""
        if self.job_file.exists():
            try:
                with open(self.job_file, 'r') as f:
                    jobs_data = json.load(f)

                for job_data in jobs_data:
                    job = DownloadJob.from_dict(job_data)
                    self.active_jobs[job.job_id] = job

                logger.info(f"Loaded {len(self.active_jobs)} active download jobs")

            except Exception as e:
                logger.error(f"Failed to load download jobs: {e}")

    def _save_jobs(self) -> None:
        """Save active download jobs to disk"""
        try:
            jobs_data = [job.to_dict() for job in self.active_jobs.values()]

            with open(self.job_file, 'w') as f:
                json.dump(jobs_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save download jobs: {e}")

    def add_progress_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Add a progress callback function

        Args:
            callback: Function to call with job_id and progress data
        """
        self.progress_callbacks.append(callback)

    def _notify_progress(self, job_id: str, progress_data: Dict[str, Any]) -> None:
        """Notify all registered progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(job_id, progress_data)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")

    def create_download_job(self, url: str, archive_id: str, filename: Optional[str] = None) -> str:
        """
        Create a new download job

        Args:
            url: Download URL
            archive_id: Associated archive ID
            filename: Optional custom filename

        Returns:
            str: Job ID
        """
        # Generate job ID
        job_id = f"job_{int(time.time())}_{len(self.active_jobs)}"

        # Determine filename
        if not filename:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path) or f"takeout_{archive_id}.zip"

        local_path = str(self.downloads_path / filename)

        # Create job
        job = DownloadJob(
            url=url,
            local_path=local_path,
            archive_id=archive_id,
            job_id=job_id
        )

        self.active_jobs[job_id] = job
        self._save_jobs()

        logger.info(f"Created download job {job_id} for {filename}")
        return job_id

    async def download_file(self, job_id: str, resume: bool = True) -> bool:
        """
        Download a file with resume capability

        Args:
            job_id: Job ID to download
            resume: Whether to resume partial downloads

        Returns:
            bool: True if download successful
        """
        if job_id not in self.active_jobs:
            logger.error(f"Job {job_id} not found")
            return False

        job = self.active_jobs[job_id]
        job.status = 'downloading'
        job.started_at = datetime.utcnow()

        try:
            local_path = Path(job.local_path)
            temp_path = Path(str(local_path) + self.temp_suffix)

            # Check if partial file exists
            resume_from = 0
            if resume and temp_path.exists():
                resume_from = temp_path.stat().st_size
                job.bytes_downloaded = resume_from
                logger.info(f"Resuming download from {resume_from} bytes")

            async with aiohttp.ClientSession() as session:
                headers = {}
                if resume_from > 0:
                    headers['Range'] = f'bytes={resume_from}-'

                async with session.get(job.url, headers=headers) as response:
                    # Check response status
                    if resume_from > 0 and response.status == 206:
                        logger.info("Successfully resuming download with partial content")
                    elif resume_from == 0 and response.status == 200:
                        logger.info("Starting fresh download")
                    else:
                        raise Exception(f"Unexpected response status: {response.status}")

                    # Get total size
                    content_length = response.headers.get('Content-Length')
                    if content_length:
                        total_size = int(content_length)
                        if resume_from > 0:
                            total_size += resume_from
                        job.expected_size = total_size

                    # Open file for writing
                    mode = 'ab' if resume_from > 0 else 'wb'
                    async with aiofiles.open(temp_path, mode) as f:
                        hasher = hashlib.new(job.checksum_algorithm)

                        # If resuming, we need to hash the existing content
                        if resume_from > 0:
                            hasher = await self._hash_existing_file(temp_path, hasher)

                        # Download in chunks
                        async for chunk in response.content.iter_chunked(self.chunk_size):
                            await f.write(chunk)
                            hasher.update(chunk)

                            job.bytes_downloaded += len(chunk)

                            # Notify progress
                            progress = {
                                'bytes_downloaded': job.bytes_downloaded,
                                'total_size': job.expected_size,
                                'percentage': (job.bytes_downloaded / job.expected_size * 100) if job.expected_size else 0,
                                'status': 'downloading'
                            }
                            self._notify_progress(job_id, progress)

                    # Calculate final checksum
                    calculated_checksum = hasher.hexdigest()

                    # Verify checksum if provided
                    if job.expected_checksum and calculated_checksum != job.expected_checksum:
                        raise Exception(f"Checksum mismatch: expected {job.expected_checksum}, got {calculated_checksum}")

                    # Move temp file to final location
                    temp_path.rename(local_path)

                    # Save checksum
                    checksum_file = self.checksums_path / f"{local_path.name}.{job.checksum_algorithm}"
                    with open(checksum_file, 'w') as f:
                        f.write(f"{calculated_checksum}  {local_path.name}\n")

                    # Update job status
                    job.status = 'completed'
                    job.completed_at = datetime.utcnow()

                    logger.info(f"Successfully downloaded {local_path.name}")
                    return True

        except Exception as e:
            job.status = 'failed'
            job.error_message = str(e)
            job.retry_count += 1

            logger.error(f"Download failed for job {job_id}: {e}")

            # Notify failure
            progress = {
                'bytes_downloaded': job.bytes_downloaded,
                'total_size': job.expected_size,
                'percentage': 0,
                'status': 'failed',
                'error': str(e)
            }
            self._notify_progress(job_id, progress)

            return False

        finally:
            self._save_jobs()

    async def _hash_existing_file(self, file_path: Path, hasher: hashlib._Hash) -> hashlib._Hash:
        """Hash existing file content for resume capability"""
        async with aiofiles.open(file_path, 'rb') as f:
            while True:
                chunk = await f.read(self.chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher

    async def download_multiple(self, job_ids: List[str]) -> Dict[str, bool]:
        """
        Download multiple files concurrently

        Args:
            job_ids: List of job IDs to download

        Returns:
            Dict mapping job IDs to success status
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def download_with_semaphore(job_id: str) -> tuple:
            async with semaphore:
                success = await self.download_file(job_id)
                return job_id, success

        # Execute downloads concurrently
        tasks = [download_with_semaphore(job_id) for job_id in job_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        download_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Download task failed: {result}")
            else:
                job_id, success = result
                download_results[job_id] = success

        return download_results

    def verify_file_integrity(self, job_id: str) -> bool:
        """
        Verify the integrity of a downloaded file

        Args:
            job_id: Job ID to verify

        Returns:
            bool: True if file is valid
        """
        if job_id not in self.active_jobs:
            logger.error(f"Job {job_id} not found")
            return False

        job = self.active_jobs[job_id]

        if job.status != 'completed':
            logger.error(f"Job {job_id} is not completed")
            return False

        try:
            file_path = Path(job.local_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False

            # Calculate checksum
            hasher = hashlib.new(job.checksum_algorithm)
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)

            calculated_checksum = hasher.hexdigest()

            # Compare with stored checksum
            checksum_file = self.checksums_path / f"{file_path.name}.{job.checksum_algorithm}"
            if checksum_file.exists():
                with open(checksum_file, 'r') as f:
                    stored_checksum = f.read().strip().split()[0]

                if calculated_checksum == stored_checksum:
                    logger.info(f"File integrity verified for {file_path.name}")
                    return True
                else:
                    logger.error(f"Checksum mismatch for {file_path.name}")
                    return False
            else:
                logger.warning(f"No stored checksum found for {file_path.name}")
                return False

        except Exception as e:
            logger.error(f"Failed to verify file integrity: {e}")
            return False

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a download job

        Args:
            job_id: Job ID to check

        Returns:
            Dict containing job status or None
        """
        if job_id not in self.active_jobs:
            return None

        job = self.active_jobs[job_id]
        status = job.to_dict()

        # Add progress percentage
        if job.expected_size and job.bytes_downloaded:
            status['progress_percentage'] = (job.bytes_downloaded / job.expected_size) * 100
        else:
            status['progress_percentage'] = 0

        return status

    def pause_download(self, job_id: str) -> bool:
        """
        Pause a download job

        Args:
            job_id: Job ID to pause

        Returns:
            bool: True if paused successfully
        """
        if job_id not in self.active_jobs:
            return False

        job = self.active_jobs[job_id]

        if job.status == 'downloading':
            job.status = 'paused'
            self._save_jobs()
            logger.info(f"Paused download job {job_id}")
            return True

        return False

    def resume_download(self, job_id: str) -> bool:
        """
        Resume a paused download job

        Args:
            job_id: Job ID to resume

        Returns:
            bool: True if resumed successfully
        """
        if job_id not in self.active_jobs:
            return False

        job = self.active_jobs[job_id]

        if job.status == 'paused':
            job.status = 'pending'
            self._save_jobs()
            logger.info(f"Resumed download job {job_id}")
            return True

        return False

    def cancel_download(self, job_id: str) -> bool:
        """
        Cancel a download job

        Args:
            job_id: Job ID to cancel

        Returns:
            bool: True if cancelled successfully
        """
        if job_id not in self.active_jobs:
            return False

        job = self.active_jobs[job_id]

        # Remove temp file if it exists
        temp_path = Path(str(job.local_path) + self.temp_suffix)
        if temp_path.exists():
            temp_path.unlink()

        # Remove job
        del self.active_jobs[job_id]
        self._save_jobs()

        logger.info(f"Cancelled download job {job_id}")
        return True

    def cleanup_completed_jobs(self, older_than_days: int = 7) -> int:
        """
        Clean up completed download jobs

        Args:
            older_than_days: Remove jobs completed more than this many days ago

        Returns:
            int: Number of jobs cleaned up
        """
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        jobs_to_remove = []

        for job_id, job in self.active_jobs.items():
            if (job.status in ['completed', 'failed'] and
                job.completed_at and
                job.completed_at < cutoff_date):
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]

        if jobs_to_remove:
            self._save_jobs()

        logger.info(f"Cleaned up {len(jobs_to_remove)} completed jobs")
        return len(jobs_to_remove)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get download statistics

        Returns:
            Dict containing download statistics
        """
        stats = {
            'total_jobs': len(self.active_jobs),
            'pending': 0,
            'downloading': 0,
            'paused': 0,
            'completed': 0,
            'failed': 0,
            'total_bytes_downloaded': 0,
            'total_files_size': 0
        }

        for job in self.active_jobs.values():
            stats[job.status] += 1
            stats['total_bytes_downloaded'] += job.bytes_downloaded

            if job.expected_size:
                stats['total_files_size'] += job.expected_size

        return stats


def create_download_manager(config: Dict[str, Any]) -> DownloadManager:
    """
    Factory function to create DownloadManager

    Args:
        config: Configuration dictionary

    Returns:
        DownloadManager instance
    """
    return DownloadManager(config)


if __name__ == "__main__":
    # Example usage and testing
    import yaml
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    # Load configuration
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Test download manager
    manager = create_download_manager(config)

    print("Download Manager Statistics:", manager.get_statistics())