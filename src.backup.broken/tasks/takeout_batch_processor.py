#!/usr/bin/env python3
"""
Google Takeout Comprehensive Batch Processor
Processes 786GB of personal data for Echo persona training
"""

import asyncio
import asyncpg
import mailbox
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TakeoutBatchProcessor:
    def __init__(self):
        self.db_url = "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/echo_brain"
        self.takeout_base = Path("/mnt/10TB2/Google_Takeout_2025/Takeout")
        self.mbox_path = self.takeout_base / "Mail" / "All mail Including Spam and Trash.mbox"
        self.photos_path = self.takeout_base / "Google Photos"

        self.batch_size = 100  # Process 100 files at a time
        self.session_id = None

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def start_processing_session(self, conn, mode: str = "batch") -> int:
        """Start a new processing session"""
        session_id = await conn.fetchval("""
            INSERT INTO takeout_processing_sessions (processing_mode, metadata)
            VALUES ($1, $2)
            RETURNING id
        """, mode, json.dumps({"start_time": datetime.now().isoformat()}))

        logger.info(f"🚀 Started processing session {session_id} ({mode} mode)")
        return session_id

    async def update_session_stats(self, conn, files_count: int, insights_count: int):
        """Update session statistics"""
        await conn.execute("""
            UPDATE takeout_processing_sessions
            SET files_processed = files_processed + $1,
                insights_extracted = insights_extracted + $2
            WHERE id = $3
        """, files_count, insights_count, self.session_id)

    async def end_processing_session(self, conn):
        """Mark session as completed"""
        await conn.execute("""
            UPDATE takeout_processing_sessions
            SET session_end = CURRENT_TIMESTAMP,
                status = 'completed'
            WHERE id = $1
        """, self.session_id)

        logger.info(f"✅ Completed processing session {self.session_id}")

    async def file_already_processed(self, conn, file_hash: str) -> bool:
        """Check if file was already processed"""
        exists = await conn.fetchval("""
            SELECT EXISTS(SELECT 1 FROM takeout_files_processed WHERE file_hash = $1)
        """, file_hash)
        return exists

    async def mark_file_processed(self, conn, file_path: str, file_type: str,
                                  file_hash: str, file_size: int,
                                  processing_time_ms: int, insights_count: int):
        """Mark file as processed in database"""
        await conn.execute("""
            INSERT INTO takeout_files_processed
            (file_path, file_type, file_hash, file_size_bytes, processing_time_ms, insights_extracted)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, file_path, file_type, file_hash, file_size, processing_time_ms, insights_count)

    async def store_insight(self, conn, file_id: int, insight_type: str,
                           entity_name: Optional[str], entity_value: Dict,
                           confidence: float, context: str = ""):
        """Store an extracted insight"""
        await conn.execute("""
            INSERT INTO takeout_insights
            (file_id, insight_type, entity_name, entity_value, confidence_score, context)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, file_id, insight_type, entity_name, json.dumps(entity_value), confidence, context)

    async def upsert_person(self, conn, name: str, relationship_type: str = "unknown",
                           seen_date: datetime = None):
        """Add or update person in knowledge graph"""
        seen_date = seen_date or datetime.now()

        await conn.execute("""
            INSERT INTO takeout_people (name, relationship_type, first_seen, last_seen, total_mentions, photo_count, email_count)
            VALUES ($1, $2, $3, $3, 1, 0, 0)
            ON CONFLICT (name) DO UPDATE
            SET total_mentions = takeout_people.total_mentions + 1,
                last_seen = GREATEST(takeout_people.last_seen, $3)
        """, name, relationship_type, seen_date.date())

    async def upsert_location(self, conn, name: str, lat: float = None, lon: float = None,
                             visited_date: datetime = None):
        """Add or update location in knowledge graph"""
        visited_date = visited_date or datetime.now()

        await conn.execute("""
            INSERT INTO takeout_locations (name, latitude, longitude, first_visited, last_visited, visit_count, photo_count)
            VALUES ($1, $2, $3, $4, $4, 1, 0)
            ON CONFLICT (name, latitude, longitude) DO UPDATE
            SET visit_count = takeout_locations.visit_count + 1,
                last_visited = GREATEST(takeout_locations.last_visited, $4)
        """, name, lat, lon, visited_date.date())

    async def process_email(self, conn, message, index: int) -> Dict[str, Any]:
        """Process a single email message"""
        insights = []

        try:
            # Extract email body
            body = ""
            if message.is_multipart():
                for part in message.walk():
                    if part.get_content_type() == "text/plain":
                        try:
                            body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            break
                        except:
                            continue
            else:
                try:
                    body = message.get_payload(decode=True).decode('utf-8', errors='ignore')
                except:
                    pass

            if not body or len(body) < 10:
                return {"insights": [], "skip": True}

            # Extract sender and recipients
            sender = message.get('From', '')
            recipients = message.get('To', '')
            subject = message.get('Subject', '')
            date_str = message.get('Date', '')

            # Parse people from email
            email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
            people = re.findall(email_pattern, f"{sender} {recipients}")

            for person_email in people[:10]:  # Limit to avoid spam
                insights.append({
                    'type': 'person',
                    'entity_name': person_email,
                    'entity_value': {'email': person_email, 'source': 'email'},
                    'confidence': 0.9,
                    'context': f"Email: {subject[:50]}"
                })

                # Add to knowledge graph
                await self.upsert_person(conn, person_email, "email_contact")

            # Extract topics from subject and body
            words = body.lower().split()
            technical_keywords = ['api', 'code', 'function', 'database', 'server', 'bug', 'deploy',
                                 'docker', 'python', 'javascript', 'sql', 'git', 'feature',
                                 'echo', 'tower', 'comfyui', 'llm', 'ai', 'ml']

            topics_found = [word for word in words if word in technical_keywords]
            if topics_found:
                topic_counts = Counter(topics_found)
                for topic, count in topic_counts.most_common(5):
                    insights.append({
                        'type': 'topic',
                        'entity_name': topic,
                        'entity_value': {'mentions': count, 'category': 'technical'},
                        'confidence': 0.7,
                        'context': subject[:100]
                    })

            # Communication patterns
            insights.append({
                'type': 'communication_pattern',
                'entity_name': None,
                'entity_value': {
                    'length': len(body),
                    'word_count': len(words),
                    'has_question': '?' in body,
                    'technical_content': len(topics_found) > 0
                },
                'confidence': 1.0,
                'context': f"Email from {sender}"
            })

            return {"insights": insights, "skip": False, "size": len(body)}

        except Exception as e:
            logger.error(f"Error processing email {index}: {e}")
            return {"insights": [], "skip": True}

    async def process_emails_batch(self, conn, limit: int = None):
        """Process emails from MBOX file"""
        logger.info(f"📧 Starting email processing from {self.mbox_path}")

        if not self.mbox_path.exists():
            logger.error(f"❌ MBOX file not found: {self.mbox_path}")
            return

        mbox = mailbox.mbox(str(self.mbox_path))
        total_emails = len(mbox)
        logger.info(f"📊 Found {total_emails} total emails")

        processed_count = 0
        insights_count = 0
        skipped_count = 0

        for index, message in enumerate(mbox):
            if limit and processed_count >= limit:
                break

            try:
                start_time = datetime.now()

                # Process email
                result = await self.process_email(conn, message, index)

                if result['skip']:
                    skipped_count += 1
                    continue

                # Calculate processing time
                processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

                # Create file hash (use message-id if available)
                message_id = message.get('Message-ID', f'email_{index}')
                file_hash = hashlib.sha256(message_id.encode()).hexdigest()

                # Check if already processed
                if await self.file_already_processed(conn, file_hash):
                    skipped_count += 1
                    continue

                # Mark file as processed
                file_id = await conn.fetchval("""
                    INSERT INTO takeout_files_processed
                    (file_path, file_type, file_hash, file_size_bytes, processing_time_ms, insights_extracted)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                """, f"mbox/email_{index}", "email", file_hash, result.get('size', 0),
                    processing_time, len(result['insights']))

                # Store insights
                for insight in result['insights']:
                    await self.store_insight(
                        conn, file_id, insight['type'], insight['entity_name'],
                        insight['entity_value'], insight['confidence'], insight['context']
                    )

                processed_count += 1
                insights_count += len(result['insights'])

                # Progress update every 100 emails
                if processed_count % 100 == 0:
                    await self.update_session_stats(conn, 100, insights_count)
                    logger.info(f"📊 Processed {processed_count}/{total_emails} emails, {insights_count} insights")
                    insights_count = 0  # Reset counter

            except Exception as e:
                logger.error(f"Error processing email {index}: {e}")
                continue

        # Final stats update
        if insights_count > 0:
            await self.update_session_stats(conn, processed_count % 100, insights_count)

        logger.info(f"✅ Email processing complete: {processed_count} processed, {skipped_count} skipped")

    async def process_photo_metadata(self, conn, json_file: Path) -> Dict[str, Any]:
        """Process a single photo JSON metadata file"""
        insights = []

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract people tags
            people_in_photo = data.get('people', [])
            for person in people_in_photo:
                person_name = person.get('name', '')
                if person_name:
                    insights.append({
                        'type': 'person',
                        'entity_name': person_name,
                        'entity_value': {'source': 'photo_tag'},
                        'confidence': 0.95,
                        'context': f"Photo: {json_file.stem}"
                    })

                    # Update photo count for person
                    await conn.execute("""
                        UPDATE takeout_people
                        SET photo_count = photo_count + 1
                        WHERE name = $1
                    """, person_name)

                    await self.upsert_person(conn, person_name, "family_or_friend",
                                            datetime.fromtimestamp(int(data.get("photoTakenTime", {}).get("timestamp", 0)) if data.get("photoTakenTime", {}).get("timestamp") else 0))

            # Extract location
            geo_data = data.get('geoData', {})
            if geo_data:
                lat = geo_data.get('latitude')
                lon = geo_data.get('longitude')

                # Get location name from geoDataExif if available
                location_name = data.get('geoDataExif', {}).get('formattedAddress', f"Location_{lat}_{lon}")

                if lat and lon:
                    insights.append({
                        'type': 'location',
                        'entity_name': location_name,
                        'entity_value': {'lat': lat, 'lon': lon, 'source': 'photo_exif'},
                        'confidence': 0.98,
                        'context': f"Photo taken at {location_name}"
                    })

                    await self.upsert_location(conn, location_name, lat, lon,
                                              datetime.fromtimestamp(int(data.get("photoTakenTime", {}).get("timestamp", 0)) if data.get("photoTakenTime", {}).get("timestamp") else 0))

                    # Update photo count for location
                    await conn.execute("""
                        UPDATE takeout_locations
                        SET photo_count = photo_count + 1
                        WHERE name = $1 AND latitude = $2 AND longitude = $3
                    """, location_name, lat, lon)

            # Extract description as topics
            description = data.get('description', '')
            if description:
                insights.append({
                    'type': 'topic',
                    'entity_name': 'photo_description',
                    'entity_value': {'description': description[:200]},
                    'confidence': 0.8,
                    'context': description[:100]
                })

            return {"insights": insights, "skip": False}

        except Exception as e:
            logger.error(f"Error processing photo metadata {json_file}: {e}")
            return {"insights": [], "skip": True}

    async def process_photos_batch(self, conn, limit: int = None):
        """Process photo metadata JSON files"""
        logger.info(f"📸 Starting photo metadata processing from {self.photos_path}")

        if not self.photos_path.exists():
            logger.error(f"❌ Photos directory not found: {self.photos_path}")
            return

        # Find all JSON files
        json_files = list(self.photos_path.rglob("*.json"))
        total_files = len(json_files)
        logger.info(f"📊 Found {total_files} JSON metadata files")

        processed_count = 0
        insights_count = 0
        skipped_count = 0

        for json_file in json_files:
            if limit and processed_count >= limit:
                break

            try:
                start_time = datetime.now()

                # Calculate file hash
                file_hash = self.calculate_file_hash(json_file)

                # Check if already processed
                if await self.file_already_processed(conn, file_hash):
                    skipped_count += 1
                    continue

                # Process metadata
                result = await self.process_photo_metadata(conn, json_file)

                if result['skip']:
                    skipped_count += 1
                    continue

                # Calculate processing time
                processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

                # Mark file as processed
                file_id = await conn.fetchval("""
                    INSERT INTO takeout_files_processed
                    (file_path, file_type, file_hash, file_size_bytes, processing_time_ms, insights_extracted)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                """, str(json_file.relative_to(self.takeout_base)), "photo_json", file_hash,
                    json_file.stat().st_size, processing_time, len(result['insights']))

                # Store insights
                for insight in result['insights']:
                    await self.store_insight(
                        conn, file_id, insight['type'], insight['entity_name'],
                        insight['entity_value'], insight['confidence'], insight['context']
                    )

                processed_count += 1
                insights_count += len(result['insights'])

                # Progress update every 100 files
                if processed_count % 100 == 0:
                    await self.update_session_stats(conn, 100, insights_count)
                    logger.info(f"📊 Processed {processed_count}/{total_files} photos, {insights_count} insights")
                    insights_count = 0

            except Exception as e:
                logger.error(f"Error processing photo {json_file}: {e}")
                continue

        # Final stats update
        if insights_count > 0:
            await self.update_session_stats(conn, processed_count % 100, insights_count)

        logger.info(f"✅ Photo processing complete: {processed_count} processed, {skipped_count} skipped")

    async def run_full_batch(self, email_limit: int = None, photo_limit: int = None):
        """Run full batch processing"""
        logger.info("🚀 Starting comprehensive Takeout batch processing")

        conn = await asyncpg.connect(self.db_url)

        try:
            # Start session
            self.session_id = await self.start_processing_session(conn, "batch")

            # Process emails
            logger.info("\n=== PHASE 1: Email Processing ===")
            await self.process_emails_batch(conn, email_limit)

            # Process photos
            logger.info("\n=== PHASE 2: Photo Metadata Processing ===")
            await self.process_photos_batch(conn, photo_limit)

            # End session
            await self.end_processing_session(conn)

            # Show final stats
            progress = await conn.fetchrow('SELECT * FROM takeout_processing_progress')
            logger.info("\n=== FINAL STATISTICS ===")
            logger.info(f"Files Processed: {progress['files_processed']}")
            logger.info(f"Insights Extracted: {progress['total_insights']}")
            logger.info(f"People Discovered: {progress['people_discovered']}")
            logger.info(f"Locations Found: {progress['locations_discovered']}")
            logger.info(f"Events Identified: {progress['events_identified']}")
            logger.info(f"Bytes Processed: {progress['bytes_processed']:,} bytes")
            logger.info(f"Progress: {((progress['bytes_processed'] or 0) / 786000000000) * 100:.2f}%")

        finally:
            await conn.close()


async def main():
    import sys

    processor = TakeoutBatchProcessor()

    # Parse command line arguments
    email_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    photo_limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    logger.info(f"Email limit: {email_limit or 'ALL'}")
    logger.info(f"Photo limit: {photo_limit or 'ALL'}")

    await processor.run_full_batch(email_limit, photo_limit)


if __name__ == '__main__':
    asyncio.run(main())
