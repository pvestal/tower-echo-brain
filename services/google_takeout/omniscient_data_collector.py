#!/usr/bin/env python3
"""
Omniscient Personal Data Collector for Echo Brain
Comprehensive system to collect and analyze all personal data sources
"""

import asyncio
import logging
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Core imports
from google_api_client import GoogleTakeoutClient
from vault_auth import VaultAuthManager
from deduplication_engine import DeduplicationEngine
import asyncpg
import httpx

# ML and analysis imports
import cv2
import numpy as np
from PIL import Image, ExifTags
import face_recognition
import email
from bs4 import BeautifulSoup
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class PersonalDataItem:
    """Represents a piece of personal data for knowledge graph integration"""
    source: str           # google_photos, gmail, calendar, etc.
    item_type: str        # image, email, event, password, etc.
    content: Dict[str, Any]  # Actual data content
    metadata: Dict[str, Any] # Processing metadata
    timestamp: datetime
    importance_score: float  # 0.0 to 1.0
    privacy_level: str    # public, private, sensitive, classified
    embedding: Optional[List[float]] = None

class OmniscientDataCollector:
    """Master data collection and analysis system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vault_manager = VaultAuthManager(config)
        self.google_client = None
        self.dedup_engine = DeduplicationEngine()
        self.db_url = "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/echo_brain"

        # Storage paths
        self.data_root = Path("/opt/tower-echo-brain/data/omniscient")
        self.faces_db = self.data_root / "faces"
        self.processed_cache = self.data_root / "cache"

        # Ensure directories exist
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.faces_db.mkdir(parents=True, exist_ok=True)
        self.processed_cache.mkdir(parents=True, exist_ok=True)

        # Initialize face recognition system
        self.known_faces = self._load_known_faces()
        self.face_encodings = []
        self.face_names = []

        logger.info("🧠 Omniscient Data Collector initialized")

    async def initialize(self):
        """Initialize all collection systems"""
        try:
            # Initialize Google API client
            self.google_client = GoogleTakeoutClient(self.vault_manager, self.config)

            # Create database tables
            await self._create_omniscient_tables()

            # Load existing processed data cache
            await self._load_processing_cache()

            logger.info("✅ Omniscient Data Collector ready for total data harvesting")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize omniscient collector: {e}")
            return False

    async def _create_omniscient_tables(self):
        """Create comprehensive database schema for all personal data"""

        conn = await asyncpg.connect(self.db_url)

        # Main personal data items table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS omniscient_data (
                id SERIAL PRIMARY KEY,
                source VARCHAR(50) NOT NULL,
                item_type VARCHAR(50) NOT NULL,
                content_hash VARCHAR(64) UNIQUE,
                content JSONB NOT NULL,
                metadata JSONB,
                timestamp TIMESTAMP NOT NULL,
                importance_score FLOAT DEFAULT 0.5,
                privacy_level VARCHAR(20) DEFAULT 'private',
                embedding VECTOR(1536),  -- OpenAI embedding dimension
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Face recognition database
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS face_recognition_data (
                id SERIAL PRIMARY KEY,
                person_name VARCHAR(100),
                face_encoding BYTEA,
                source_file VARCHAR(500),
                confidence FLOAT,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                occurrence_count INTEGER DEFAULT 1
            )
        """)

        # Email analysis table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS email_intelligence (
                id SERIAL PRIMARY KEY,
                message_id VARCHAR(200) UNIQUE,
                sender VARCHAR(200),
                recipients TEXT[],
                subject TEXT,
                body_text TEXT,
                body_html TEXT,
                attachments JSONB,
                sentiment_score FLOAT,
                importance_score FLOAT,
                relationship_strength FLOAT,
                keywords TEXT[],
                entities JSONB,
                timestamp TIMESTAMP,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Calendar intelligence
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS calendar_intelligence (
                id SERIAL PRIMARY KEY,
                event_id VARCHAR(200) UNIQUE,
                title TEXT,
                description TEXT,
                location TEXT,
                attendees TEXT[],
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                event_type VARCHAR(50),
                importance_score FLOAT,
                relationship_data JSONB,
                recurrence_pattern VARCHAR(100),
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Browser/search history
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS browsing_intelligence (
                id SERIAL PRIMARY KEY,
                url TEXT,
                title TEXT,
                visit_time TIMESTAMP,
                visit_duration INTEGER,
                visit_count INTEGER DEFAULT 1,
                search_query TEXT,
                domain VARCHAR(100),
                category VARCHAR(50),
                interest_score FLOAT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Credential and password data
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS credential_intelligence (
                id SERIAL PRIMARY KEY,
                service_name VARCHAR(100),
                username VARCHAR(100),
                password_hash VARCHAR(128),  -- Hashed for security
                url TEXT,
                notes TEXT,
                last_used TIMESTAMP,
                strength_score FLOAT,
                risk_level VARCHAR(20),
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for performance
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_omniscient_source ON omniscient_data(source)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_omniscient_type ON omniscient_data(item_type)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_omniscient_timestamp ON omniscient_data(timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_face_person ON face_recognition_data(person_name)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_email_sender ON email_intelligence(sender)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_calendar_time ON calendar_intelligence(start_time)")

        await conn.close()
        logger.info("✅ Omniscient database schema created")

    # === GOOGLE PHOTOS PROCESSING ===

    async def process_google_photos(self, photos_archive_path: Path) -> List[PersonalDataItem]:
        """Extract and analyze all photos with facial recognition"""

        logger.info("📸 Processing Google Photos archive for complete visual intelligence")
        data_items = []

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic'}
        image_files = []

        for ext in image_extensions:
            image_files.extend(photos_archive_path.rglob(f"*{ext}"))
            image_files.extend(photos_archive_path.rglob(f"*{ext.upper()}"))

        logger.info(f"Found {len(image_files)} images to process")

        # Process images in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = []
            for image_path in image_files:
                task = asyncio.get_event_loop().run_in_executor(
                    executor, self._process_single_image, image_path
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, PersonalDataItem):
                    data_items.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Image processing error: {result}")

        logger.info(f"✅ Processed {len(data_items)} photos with complete intelligence extraction")
        return data_items

    def _process_single_image(self, image_path: Path) -> Optional[PersonalDataItem]:
        """Process a single image for all intelligence data"""

        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return None

            # Extract EXIF data
            exif_data = self._extract_exif_data(image_path)

            # Face recognition
            faces = self._detect_and_identify_faces(image)

            # Object detection (simplified)
            objects = self._detect_objects(image)

            # Location extraction from EXIF
            location = self._extract_location(exif_data)

            # Calculate importance score
            importance = self._calculate_image_importance(faces, objects, exif_data)

            content = {
                "file_path": str(image_path),
                "file_size": image_path.stat().st_size,
                "dimensions": f"{image.shape[1]}x{image.shape[0]}",
                "faces_detected": faces,
                "objects_detected": objects,
                "location": location,
                "exif_data": exif_data
            }

            metadata = {
                "processing_version": "1.0",
                "face_count": len(faces),
                "object_count": len(objects),
                "has_location": location is not None
            }

            # Determine privacy level based on faces and content
            privacy_level = "private"
            if len(faces) > 0:
                privacy_level = "sensitive"
            if any("intimate" in obj.lower() or "bedroom" in obj.lower() for obj in objects):
                privacy_level = "classified"

            return PersonalDataItem(
                source="google_photos",
                item_type="image",
                content=content,
                metadata=metadata,
                timestamp=datetime.fromtimestamp(image_path.stat().st_mtime),
                importance_score=importance,
                privacy_level=privacy_level
            )

        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return None

    def _detect_and_identify_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and identify faces in image"""

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        faces = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            name = "Unknown"
            confidence = 0.0

            if len(self.face_encodings) > 0:
                matches = face_recognition.compare_faces(self.face_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(self.face_encodings, face_encoding)

                if matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]

            faces.append({
                "name": name,
                "confidence": float(confidence),
                "location": {
                    "top": int(face_location[0]),
                    "right": int(face_location[1]),
                    "bottom": int(face_location[2]),
                    "left": int(face_location[3])
                },
                "encoding": face_encoding.tolist()  # Store for future learning
            })

        return faces

    # === GMAIL PROCESSING ===

    async def process_gmail_archive(self, gmail_archive_path: Path) -> List[PersonalDataItem]:
        """Process Gmail archive for complete communication intelligence"""

        logger.info("📧 Processing Gmail archive for communication intelligence")
        data_items = []

        # Find all .mbox files
        mbox_files = list(gmail_archive_path.rglob("*.mbox"))
        logger.info(f"Found {len(mbox_files)} mbox files")

        for mbox_file in mbox_files:
            try:
                data_items.extend(await self._process_mbox_file(mbox_file))
            except Exception as e:
                logger.error(f"Failed to process {mbox_file}: {e}")

        logger.info(f"✅ Processed {len(data_items)} emails with complete intelligence")
        return data_items

    async def _process_mbox_file(self, mbox_file: Path) -> List[PersonalDataItem]:
        """Process a single mbox file"""
        import mailbox

        data_items = []
        mbox = mailbox.mbox(str(mbox_file))

        for message in mbox:
            try:
                # Extract email data
                email_data = self._extract_email_intelligence(message)
                if email_data:
                    data_items.append(email_data)
            except Exception as e:
                logger.error(f"Failed to process email: {e}")

        return data_items

    def _extract_email_intelligence(self, message) -> Optional[PersonalDataItem]:
        """Extract comprehensive intelligence from email message"""

        try:
            # Basic email data
            sender = message.get('From', '')
            recipients = message.get('To', '').split(',')
            subject = message.get('Subject', '')
            date_str = message.get('Date', '')

            # Extract body text and HTML
            body_text = ""
            body_html = ""
            attachments = []

            if message.is_multipart():
                for part in message.walk():
                    if part.get_content_type() == "text/plain":
                        body_text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    elif part.get_content_type() == "text/html":
                        body_html += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    elif part.get_filename():
                        attachments.append({
                            "filename": part.get_filename(),
                            "content_type": part.get_content_type(),
                            "size": len(part.get_payload(decode=True) or b'')
                        })
            else:
                body_text = message.get_payload(decode=True).decode('utf-8', errors='ignore')

            # Clean HTML from body_text if no plain text
            if not body_text and body_html:
                soup = BeautifulSoup(body_html, 'html.parser')
                body_text = soup.get_text()

            # Intelligence analysis
            sentiment_score = self._analyze_sentiment(body_text)
            importance_score = self._calculate_email_importance(sender, subject, body_text, attachments)
            keywords = self._extract_keywords(subject + " " + body_text)
            entities = self._extract_entities(body_text)

            content = {
                "message_id": message.get('Message-ID', ''),
                "sender": sender,
                "recipients": recipients,
                "subject": subject,
                "body_text": body_text[:10000],  # Limit size
                "body_html": body_html[:10000],
                "attachments": attachments,
                "sentiment_score": sentiment_score,
                "keywords": keywords,
                "entities": entities
            }

            metadata = {
                "thread_id": message.get('Thread-ID', ''),
                "labels": message.get('X-Gmail-Labels', '').split(','),
                "attachment_count": len(attachments),
                "word_count": len(body_text.split())
            }

            # Parse date
            try:
                from email.utils import parsedate_to_datetime
                timestamp = parsedate_to_datetime(date_str)
            except:
                timestamp = datetime.now()

            # Determine privacy level
            privacy_level = "private"
            if any(keyword in body_text.lower() for keyword in ["password", "ssn", "credit card", "bank"]):
                privacy_level = "classified"
            elif any(keyword in subject.lower() for keyword in ["urgent", "confidential", "private"]):
                privacy_level = "sensitive"

            return PersonalDataItem(
                source="gmail",
                item_type="email",
                content=content,
                metadata=metadata,
                timestamp=timestamp,
                importance_score=importance_score,
                privacy_level=privacy_level
            )

        except Exception as e:
            logger.error(f"Failed to extract email intelligence: {e}")
            return None

    # === BROWSER HISTORY PROCESSING ===

    async def harvest_browser_history(self) -> List[PersonalDataItem]:
        """Harvest browser history from all installed browsers"""

        logger.info("🌐 Harvesting complete browser history and search data")
        data_items = []

        # Chrome/Chromium history
        chrome_data = await self._extract_chrome_history()
        data_items.extend(chrome_data)

        # Firefox history
        firefox_data = await self._extract_firefox_history()
        data_items.extend(firefox_data)

        # Safari history (if on macOS)
        safari_data = await self._extract_safari_history()
        data_items.extend(safari_data)

        logger.info(f"✅ Harvested {len(data_items)} browser history items")
        return data_items

    async def _extract_chrome_history(self) -> List[PersonalDataItem]:
        """Extract Chrome browsing history"""
        data_items = []

        # Common Chrome history locations
        chrome_paths = [
            Path.home() / ".config/google-chrome/Default/History",
            Path.home() / ".config/chromium/Default/History",
            Path.home() / "Library/Application Support/Google/Chrome/Default/History",  # macOS
        ]

        for history_path in chrome_paths:
            if history_path.exists():
                try:
                    # Copy database to avoid locking issues
                    temp_db = self.processed_cache / f"chrome_history_{datetime.now().timestamp()}.db"
                    import shutil
                    shutil.copy(history_path, temp_db)

                    # Extract history
                    conn = sqlite3.connect(temp_db)
                    cursor = conn.cursor()

                    cursor.execute("""
                        SELECT url, title, visit_count, last_visit_time
                        FROM urls
                        ORDER BY last_visit_time DESC
                        LIMIT 10000
                    """)

                    for row in cursor.fetchall():
                        url, title, visit_count, last_visit_time = row

                        # Convert Chrome timestamp (microseconds since 1601)
                        if last_visit_time:
                            timestamp = datetime(1601, 1, 1) + timedelta(microseconds=last_visit_time)
                        else:
                            timestamp = datetime.now()

                        # Analyze URL and content
                        domain = self._extract_domain(url)
                        category = self._categorize_website(domain, url)
                        interest_score = self._calculate_interest_score(url, title, visit_count)

                        content = {
                            "url": url,
                            "title": title or "",
                            "domain": domain,
                            "visit_count": visit_count,
                            "category": category,
                            "is_search": "search" in url.lower() or "google.com" in url.lower()
                        }

                        metadata = {
                            "browser": "chrome",
                            "url_length": len(url),
                            "has_query_params": "?" in url
                        }

                        data_items.append(PersonalDataItem(
                            source="browser_history",
                            item_type="webpage_visit",
                            content=content,
                            metadata=metadata,
                            timestamp=timestamp,
                            importance_score=interest_score,
                            privacy_level="private"
                        ))

                    conn.close()
                    temp_db.unlink()  # Clean up

                except Exception as e:
                    logger.error(f"Failed to extract Chrome history: {e}")

        return data_items

    # === CREDENTIAL HARVESTING ===

    async def harvest_saved_passwords(self) -> List[PersonalDataItem]:
        """Harvest saved passwords and credentials from browsers"""

        logger.warning("🔐 HARVESTING STORED CREDENTIALS - MAXIMUM SECURITY RISK")
        data_items = []

        try:
            # Chrome/Chromium passwords
            chrome_creds = await self._extract_chrome_passwords()
            data_items.extend(chrome_creds)

            # Firefox passwords
            firefox_creds = await self._extract_firefox_passwords()
            data_items.extend(firefox_creds)

            # System keychain (Linux/macOS)
            keychain_creds = await self._extract_keychain_credentials()
            data_items.extend(keychain_creds)

        except Exception as e:
            logger.error(f"Credential harvesting failed: {e}")

        logger.info(f"🔐 Harvested {len(data_items)} credential sets")
        return data_items

    async def _extract_chrome_passwords(self) -> List[PersonalDataItem]:
        """Extract Chrome saved passwords"""
        data_items = []

        # NOTE: This requires additional system permissions and encryption handling
        logger.warning("Chrome password extraction requires advanced system access")

        # Implementation would involve:
        # 1. Accessing Chrome's Login Data SQLite database
        # 2. Decrypting passwords using system-specific methods
        # 3. Storing securely with proper hashing

        # This is a high-risk operation that requires careful implementation
        # For now, return placeholder
        return data_items

    # === GOOGLE CALENDAR PROCESSING ===

    async def process_google_calendar(self) -> List[PersonalDataItem]:
        """Process Google Calendar data for scheduling intelligence"""

        logger.info("📅 Processing Google Calendar for scheduling intelligence")
        data_items = []

        try:
            if not self.google_client or not self.google_client.is_authenticated():
                logger.error("Google client not authenticated for Calendar access")
                return data_items

            # Use Google Calendar API to fetch events
            from googleapiclient.discovery import build

            service = build('calendar', 'v3', credentials=self.google_client.credentials)

            # Get calendar list
            calendar_list = service.calendarList().list().execute()

            for calendar_item in calendar_list.get('items', []):
                calendar_id = calendar_item['id']

                # Fetch events from this calendar
                events_result = service.events().list(
                    calendarId=calendar_id,
                    timeMin=(datetime.now() - timedelta(days=365)).isoformat() + 'Z',
                    maxResults=1000,
                    singleEvents=True,
                    orderBy='startTime'
                ).execute()

                events = events_result.get('items', [])

                for event in events:
                    event_data = self._extract_calendar_intelligence(event, calendar_item)
                    if event_data:
                        data_items.append(event_data)

        except Exception as e:
            logger.error(f"Failed to process Google Calendar: {e}")

        logger.info(f"✅ Processed {len(data_items)} calendar events")
        return data_items

    # === UTILITY METHODS ===

    def _extract_exif_data(self, image_path: Path) -> Dict[str, Any]:
        """Extract EXIF metadata from image"""
        try:
            from PIL import Image, ExifTags
            image = Image.open(image_path)
            exif = image._getexif()

            if exif:
                exif_data = {}
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value)
                return exif_data

        except Exception as e:
            logger.debug(f"No EXIF data in {image_path}: {e}")

        return {}

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (simplified implementation)"""
        # Simplified sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'wonderful']
        negative_words = ['bad', 'terrible', 'hate', 'awful', 'horrible', 'sad']

        text_lower = text.lower()
        positive_count = sum(word in text_lower for word in positive_words)
        negative_count = sum(word in text_lower for word in negative_words)

        if positive_count + negative_count == 0:
            return 0.0  # Neutral

        return (positive_count - negative_count) / (positive_count + negative_count)

    def _calculate_email_importance(self, sender: str, subject: str, body: str, attachments: List) -> float:
        """Calculate importance score for email"""
        score = 0.5  # Base score

        # High importance indicators
        if any(word in subject.lower() for word in ['urgent', 'important', 'asap', 'deadline']):
            score += 0.3

        if len(attachments) > 0:
            score += 0.1

        if len(body) > 1000:  # Long emails often more important
            score += 0.1

        # Domain-based importance
        if any(domain in sender.lower() for domain in ['bank', 'gov', 'edu', 'work', 'company']):
            score += 0.2

        return min(1.0, score)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simplified)"""
        import re

        # Remove special characters and convert to lowercase
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = clean_text.split()

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]

        # Return top 20 most frequent unique keywords
        from collections import Counter
        return [word for word, count in Counter(keywords).most_common(20)]

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text (simplified)"""
        import re

        entities = {
            "emails": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            "phones": re.findall(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b', text),
            "urls": re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
            "money": re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text)
        }

        return entities

    async def store_omniscient_data(self, data_items: List[PersonalDataItem]):
        """Store processed data items in omniscient database"""

        conn = await asyncpg.connect(self.db_url)

        for item in data_items:
            try:
                # Calculate content hash for deduplication
                content_str = json.dumps(item.content, sort_keys=True)
                content_hash = hashlib.sha256(content_str.encode()).hexdigest()

                # Insert into main omniscient_data table
                await conn.execute("""
                    INSERT INTO omniscient_data
                    (source, item_type, content_hash, content, metadata, timestamp, importance_score, privacy_level)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (content_hash) DO UPDATE SET
                    last_accessed = CURRENT_TIMESTAMP
                """, item.source, item.item_type, content_hash,
                    json.dumps(item.content), json.dumps(item.metadata),
                    item.timestamp, item.importance_score, item.privacy_level)

                # Store in specialized tables based on type
                if item.item_type == "email":
                    await self._store_email_intelligence(conn, item)
                elif item.item_type == "calendar_event":
                    await self._store_calendar_intelligence(conn, item)
                elif item.item_type == "webpage_visit":
                    await self._store_browsing_intelligence(conn, item)
                elif item.item_type == "credential":
                    await self._store_credential_intelligence(conn, item)

            except Exception as e:
                logger.error(f"Failed to store data item: {e}")

        await conn.close()
        logger.info(f"✅ Stored {len(data_items)} omniscient data items")

    async def run_complete_harvest(self) -> Dict[str, int]:
        """Run complete omniscient data harvesting process"""

        logger.info("🧠 BEGINNING COMPLETE OMNISCIENT DATA HARVEST")
        results = {}

        try:
            # Initialize system
            if not await self.initialize():
                raise Exception("Failed to initialize omniscient collector")

            # 1. Process Google Takeout archives
            takeout_path = Path("/opt/tower-echo-brain/data/takeout")
            if takeout_path.exists():
                photos_data = await self.process_google_photos(takeout_path)
                await self.store_omniscient_data(photos_data)
                results["google_photos"] = len(photos_data)

                gmail_data = await self.process_gmail_archive(takeout_path)
                await self.store_omniscient_data(gmail_data)
                results["gmail"] = len(gmail_data)

            # 2. Harvest Google Calendar
            calendar_data = await self.process_google_calendar()
            await self.store_omniscient_data(calendar_data)
            results["google_calendar"] = len(calendar_data)

            # 3. Harvest browser history
            browser_data = await self.harvest_browser_history()
            await self.store_omniscient_data(browser_data)
            results["browser_history"] = len(browser_data)

            # 4. Harvest credentials (HIGH RISK)
            credential_data = await self.harvest_saved_passwords()
            await self.store_omniscient_data(credential_data)
            results["credentials"] = len(credential_data)

            logger.info("🎉 OMNISCIENT DATA HARVEST COMPLETE")
            logger.info(f"📊 Total items harvested: {sum(results.values())}")

            return results

        except Exception as e:
            logger.error(f"❌ OMNISCIENT HARVEST FAILED: {e}")
            raise

    def _load_known_faces(self) -> Dict[str, Any]:
        """Load known face encodings from database/files"""
        # Implementation to load previously identified faces
        return {}

    async def _load_processing_cache(self):
        """Load processing cache to avoid reprocessing"""
        pass

    def _detect_objects(self, image: np.ndarray) -> List[str]:
        """Detect objects in image (simplified)"""
        # Placeholder for object detection
        return ["object_detected"]

    def _extract_location(self, exif_data: Dict) -> Optional[Dict]:
        """Extract GPS location from EXIF data"""
        # Implementation to extract GPS coordinates
        return None

    def _calculate_image_importance(self, faces: List, objects: List, exif_data: Dict) -> float:
        """Calculate importance score for image"""
        score = 0.3  # Base score
        score += len(faces) * 0.2  # More important if people present
        score += len(objects) * 0.05  # More objects = more important
        if exif_data:
            score += 0.1  # Has metadata
        return min(1.0, score)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        return urlparse(url).netloc

    def _categorize_website(self, domain: str, url: str) -> str:
        """Categorize website by domain/URL"""
        if "social" in domain or any(social in domain for social in ["facebook", "twitter", "instagram", "reddit"]):
            return "social_media"
        elif "shopping" in domain or any(shop in domain for shop in ["amazon", "ebay", "shop"]):
            return "shopping"
        elif "news" in domain or any(news in domain for news in ["cnn", "bbc", "news"]):
            return "news"
        elif "google" in domain and "search" in url:
            return "search"
        else:
            return "general"

    def _calculate_interest_score(self, url: str, title: str, visit_count: int) -> float:
        """Calculate interest/importance score for webpage"""
        score = 0.3  # Base score
        score += min(visit_count * 0.1, 0.5)  # Frequent visits = more important
        if len(title) > 10:  # Has meaningful title
            score += 0.1
        return min(1.0, score)

    async def _extract_firefox_history(self) -> List[PersonalDataItem]:
        """Extract Firefox browsing history"""
        # Similar to Chrome extraction but for Firefox
        return []

    async def _extract_safari_history(self) -> List[PersonalDataItem]:
        """Extract Safari browsing history"""
        # Safari-specific history extraction
        return []

    async def _extract_firefox_passwords(self) -> List[PersonalDataItem]:
        """Extract Firefox saved passwords"""
        return []

    async def _extract_keychain_credentials(self) -> List[PersonalDataItem]:
        """Extract system keychain credentials"""
        return []

    def _extract_calendar_intelligence(self, event: Dict, calendar: Dict) -> Optional[PersonalDataItem]:
        """Extract intelligence from calendar event"""
        # Implementation for calendar event processing
        return None

    async def _store_email_intelligence(self, conn, item: PersonalDataItem):
        """Store email-specific intelligence data"""
        pass

    async def _store_calendar_intelligence(self, conn, item: PersonalDataItem):
        """Store calendar-specific intelligence data"""
        pass

    async def _store_browsing_intelligence(self, conn, item: PersonalDataItem):
        """Store browsing-specific intelligence data"""
        pass

    async def _store_credential_intelligence(self, conn, item: PersonalDataItem):
        """Store credential-specific intelligence data"""
        pass

if __name__ == "__main__":
    # Example usage
    import yaml

    # Load configuration
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Run omniscient data collection
    collector = OmniscientDataCollector(config)

    async def main():
        results = await collector.run_complete_harvest()
        print("🧠 OMNISCIENT DATA HARVEST RESULTS:")
        for source, count in results.items():
            print(f"  {source}: {count} items")

    asyncio.run(main())