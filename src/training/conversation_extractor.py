#!/usr/bin/env python3
"""
Conversation Data Extraction System for Echo Brain
Extracts and processes existing conversation data from multiple sources for training and learning.
"""

import asyncio
import logging
import json
import psycopg2
import psycopg2.extras
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator
import os
import re
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class ConversationData:
    """Structured conversation data"""
    conversation_id: str
    timestamp: datetime
    user_input: str
    assistant_response: str
    context: Dict[str, Any]
    source: str  # echo_brain, claude_conversations, knowledge_base
    quality_score: float
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    intent: Optional[str] = None
    complexity_score: Optional[float] = None
    topics: List[str] = None

@dataclass
class TrainingDataset:
    """Complete training dataset structure"""
    conversations: List[ConversationData]
    metadata: Dict[str, Any]
    total_conversations: int
    date_range: Tuple[datetime, datetime]
    sources: List[str]
    quality_distribution: Dict[str, int]

class ConversationExtractor:
    """Extract conversation data from multiple Tower system sources"""

    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.conversation_cache = {}
        self.quality_filters = {
            'min_length': 10,
            'max_length': 10000,
            'min_response_length': 5,
            'exclude_patterns': [
                r'^/\w+',  # Commands
                r'^\s*$',  # Empty
                r'^(yes|no|ok|sure)$',  # Too short
            ]
        }

    async def extract_all_conversations(self,
                                      start_date: Optional[datetime] = None,
                                      end_date: Optional[datetime] = None) -> TrainingDataset:
        """Extract conversations from all available sources"""
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=365)  # Last year
            if not end_date:
                end_date = datetime.now()

            logger.info(f"🔄 Extracting conversations from {start_date} to {end_date}")

            all_conversations = []

            # Extract from Echo Brain database
            echo_conversations = await self._extract_echo_brain_conversations(start_date, end_date)
            all_conversations.extend(echo_conversations)

            # Extract from Knowledge Base
            kb_conversations = await self._extract_knowledge_base_conversations(start_date, end_date)
            all_conversations.extend(kb_conversations)

            # Extract from Claude conversation logs
            claude_conversations = await self._extract_claude_conversations(start_date, end_date)
            all_conversations.extend(claude_conversations)

            # Extract from any other conversation sources
            other_conversations = await self._extract_other_sources(start_date, end_date)
            all_conversations.extend(other_conversations)

            # Process and quality-filter conversations
            processed_conversations = await self._process_conversations(all_conversations)

            # Generate metadata
            metadata = self._generate_metadata(processed_conversations, start_date, end_date)

            dataset = TrainingDataset(
                conversations=processed_conversations,
                metadata=metadata,
                total_conversations=len(processed_conversations),
                date_range=(start_date, end_date),
                sources=list(set([conv.source for conv in processed_conversations])),
                quality_distribution=self._analyze_quality_distribution(processed_conversations)
            )

            logger.info(f"✅ Extracted {len(processed_conversations)} conversations from {len(dataset.sources)} sources")

            return dataset

        except Exception as e:
            logger.error(f"❌ Error extracting conversations: {e}")
            raise

    async def _extract_echo_brain_conversations(self, start_date: datetime, end_date: datetime) -> List[ConversationData]:
        """Extract conversations from Echo Brain interaction logs"""
        conversations = []
        try:
            # Try multiple possible database configurations
            db_configs = [
                {**self.db_config, 'database': 'echo_brain'},
                {**self.db_config, 'database': 'tower_consolidated'},
                self.db_config  # Default
            ]

            connection = None
            for config in db_configs:
                try:
                    connection = psycopg2.connect(**config)
                    break
                except psycopg2.Error:
                    continue

            if not connection:
                logger.warning("⚠️ Could not connect to Echo Brain database")
                return conversations

            cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

            # Query interaction logs
            query = """
                SELECT
                    conversation_id, timestamp, query, response,
                    model_used, processing_time, intent, confidence,
                    tier, metadata
                FROM interaction_logs
                WHERE timestamp BETWEEN %s AND %s
                ORDER BY timestamp ASC
            """

            cursor.execute(query, (start_date, end_date))
            results = cursor.fetchall()

            for row in results:
                # Parse metadata
                metadata = {}
                if row['metadata']:
                    try:
                        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                    except:
                        pass

                conversation = ConversationData(
                    conversation_id=row['conversation_id'] or f"echo_{row['timestamp']}",
                    timestamp=row['timestamp'],
                    user_input=row['query'] or "",
                    assistant_response=row['response'] or "",
                    context=metadata,
                    source="echo_brain",
                    quality_score=self._calculate_quality_score(row['query'], row['response']),
                    model_used=row['model_used'],
                    intent=row['intent'],
                    complexity_score=metadata.get('complexity_score')
                )
                conversations.append(conversation)

            connection.close()
            logger.info(f"✅ Extracted {len(conversations)} conversations from Echo Brain")

        except Exception as e:
            logger.warning(f"⚠️ Error extracting Echo Brain conversations: {e}")

        return conversations

    async def _extract_knowledge_base_conversations(self, start_date: datetime, end_date: datetime) -> List[ConversationData]:
        """Extract conversation-like data from Knowledge Base articles"""
        conversations = []
        try:
            # Connect to Knowledge Base database
            connection = psycopg2.connect(**self.db_config)
            cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

            # Query articles with Q&A format
            query = """
                SELECT
                    id, title, content, created_at, updated_at,
                    metadata, tags
                FROM articles
                WHERE created_at BETWEEN %s AND %s
                AND (content ILIKE '%Q:%' OR content ILIKE '%A:%' OR content ILIKE '%question%')
                ORDER BY created_at ASC
            """

            cursor.execute(query, (start_date, end_date))
            results = cursor.fetchall()

            for row in results:
                # Extract Q&A pairs from content
                qa_pairs = self._extract_qa_pairs(row['content'])

                for i, (question, answer) in enumerate(qa_pairs):
                    conversation = ConversationData(
                        conversation_id=f"kb_{row['id']}_{i}",
                        timestamp=row['created_at'],
                        user_input=question,
                        assistant_response=answer,
                        context={
                            'article_title': row['title'],
                            'article_id': row['id'],
                            'tags': row['tags'] or []
                        },
                        source="knowledge_base",
                        quality_score=self._calculate_quality_score(question, answer),
                        topics=self._extract_topics(row['title'], row['content'])
                    )
                    conversations.append(conversation)

            connection.close()
            logger.info(f"✅ Extracted {len(conversations)} Q&A pairs from Knowledge Base")

        except Exception as e:
            logger.warning(f"⚠️ Error extracting Knowledge Base conversations: {e}")

        return conversations

    async def _extract_claude_conversations(self, start_date: datetime, end_date: datetime) -> List[ConversationData]:
        """Extract conversations from Claude conversation logs"""
        conversations = []
        try:
            # Check for Claude conversation logs in various locations
            log_directories = [
                "/home/patrick/.claude/conversations",
                "/opt/tower-echo-brain/logs",
                "/var/log/tower-echo-brain"
            ]

            for log_dir in log_directories:
                if os.path.exists(log_dir):
                    conversations.extend(await self._parse_claude_logs(log_dir, start_date, end_date))

            logger.info(f"✅ Extracted {len(conversations)} conversations from Claude logs")

        except Exception as e:
            logger.warning(f"⚠️ Error extracting Claude conversations: {e}")

        return conversations

    async def _parse_claude_logs(self, log_dir: str, start_date: datetime, end_date: datetime) -> List[ConversationData]:
        """Parse Claude conversation logs from directory"""
        conversations = []
        try:
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    if file.endswith(('.json', '.log', '.txt')):
                        file_path = os.path.join(root, file)
                        file_conversations = await self._parse_log_file(file_path, start_date, end_date)
                        conversations.extend(file_conversations)

        except Exception as e:
            logger.error(f"❌ Error parsing Claude logs: {e}")

        return conversations

    async def _parse_log_file(self, file_path: str, start_date: datetime, end_date: datetime) -> List[ConversationData]:
        """Parse individual log file for conversations"""
        conversations = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    # JSON format logs
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            conv = self._parse_json_conversation(item, file_path)
                            if conv and start_date <= conv.timestamp <= end_date:
                                conversations.append(conv)
                    elif isinstance(data, dict) and 'conversations' in data:
                        for item in data['conversations']:
                            conv = self._parse_json_conversation(item, file_path)
                            if conv and start_date <= conv.timestamp <= end_date:
                                conversations.append(conv)

                else:
                    # Text format logs
                    content = f.read()
                    text_conversations = self._parse_text_conversations(content, file_path)
                    conversations.extend([
                        conv for conv in text_conversations
                        if start_date <= conv.timestamp <= end_date
                    ])

        except Exception as e:
            logger.warning(f"⚠️ Error parsing {file_path}: {e}")

        return conversations

    def _parse_json_conversation(self, data: Dict, source_file: str) -> Optional[ConversationData]:
        """Parse individual JSON conversation entry"""
        try:
            # Handle different JSON formats
            if 'messages' in data:
                # Chat format
                messages = data['messages']
                user_msgs = [msg for msg in messages if msg.get('role') == 'user']
                assistant_msgs = [msg for msg in messages if msg.get('role') == 'assistant']

                if user_msgs and assistant_msgs:
                    return ConversationData(
                        conversation_id=data.get('id', f"claude_{hashlib.md5(str(data).encode()).hexdigest()[:8]}"),
                        timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                        user_input=user_msgs[-1].get('content', ''),
                        assistant_response=assistant_msgs[-1].get('content', ''),
                        context={'source_file': source_file, 'full_conversation': data},
                        source="claude_conversations",
                        quality_score=self._calculate_quality_score(
                            user_msgs[-1].get('content', ''),
                            assistant_msgs[-1].get('content', '')
                        )
                    )

            elif 'input' in data and 'output' in data:
                # Simple input/output format
                return ConversationData(
                    conversation_id=data.get('id', f"claude_{hashlib.md5(str(data).encode()).hexdigest()[:8]}"),
                    timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                    user_input=data['input'],
                    assistant_response=data['output'],
                    context={'source_file': source_file},
                    source="claude_conversations",
                    quality_score=self._calculate_quality_score(data['input'], data['output'])
                )

        except Exception as e:
            logger.warning(f"⚠️ Error parsing JSON conversation: {e}")

        return None

    def _parse_text_conversations(self, content: str, source_file: str) -> List[ConversationData]:
        """Parse text-based conversation logs"""
        conversations = []
        try:
            # Look for conversation patterns
            patterns = [
                r'User:\s*(.*?)\nAssistant:\s*(.*?)(?=\nUser:|\nTimestamp:|\Z)',
                r'Human:\s*(.*?)\nClaude:\s*(.*?)(?=\nHuman:|\nTimestamp:|\Z)',
                r'Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\nTimestamp:|\Z)'
            ]

            for pattern in patterns:
                matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    user_input = match.group(1).strip()
                    assistant_response = match.group(2).strip()

                    conversation = ConversationData(
                        conversation_id=f"text_{hashlib.md5(f'{user_input}{assistant_response}'.encode()).hexdigest()[:8]}",
                        timestamp=datetime.now(),  # Default timestamp for text logs
                        user_input=user_input,
                        assistant_response=assistant_response,
                        context={'source_file': source_file},
                        source="claude_conversations",
                        quality_score=self._calculate_quality_score(user_input, assistant_response)
                    )
                    conversations.append(conversation)

        except Exception as e:
            logger.warning(f"⚠️ Error parsing text conversations: {e}")

        return conversations

    async def _extract_other_sources(self, start_date: datetime, end_date: datetime) -> List[ConversationData]:
        """Extract from other potential conversation sources"""
        conversations = []

        # Check for SQLite databases
        db_files = [
            "/opt/tower-echo-brain/photos.db",
            "/opt/tower-echo-brain/data/conversations.db",
            "/tmp/claude_conversations.db"
        ]

        for db_file in db_files:
            if os.path.exists(db_file):
                sqlite_conversations = await self._extract_from_sqlite(db_file, start_date, end_date)
                conversations.extend(sqlite_conversations)

        return conversations

    async def _extract_from_sqlite(self, db_path: str, start_date: datetime, end_date: datetime) -> List[ConversationData]:
        """Extract conversations from SQLite database"""
        conversations = []
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Try to find conversation-like tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            for (table_name,) in tables:
                if any(keyword in table_name.lower() for keyword in ['conversation', 'message', 'chat', 'interaction']):
                    # Get table schema
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = [col[1] for col in cursor.fetchall()]

                    # Build query based on available columns
                    select_cols = []
                    if 'id' in columns: select_cols.append('id')
                    if 'timestamp' in columns: select_cols.append('timestamp')
                    if 'created_at' in columns: select_cols.append('created_at')
                    if 'user_input' in columns: select_cols.append('user_input')
                    if 'query' in columns: select_cols.append('query')
                    if 'response' in columns: select_cols.append('response')
                    if 'assistant_response' in columns: select_cols.append('assistant_response')

                    if len(select_cols) >= 3:  # Need at least id, timestamp, and content
                        query = f"SELECT {', '.join(select_cols)} FROM {table_name}"
                        cursor.execute(query)
                        rows = cursor.fetchall()

                        for row in rows:
                            # Parse row data
                            row_dict = dict(zip(select_cols, row))

                            # Extract timestamp
                            timestamp = datetime.now()
                            for ts_col in ['timestamp', 'created_at']:
                                if ts_col in row_dict and row_dict[ts_col]:
                                    try:
                                        timestamp = datetime.fromisoformat(str(row_dict[ts_col]))
                                        break
                                    except:
                                        pass

                            if start_date <= timestamp <= end_date:
                                # Extract content
                                user_input = ""
                                assistant_response = ""

                                for input_col in ['user_input', 'query']:
                                    if input_col in row_dict and row_dict[input_col]:
                                        user_input = str(row_dict[input_col])
                                        break

                                for response_col in ['response', 'assistant_response']:
                                    if response_col in row_dict and row_dict[response_col]:
                                        assistant_response = str(row_dict[response_col])
                                        break

                                if user_input and assistant_response:
                                    conversation = ConversationData(
                                        conversation_id=f"sqlite_{db_path}_{row_dict.get('id', len(conversations))}",
                                        timestamp=timestamp,
                                        user_input=user_input,
                                        assistant_response=assistant_response,
                                        context={'database': db_path, 'table': table_name},
                                        source="sqlite_database",
                                        quality_score=self._calculate_quality_score(user_input, assistant_response)
                                    )
                                    conversations.append(conversation)

            conn.close()
            logger.info(f"✅ Extracted {len(conversations)} conversations from {db_path}")

        except Exception as e:
            logger.warning(f"⚠️ Error extracting from SQLite {db_path}: {e}")

        return conversations

    def _extract_qa_pairs(self, content: str) -> List[Tuple[str, str]]:
        """Extract Q&A pairs from article content"""
        qa_pairs = []
        try:
            # Various Q&A patterns
            patterns = [
                r'Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\n\n|\Z)',
                r'Question:\s*(.*?)\nAnswer:\s*(.*?)(?=\nQuestion:|\n\n|\Z)',
                r'#\s*(.*?)\n(.*?)(?=\n#|\n\n|\Z)'
            ]

            for pattern in patterns:
                matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    question = match.group(1).strip()
                    answer = match.group(2).strip()

                    if len(question) > 10 and len(answer) > 10:
                        qa_pairs.append((question, answer))

        except Exception as e:
            logger.warning(f"⚠️ Error extracting Q&A pairs: {e}")

        return qa_pairs

    def _calculate_quality_score(self, user_input: str, assistant_response: str) -> float:
        """Calculate quality score for conversation"""
        try:
            score = 1.0

            # Length checks
            if len(user_input) < self.quality_filters['min_length']:
                score -= 0.3
            if len(assistant_response) < self.quality_filters['min_response_length']:
                score -= 0.4
            if len(user_input) > self.quality_filters['max_length']:
                score -= 0.2

            # Pattern checks
            for pattern in self.quality_filters['exclude_patterns']:
                if re.match(pattern, user_input.strip(), re.IGNORECASE):
                    score -= 0.5
                if re.match(pattern, assistant_response.strip(), re.IGNORECASE):
                    score -= 0.5

            # Content quality
            if len(user_input.split()) < 3:
                score -= 0.2
            if len(assistant_response.split()) < 5:
                score -= 0.3

            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score))

        except:
            return 0.5  # Default medium quality

    def _extract_topics(self, title: str, content: str) -> List[str]:
        """Extract topics from content using simple keyword extraction"""
        try:
            text = f"{title} {content}".lower()

            # Common technical topics
            topics = []
            topic_keywords = {
                'programming': ['code', 'programming', 'python', 'javascript', 'function', 'variable'],
                'ai_ml': ['ai', 'machine learning', 'neural', 'model', 'training', 'algorithm'],
                'system_admin': ['server', 'database', 'deployment', 'configuration', 'service'],
                'web_dev': ['html', 'css', 'web', 'frontend', 'backend', 'api'],
                'data': ['data', 'analysis', 'visualization', 'csv', 'json', 'database'],
                'security': ['security', 'authentication', 'authorization', 'encryption', 'vulnerability']
            }

            for topic, keywords in topic_keywords.items():
                if any(keyword in text for keyword in keywords):
                    topics.append(topic)

            return topics if topics else ['general']

        except:
            return ['general']

    async def _process_conversations(self, conversations: List[ConversationData]) -> List[ConversationData]:
        """Process and filter conversations for quality"""
        processed = []

        for conv in conversations:
            # Quality filtering
            if conv.quality_score < 0.3:
                continue

            # Deduplication
            conv_hash = hashlib.md5(f"{conv.user_input}{conv.assistant_response}".encode()).hexdigest()
            if conv_hash in self.conversation_cache:
                continue
            self.conversation_cache[conv_hash] = True

            # Clean and enhance
            conv.user_input = self._clean_text(conv.user_input)
            conv.assistant_response = self._clean_text(conv.assistant_response)
            conv.tokens_used = len(conv.user_input.split()) + len(conv.assistant_response.split())

            processed.append(conv)

        # Sort by timestamp
        processed.sort(key=lambda x: x.timestamp)

        return processed

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)  # Remove special chars
        text = text.strip()

        return text

    def _generate_metadata(self, conversations: List[ConversationData], start_date: datetime, end_date: datetime) -> Dict:
        """Generate dataset metadata"""
        metadata = {
            'extraction_timestamp': datetime.now().isoformat(),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_conversations': len(conversations),
            'sources': {},
            'quality_stats': {
                'average_quality': np.mean([c.quality_score for c in conversations]),
                'high_quality_count': len([c for c in conversations if c.quality_score > 0.8]),
                'medium_quality_count': len([c for c in conversations if 0.5 <= c.quality_score <= 0.8]),
                'low_quality_count': len([c for c in conversations if c.quality_score < 0.5])
            },
            'token_stats': {
                'total_tokens': sum([c.tokens_used for c in conversations if c.tokens_used]),
                'average_tokens': np.mean([c.tokens_used for c in conversations if c.tokens_used])
            }
        }

        # Source breakdown
        for conv in conversations:
            if conv.source not in metadata['sources']:
                metadata['sources'][conv.source] = 0
            metadata['sources'][conv.source] += 1

        return metadata

    def _analyze_quality_distribution(self, conversations: List[ConversationData]) -> Dict[str, int]:
        """Analyze quality score distribution"""
        distribution = {
            'excellent': 0,    # > 0.9
            'good': 0,        # 0.7-0.9
            'medium': 0,      # 0.5-0.7
            'poor': 0,        # 0.3-0.5
            'very_poor': 0    # < 0.3
        }

        for conv in conversations:
            score = conv.quality_score
            if score > 0.9:
                distribution['excellent'] += 1
            elif score > 0.7:
                distribution['good'] += 1
            elif score > 0.5:
                distribution['medium'] += 1
            elif score > 0.3:
                distribution['poor'] += 1
            else:
                distribution['very_poor'] += 1

        return distribution

    async def export_dataset(self, dataset: TrainingDataset, output_path: str):
        """Export dataset to various formats"""
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Export as JSON
            json_data = {
                'metadata': dataset.metadata,
                'conversations': [asdict(conv) for conv in dataset.conversations]
            }

            with open(output_dir / 'conversations.json', 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, default=str)

            # Export as CSV for analysis
            df = pd.DataFrame([asdict(conv) for conv in dataset.conversations])
            df.to_csv(output_dir / 'conversations.csv', index=False)

            # Export metadata
            with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(dataset.metadata, f, indent=2, default=str)

            # Export quality report
            quality_report = self._generate_quality_report(dataset)
            with open(output_dir / 'quality_report.txt', 'w', encoding='utf-8') as f:
                f.write(quality_report)

            logger.info(f"✅ Dataset exported to {output_path}")

        except Exception as e:
            logger.error(f"❌ Error exporting dataset: {e}")
            raise

    def _generate_quality_report(self, dataset: TrainingDataset) -> str:
        """Generate human-readable quality report"""
        report = f"""
Conversation Dataset Quality Report
==================================
Generated: {datetime.now()}
Date Range: {dataset.date_range[0]} to {dataset.date_range[1]}

SUMMARY
-------
Total Conversations: {dataset.total_conversations:,}
Sources: {', '.join(dataset.sources)}

QUALITY DISTRIBUTION
-------------------
Excellent (>0.9): {dataset.quality_distribution.get('excellent', 0):,}
Good (0.7-0.9): {dataset.quality_distribution.get('good', 0):,}
Medium (0.5-0.7): {dataset.quality_distribution.get('medium', 0):,}
Poor (0.3-0.5): {dataset.quality_distribution.get('poor', 0):,}
Very Poor (<0.3): {dataset.quality_distribution.get('very_poor', 0):,}

METADATA
--------
Average Quality Score: {dataset.metadata.get('quality_stats', {}).get('average_quality', 0):.3f}
Total Tokens: {dataset.metadata.get('token_stats', {}).get('total_tokens', 0):,}
Average Tokens per Conversation: {dataset.metadata.get('token_stats', {}).get('average_tokens', 0):.1f}

SOURCE BREAKDOWN
---------------
"""
        for source, count in dataset.metadata.get('sources', {}).items():
            percentage = (count / dataset.total_conversations) * 100
            report += f"{source}: {count:,} ({percentage:.1f}%)\n"

        return report

# Factory function for Echo Brain integration
def create_conversation_extractor(db_config: Dict[str, str]) -> ConversationExtractor:
    """Create conversation extractor with database configuration"""
    return ConversationExtractor(db_config)