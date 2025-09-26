#!/usr/bin/env python3
"""
Echo Feature Store - Centralized Feature Management
=================================================

Comprehensive feature store for Echo Brain that provides:
- Centralized feature definition and storage
- Feature versioning and lineage tracking
- Real-time and batch feature serving
- Feature sharing across models and experiments
- Data quality monitoring and validation
- Feature engineering pipeline management

Features:
- Time-travel capabilities for reproducible experiments
- Feature transformation and aggregation
- Schema evolution and compatibility
- Performance optimization with caching
- Integration with training and inference pipelines
"""

import asyncio
import logging
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib
import pickle
import redis
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
import asyncpg

# Configuration
FEATURE_STORE_DB = "/opt/tower-echo-brain/data/feature_store.db"
FEATURE_DATA_PATH = "/opt/tower-echo-brain/feature_store/"
FEATURE_CACHE_PATH = "/opt/tower-echo-brain/feature_store/cache/"
REDIS_URL = "redis://localhost:6379"
BATCH_SIZE = 1000

class FeatureType(Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    BOOLEAN = "boolean"
    TEXT = "text"
    VECTOR = "vector"
    TIMESTAMP = "timestamp"

class FeatureStatus(Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class ComputationMode(Enum):
    BATCH = "batch"
    REALTIME = "realtime"
    STREAMING = "streaming"

@dataclass
class FeatureDefinition:
    feature_id: str
    name: str
    description: str
    feature_type: FeatureType
    entity_type: str  # e.g., "user", "session", "decision"
    source_table: str
    computation_logic: str
    dependencies: List[str]
    version: str
    status: FeatureStatus
    created_at: datetime
    created_by: str
    tags: List[str] = None
    validation_rules: Dict[str, Any] = None
    transformation_config: Dict[str, Any] = None

@dataclass
class FeatureValue:
    feature_id: str
    entity_id: str
    value: Any
    timestamp: datetime
    version: str
    metadata: Dict[str, Any] = None

@dataclass
class FeatureSet:
    feature_set_id: str
    name: str
    description: str
    features: List[str]
    entity_type: str
    created_at: datetime
    version: str

class FeatureStore:
    def __init__(self):
        self.db_path = FEATURE_STORE_DB
        self.data_path = Path(FEATURE_DATA_PATH)
        self.cache_path = Path(FEATURE_CACHE_PATH)
        self.logger = logging.getLogger(__name__)
        self._ensure_directories()
        self._init_database()
        self.redis_client = None
        self._init_redis()
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        self.feature_sets: Dict[str, FeatureSet] = {}

    def _ensure_directories(self):
        """Create necessary directories"""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize feature store database"""
        with sqlite3.connect(self.db_path) as conn:
            # Feature definitions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feature_definitions (
                    feature_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    feature_type TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    source_table TEXT,
                    computation_logic TEXT,
                    dependencies TEXT,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    created_by TEXT,
                    tags TEXT,
                    validation_rules TEXT,
                    transformation_config TEXT
                )
            ''')
            
            # Feature values table (for small features, large ones go to Parquet)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feature_values (
                    value_id TEXT PRIMARY KEY,
                    feature_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    value_json TEXT,
                    value_numeric REAL,
                    value_text TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    version TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (feature_id) REFERENCES feature_definitions (feature_id)
                )
            ''')
            
            # Feature sets table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feature_sets (
                    feature_set_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    features TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    version TEXT NOT NULL
                )
            ''')
            
            # Feature lineage table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feature_lineage (
                    lineage_id TEXT PRIMARY KEY,
                    source_feature_id TEXT,
                    target_feature_id TEXT,
                    relationship_type TEXT,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (source_feature_id) REFERENCES feature_definitions (feature_id),
                    FOREIGN KEY (target_feature_id) REFERENCES feature_definitions (feature_id)
                )
            ''')
            
            # Feature usage tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feature_usage (
                    usage_id TEXT PRIMARY KEY,
                    feature_id TEXT NOT NULL,
                    model_name TEXT,
                    experiment_id TEXT,
                    usage_type TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (feature_id) REFERENCES feature_definitions (feature_id)
                )
            ''')
            
            # Feature quality metrics
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feature_quality (
                    quality_id TEXT PRIMARY KEY,
                    feature_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    completeness REAL,
                    uniqueness REAL,
                    validity REAL,
                    consistency REAL,
                    distribution_drift REAL,
                    quality_score REAL,
                    FOREIGN KEY (feature_id) REFERENCES feature_definitions (feature_id)
                )
            ''')

            # Indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_values_feature ON feature_values(feature_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_values_entity ON feature_values(entity_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_values_timestamp ON feature_values(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_quality_feature ON feature_quality(feature_id)')

    def _init_redis(self):
        """Initialize Redis connection for caching"""
        try:
            import redis
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.redis_client = None

    def _generate_id(self, prefix: str = "feat") -> str:
        """Generate unique ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(str(np.random.random()).encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_suffix}"

    async def register_feature(
        self,
        name: str,
        description: str,
        feature_type: FeatureType,
        entity_type: str,
        source_table: str = "",
        computation_logic: str = "",
        dependencies: List[str] = None,
        tags: List[str] = None,
        validation_rules: Dict[str, Any] = None,
        transformation_config: Dict[str, Any] = None,
        created_by: str = "system"
    ) -> str:
        """Register a new feature definition"""
        
        feature_id = self._generate_id("feature")
        dependencies = dependencies or []
        tags = tags or []
        validation_rules = validation_rules or {}
        transformation_config = transformation_config or {}
        
        feature_def = FeatureDefinition(
            feature_id=feature_id,
            name=name,
            description=description,
            feature_type=feature_type,
            entity_type=entity_type,
            source_table=source_table,
            computation_logic=computation_logic,
            dependencies=dependencies,
            version="1.0.0",
            status=FeatureStatus.ACTIVE,
            created_at=datetime.now(),
            created_by=created_by,
            tags=tags,
            validation_rules=validation_rules,
            transformation_config=transformation_config
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO feature_definitions (
                    feature_id, name, description, feature_type, entity_type,
                    source_table, computation_logic, dependencies, version, status,
                    created_at, created_by, tags, validation_rules, transformation_config
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feature_id, name, description, feature_type.value, entity_type,
                source_table, computation_logic, json.dumps(dependencies),
                feature_def.version, feature_def.status.value, feature_def.created_at,
                created_by, json.dumps(tags), json.dumps(validation_rules),
                json.dumps(transformation_config)
            ))
        
        # Cache feature definition
        self.feature_definitions[feature_id] = feature_def
        
        self.logger.info(f"Feature registered: {feature_id} - {name}")
        return feature_id

    async def store_feature_values(
        self,
        feature_id: str,
        values: List[FeatureValue],
        storage_mode: str = "auto"  # "auto", "sqlite", "parquet"
    ) -> bool:
        """Store feature values"""
        
        if feature_id not in self.feature_definitions:
            await self._load_feature_definition(feature_id)
        
        if feature_id not in self.feature_definitions:
            raise ValueError(f"Feature {feature_id} not found")
        
        feature_def = self.feature_definitions[feature_id]
        
        # Decide storage method
        if storage_mode == "auto":
            storage_mode = "parquet" if len(values) > 1000 else "sqlite"
        
        if storage_mode == "parquet":
            return await self._store_values_parquet(feature_id, values)
        else:
            return await self._store_values_sqlite(feature_id, values)

    async def _store_values_sqlite(self, feature_id: str, values: List[FeatureValue]) -> bool:
        """Store feature values in SQLite"""
        
        with sqlite3.connect(self.db_path) as conn:
            for value in values:
                value_id = self._generate_id("value")
                
                # Serialize value based on type
                value_json = None
                value_numeric = None
                value_text = None
                
                if isinstance(value.value, (dict, list)):
                    value_json = json.dumps(value.value)
                elif isinstance(value.value, (int, float)):
                    value_numeric = float(value.value)
                else:
                    value_text = str(value.value)
                
                conn.execute('''
                    INSERT INTO feature_values (
                        value_id, feature_id, entity_id, value_json, value_numeric,
                        value_text, timestamp, version, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    value_id, feature_id, value.entity_id, value_json, value_numeric,
                    value_text, value.timestamp, value.version,
                    json.dumps(value.metadata) if value.metadata else None
                ))
        
        # Clear cache for this feature
        await self._clear_feature_cache(feature_id)
        
        self.logger.info(f"Stored {len(values)} values for feature {feature_id} in SQLite")
        return True

    async def _store_values_parquet(self, feature_id: str, values: List[FeatureValue]) -> bool:
        """Store feature values in Parquet format"""
        
        # Convert to DataFrame
        data = []
        for value in values:
            data.append({
                "entity_id": value.entity_id,
                "value": value.value,
                "timestamp": value.timestamp,
                "version": value.version,
                "metadata": json.dumps(value.metadata) if value.metadata else None
            })
        
        df = pd.DataFrame(data)
        
        # Partition by date for efficient querying
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        # Save to Parquet with partitioning
        feature_dir = self.data_path / feature_id
        feature_dir.mkdir(exist_ok=True)
        
        for date, group in df.groupby('date'):
            date_str = date.strftime('%Y-%m-%d')
            parquet_path = feature_dir / f"date={date_str}" / "data.parquet"
            parquet_path.parent.mkdir(exist_ok=True)
            
            # Append to existing file if it exists
            if parquet_path.exists():
                existing_df = pd.read_parquet(parquet_path)
                combined_df = pd.concat([existing_df, group.drop('date', axis=1)])
                combined_df.to_parquet(parquet_path, index=False)
            else:
                group.drop('date', axis=1).to_parquet(parquet_path, index=False)
        
        # Clear cache for this feature
        await self._clear_feature_cache(feature_id)
        
        self.logger.info(f"Stored {len(values)} values for feature {feature_id} in Parquet")
        return True

    async def get_feature_values(
        self,
        feature_id: str,
        entity_ids: List[str] = None,
        start_time: datetime = None,
        end_time: datetime = None,
        version: str = None,
        use_cache: bool = True
    ) -> List[FeatureValue]:
        """Get feature values"""
        
        # Check cache first
        if use_cache and self.redis_client:
            cache_key = self._get_cache_key(feature_id, entity_ids, start_time, end_time, version)
            cached_values = await self._get_from_cache(cache_key)
            if cached_values:
                return cached_values
        
        # Load from storage
        values = []
        
        # Try Parquet first
        parquet_values = await self._load_values_parquet(feature_id, entity_ids, start_time, end_time, version)
        if parquet_values:
            values.extend(parquet_values)
        else:
            # Fall back to SQLite
            sqlite_values = await self._load_values_sqlite(feature_id, entity_ids, start_time, end_time, version)
            values.extend(sqlite_values)
        
        # Cache results
        if use_cache and self.redis_client and values:
            await self._store_in_cache(cache_key, values, ttl=3600)  # 1 hour TTL
        
        return values

    async def _load_values_sqlite(
        self,
        feature_id: str,
        entity_ids: List[str] = None,
        start_time: datetime = None,
        end_time: datetime = None,
        version: str = None
    ) -> List[FeatureValue]:
        """Load feature values from SQLite"""
        
        query = "SELECT * FROM feature_values WHERE feature_id = ?"
        params = [feature_id]
        
        if entity_ids:
            placeholders = ','.join(['?' for _ in entity_ids])
            query += f" AND entity_id IN ({placeholders})"
            params.extend(entity_ids)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if version:
            query += " AND version = ?"
            params.append(version)
        
        query += " ORDER BY timestamp DESC"
        
        values = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            
            for row in cursor.fetchall():
                # Deserialize value
                if row[3]:  # value_json
                    value = json.loads(row[3])
                elif row[4] is not None:  # value_numeric
                    value = row[4]
                else:  # value_text
                    value = row[5]
                
                feature_value = FeatureValue(
                    feature_id=row[1],
                    entity_id=row[2],
                    value=value,
                    timestamp=datetime.fromisoformat(row[6]),
                    version=row[7],
                    metadata=json.loads(row[8]) if row[8] else None
                )
                values.append(feature_value)
        
        return values

    async def _load_values_parquet(
        self,
        feature_id: str,
        entity_ids: List[str] = None,
        start_time: datetime = None,
        end_time: datetime = None,
        version: str = None
    ) -> List[FeatureValue]:
        """Load feature values from Parquet files"""
        
        feature_dir = self.data_path / feature_id
        if not feature_dir.exists():
            return []
        
        # Find relevant date partitions
        date_dirs = []
        if start_time or end_time:
            start_date = start_time.date() if start_time else datetime(2020, 1, 1).date()
            end_date = end_time.date() if end_time else datetime.now().date()
            
            for date_dir in feature_dir.iterdir():
                if date_dir.is_dir() and date_dir.name.startswith("date="):
                    date_str = date_dir.name.split("=")[1]
                    try:
                        date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        if start_date <= date <= end_date:
                            date_dirs.append(date_dir)
                    except ValueError:
                        continue
        else:
            date_dirs = [d for d in feature_dir.iterdir() if d.is_dir() and d.name.startswith("date=")]
        
        # Load data from all relevant partitions
        all_dfs = []
        for date_dir in date_dirs:
            parquet_file = date_dir / "data.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                all_dfs.append(df)
        
        if not all_dfs:
            return []
        
        # Combine all data
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Apply filters
        if entity_ids:
            combined_df = combined_df[combined_df['entity_id'].isin(entity_ids)]
        
        if start_time:
            combined_df = combined_df[pd.to_datetime(combined_df['timestamp']) >= start_time]
        
        if end_time:
            combined_df = combined_df[pd.to_datetime(combined_df['timestamp']) <= end_time]
        
        if version:
            combined_df = combined_df[combined_df['version'] == version]
        
        # Convert to FeatureValue objects
        values = []
        for _, row in combined_df.iterrows():
            feature_value = FeatureValue(
                feature_id=feature_id,
                entity_id=row['entity_id'],
                value=row['value'],
                timestamp=pd.to_datetime(row['timestamp']).to_pydatetime(),
                version=row['version'],
                metadata=json.loads(row['metadata']) if row['metadata'] else None
            )
            values.append(feature_value)
        
        # Sort by timestamp
        values.sort(key=lambda x: x.timestamp, reverse=True)
        
        return values

    async def create_feature_set(
        self,
        name: str,
        description: str,
        features: List[str],
        entity_type: str
    ) -> str:
        """Create a feature set"""
        
        feature_set_id = self._generate_id("fset")
        
        feature_set = FeatureSet(
            feature_set_id=feature_set_id,
            name=name,
            description=description,
            features=features,
            entity_type=entity_type,
            created_at=datetime.now(),
            version="1.0.0"
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO feature_sets (
                    feature_set_id, name, description, features, entity_type, created_at, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                feature_set_id, name, description, json.dumps(features),
                entity_type, feature_set.created_at, feature_set.version
            ))
        
        self.feature_sets[feature_set_id] = feature_set
        
        self.logger.info(f"Feature set created: {feature_set_id} - {name}")
        return feature_set_id

    async def get_feature_set_values(
        self,
        feature_set_id: str,
        entity_ids: List[str],
        timestamp: datetime = None
    ) -> pd.DataFrame:
        """Get values for all features in a feature set"""
        
        if feature_set_id not in self.feature_sets:
            await self._load_feature_set(feature_set_id)
        
        if feature_set_id not in self.feature_sets:
            raise ValueError(f"Feature set {feature_set_id} not found")
        
        feature_set = self.feature_sets[feature_set_id]
        timestamp = timestamp or datetime.now()
        
        # Get values for each feature
        all_data = {}
        for feature_id in feature_set.features:
            feature_values = await self.get_feature_values(
                feature_id=feature_id,
                entity_ids=entity_ids,
                end_time=timestamp
            )
            
            # Get latest value for each entity
            entity_values = {}
            for value in feature_values:
                if value.entity_id not in entity_values or value.timestamp > entity_values[value.entity_id]['timestamp']:
                    entity_values[value.entity_id] = {
                        'value': value.value,
                        'timestamp': value.timestamp
                    }
            
            # Add to data
            feature_name = await self._get_feature_name(feature_id)
            for entity_id in entity_ids:
                if entity_id not in all_data:
                    all_data[entity_id] = {'entity_id': entity_id}
                
                all_data[entity_id][feature_name] = entity_values.get(entity_id, {}).get('value')
        
        # Convert to DataFrame
        df = pd.DataFrame(list(all_data.values()))
        return df

    async def compute_feature(
        self,
        feature_id: str,
        entity_ids: List[str] = None,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> List[FeatureValue]:
        """Compute feature values using defined computation logic"""
        
        if feature_id not in self.feature_definitions:
            await self._load_feature_definition(feature_id)
        
        feature_def = self.feature_definitions[feature_id]
        
        if not feature_def.computation_logic:
            raise ValueError(f"No computation logic defined for feature {feature_id}")
        
        # This is a simplified implementation
        # In practice, you'd have a more sophisticated feature computation engine
        try:
            # Execute computation logic
            # This could involve SQL queries, Python functions, etc.
            computed_values = await self._execute_computation_logic(
                feature_def.computation_logic,
                entity_ids,
                start_time,
                end_time
            )
            
            # Create FeatureValue objects
            feature_values = []
            for entity_id, value in computed_values.items():
                feature_value = FeatureValue(
                    feature_id=feature_id,
                    entity_id=entity_id,
                    value=value,
                    timestamp=datetime.now(),
                    version=feature_def.version
                )
                feature_values.append(feature_value)
            
            # Store computed values
            await self.store_feature_values(feature_id, feature_values)
            
            return feature_values
            
        except Exception as e:
            self.logger.error(f"Feature computation failed for {feature_id}: {e}")
            raise

    async def _execute_computation_logic(
        self,
        computation_logic: str,
        entity_ids: List[str] = None,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> Dict[str, Any]:
        """Execute feature computation logic"""
        
        # This is a placeholder for feature computation
        # In practice, this would:
        # 1. Parse the computation logic
        # 2. Execute SQL queries or Python functions
        # 3. Handle dependencies
        # 4. Return computed values
        
        # Simulated computation
        computed_values = {}
        
        if entity_ids:
            for entity_id in entity_ids:
                # Simulate feature computation
                computed_values[entity_id] = np.random.random()
        
        return computed_values

    async def monitor_feature_quality(self, feature_id: str) -> Dict[str, float]:
        """Monitor feature quality metrics"""
        
        # Get recent feature values
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)  # Last 7 days
        
        values = await self.get_feature_values(
            feature_id=feature_id,
            start_time=start_time,
            end_time=end_time
        )
        
        if not values:
            return {}
        
        # Calculate quality metrics
        total_count = len(values)
        non_null_count = sum(1 for v in values if v.value is not None)
        unique_values = len(set(str(v.value) for v in values))
        
        # Completeness: ratio of non-null values
        completeness = non_null_count / total_count if total_count > 0 else 0
        
        # Uniqueness: ratio of unique values (relevant for ID-like features)
        uniqueness = unique_values / total_count if total_count > 0 else 0
        
        # Validity: check against validation rules
        validity = await self._check_feature_validity(feature_id, values)
        
        # Consistency: temporal consistency
        consistency = await self._check_feature_consistency(feature_id, values)
        
        # Overall quality score
        quality_score = (completeness + validity + consistency) / 3
        
        quality_metrics = {
            "completeness": completeness,
            "uniqueness": uniqueness,
            "validity": validity,
            "consistency": consistency,
            "quality_score": quality_score
        }
        
        # Store quality metrics
        quality_id = self._generate_id("quality")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO feature_quality (
                    quality_id, feature_id, timestamp, completeness, uniqueness,
                    validity, consistency, quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                quality_id, feature_id, datetime.now(), completeness, uniqueness,
                validity, consistency, quality_score
            ))
        
        return quality_metrics

    async def _check_feature_validity(self, feature_id: str, values: List[FeatureValue]) -> float:
        """Check feature validity against validation rules"""
        
        if feature_id not in self.feature_definitions:
            return 1.0
        
        feature_def = self.feature_definitions[feature_id]
        validation_rules = feature_def.validation_rules or {}
        
        if not validation_rules:
            return 1.0
        
        valid_count = 0
        total_count = len(values)
        
        for value in values:
            is_valid = True
            
            # Check range
            if 'min_value' in validation_rules and isinstance(value.value, (int, float)):
                if value.value < validation_rules['min_value']:
                    is_valid = False
            
            if 'max_value' in validation_rules and isinstance(value.value, (int, float)):
                if value.value > validation_rules['max_value']:
                    is_valid = False
            
            # Check allowed values
            if 'allowed_values' in validation_rules:
                if value.value not in validation_rules['allowed_values']:
                    is_valid = False
            
            # Check regex pattern
            if 'pattern' in validation_rules and isinstance(value.value, str):
                import re
                if not re.match(validation_rules['pattern'], value.value):
                    is_valid = False
            
            if is_valid:
                valid_count += 1
        
        return valid_count / total_count if total_count > 0 else 1.0

    async def _check_feature_consistency(self, feature_id: str, values: List[FeatureValue]) -> float:
        """Check temporal consistency of feature values"""
        
        if len(values) < 2:
            return 1.0
        
        # Sort by timestamp
        sorted_values = sorted(values, key=lambda x: x.timestamp)
        
        # Check for sudden changes (simple heuristic)
        changes = []
        for i in range(1, len(sorted_values)):
            prev_val = sorted_values[i-1].value
            curr_val = sorted_values[i].value
            
            if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                if prev_val != 0:
                    change_ratio = abs(curr_val - prev_val) / abs(prev_val)
                    changes.append(change_ratio)
        
        if not changes:
            return 1.0
        
        # Consider values consistent if most changes are < 50%
        consistent_changes = sum(1 for change in changes if change < 0.5)
        consistency = consistent_changes / len(changes)
        
        return consistency

    # Cache management methods
    def _get_cache_key(
        self,
        feature_id: str,
        entity_ids: List[str] = None,
        start_time: datetime = None,
        end_time: datetime = None,
        version: str = None
    ) -> str:
        """Generate cache key for feature values"""
        
        key_parts = [feature_id]
        
        if entity_ids:
            key_parts.append(f"entities:{'|'.join(sorted(entity_ids))}")
        
        if start_time:
            key_parts.append(f"start:{start_time.isoformat()}")
        
        if end_time:
            key_parts.append(f"end:{end_time.isoformat()}")
        
        if version:
            key_parts.append(f"version:{version}")
        
        cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
        return f"feature_store:{cache_key}"

    async def _get_from_cache(self, cache_key: str) -> Optional[List[FeatureValue]]:
        """Get feature values from cache"""
        
        if not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                values_data = json.loads(cached_data)
                values = []
                for value_data in values_data:
                    value = FeatureValue(
                        feature_id=value_data['feature_id'],
                        entity_id=value_data['entity_id'],
                        value=value_data['value'],
                        timestamp=datetime.fromisoformat(value_data['timestamp']),
                        version=value_data['version'],
                        metadata=value_data.get('metadata')
                    )
                    values.append(value)
                return values
        except Exception as e:
            self.logger.warning(f"Cache read error: {e}")
        
        return None

    async def _store_in_cache(self, cache_key: str, values: List[FeatureValue], ttl: int = 3600) -> bool:
        """Store feature values in cache"""
        
        if not self.redis_client:
            return False
        
        try:
            values_data = []
            for value in values:
                value_data = {
                    'feature_id': value.feature_id,
                    'entity_id': value.entity_id,
                    'value': value.value,
                    'timestamp': value.timestamp.isoformat(),
                    'version': value.version,
                    'metadata': value.metadata
                }
                values_data.append(value_data)
            
            self.redis_client.setex(cache_key, ttl, json.dumps(values_data))
            return True
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")
            return False

    async def _clear_feature_cache(self, feature_id: str) -> bool:
        """Clear cache for a specific feature"""
        
        if not self.redis_client:
            return False
        
        try:
            # Find all cache keys for this feature
            pattern = f"feature_store:*{feature_id}*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                self.redis_client.delete(*keys)
            
            return True
        except Exception as e:
            self.logger.warning(f"Cache clear error: {e}")
            return False

    # Helper methods
    async def _load_feature_definition(self, feature_id: str) -> bool:
        """Load feature definition from database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM feature_definitions WHERE feature_id = ?
            ''', (feature_id,))
            
            row = cursor.fetchone()
            if not row:
                return False
            
            feature_def = FeatureDefinition(
                feature_id=row[0],
                name=row[1],
                description=row[2],
                feature_type=FeatureType(row[3]),
                entity_type=row[4],
                source_table=row[5],
                computation_logic=row[6],
                dependencies=json.loads(row[7]) if row[7] else [],
                version=row[8],
                status=FeatureStatus(row[9]),
                created_at=datetime.fromisoformat(row[10]),
                created_by=row[11],
                tags=json.loads(row[12]) if row[12] else [],
                validation_rules=json.loads(row[13]) if row[13] else {},
                transformation_config=json.loads(row[14]) if row[14] else {}
            )
            
            self.feature_definitions[feature_id] = feature_def
            return True

    async def _load_feature_set(self, feature_set_id: str) -> bool:
        """Load feature set from database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM feature_sets WHERE feature_set_id = ?
            ''', (feature_set_id,))
            
            row = cursor.fetchone()
            if not row:
                return False
            
            feature_set = FeatureSet(
                feature_set_id=row[0],
                name=row[1],
                description=row[2],
                features=json.loads(row[3]),
                entity_type=row[4],
                created_at=datetime.fromisoformat(row[5]),
                version=row[6]
            )
            
            self.feature_sets[feature_set_id] = feature_set
            return True

    async def _get_feature_name(self, feature_id: str) -> str:
        """Get feature name by ID"""
        
        if feature_id in self.feature_definitions:
            return self.feature_definitions[feature_id].name
        
        await self._load_feature_definition(feature_id)
        
        if feature_id in self.feature_definitions:
            return self.feature_definitions[feature_id].name
        
        return feature_id  # Fallback to ID

# Example usage
async def example_feature_store():
    """Example feature store usage"""
    
    store = FeatureStore()
    
    # Register Echo decision confidence feature
    confidence_feature_id = await store.register_feature(
        name="echo_decision_confidence",
        description="Confidence score of Echo's decisions",
        feature_type=FeatureType.NUMERICAL,
        entity_type="decision",
        validation_rules={
            "min_value": 0.0,
            "max_value": 1.0
        }
    )
    
    # Register user interaction frequency feature
    interaction_feature_id = await store.register_feature(
        name="user_interaction_frequency",
        description="Frequency of user interactions per hour",
        feature_type=FeatureType.NUMERICAL,
        entity_type="user"
    )
    
    # Store some values
    values = [
        FeatureValue(
            feature_id=confidence_feature_id,
            entity_id="decision_001",
            value=0.85,
            timestamp=datetime.now(),
            version="1.0.0"
        ),
        FeatureValue(
            feature_id=confidence_feature_id,
            entity_id="decision_002",
            value=0.92,
            timestamp=datetime.now(),
            version="1.0.0"
        )
    ]
    
    await store.store_feature_values(confidence_feature_id, values)
    
    # Create feature set for Echo model
    feature_set_id = await store.create_feature_set(
        name="echo_decision_features",
        description="Features for Echo decision making",
        features=[confidence_feature_id, interaction_feature_id],
        entity_type="decision"
    )
    
    # Get feature set values
    df = await store.get_feature_set_values(
        feature_set_id=feature_set_id,
        entity_ids=["decision_001", "decision_002"]
    )
    
    print(f"Feature set DataFrame:\n{df}")
    
    # Monitor feature quality
    quality_metrics = await store.monitor_feature_quality(confidence_feature_id)
    print(f"Quality metrics: {quality_metrics}")

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI, HTTPException
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # FastAPI app for feature store API
    app = FastAPI(title="Echo Feature Store", version="1.0.0")
    feature_store = FeatureStore()
    
    @app.post("/features/register")
    async def register_feature_endpoint(
        name: str,
        description: str,
        feature_type: str,
        entity_type: str,
        validation_rules: Dict[str, Any] = None
    ):
        try:
            feature_type_enum = FeatureType(feature_type)
            feature_id = await feature_store.register_feature(
                name, description, feature_type_enum, entity_type,
                validation_rules=validation_rules
            )
            return {"feature_id": feature_id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/features/{feature_id}/values")
    async def store_values_endpoint(feature_id: str, values: List[Dict[str, Any]]):
        try:
            feature_values = []
            for value_data in values:
                feature_value = FeatureValue(
                    feature_id=feature_id,
                    entity_id=value_data['entity_id'],
                    value=value_data['value'],
                    timestamp=datetime.fromisoformat(value_data['timestamp']),
                    version=value_data.get('version', '1.0.0')
                )
                feature_values.append(feature_value)
            
            success = await feature_store.store_feature_values(feature_id, feature_values)
            return {"success": success}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/features/{feature_id}/values")
    async def get_values_endpoint(
        feature_id: str,
        entity_ids: List[str] = None,
        start_time: str = None,
        end_time: str = None
    ):
        try:
            start_dt = datetime.fromisoformat(start_time) if start_time else None
            end_dt = datetime.fromisoformat(end_time) if end_time else None
            
            values = await feature_store.get_feature_values(
                feature_id, entity_ids, start_dt, end_dt
            )
            
            return {
                "values": [
                    {
                        "entity_id": v.entity_id,
                        "value": v.value,
                        "timestamp": v.timestamp.isoformat(),
                        "version": v.version
                    }
                    for v in values
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/feature-sets")
    async def create_feature_set_endpoint(
        name: str,
        description: str,
        features: List[str],
        entity_type: str
    ):
        try:
            feature_set_id = await feature_store.create_feature_set(
                name, description, features, entity_type
            )
            return {"feature_set_id": feature_set_id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/feature-sets/{feature_set_id}/values")
    async def get_feature_set_values_endpoint(
        feature_set_id: str,
        entity_ids: List[str]
    ):
        try:
            df = await feature_store.get_feature_set_values(feature_set_id, entity_ids)
            return {"data": df.to_dict('records')}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/features/{feature_id}/quality")
    async def get_feature_quality_endpoint(feature_id: str):
        try:
            quality_metrics = await feature_store.monitor_feature_quality(feature_id)
            return quality_metrics
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8344)
