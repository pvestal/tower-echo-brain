#!/usr/bin/env python3
"""
User Preferences System for Echo Brain Board of Directors
Manages user-specific settings, constraints, and customization options
"""

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import psycopg2
import psycopg2.extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logger = logging.getLogger(__name__)

class PreferenceType(Enum):
    DIRECTOR_WEIGHT = "director_weight"
    RISK_TOLERANCE = "risk_tolerance"
    AUTOMATION_LEVEL = "automation_level"
    NOTIFICATION_SETTING = "notification_setting"
    APPROVAL_THRESHOLD = "approval_threshold"
    ESCALATION_RULE = "escalation_rule"
    CONTEXT_FILTER = "context_filter"
    PRIORITY_RULE = "priority_rule"

class ConstraintType(Enum):
    TIME_LIMIT = "time_limit"
    RESOURCE_LIMIT = "resource_limit"
    APPROVAL_REQUIRED = "approval_required"
    BLACKLIST_PATTERN = "blacklist_pattern"
    WHITELIST_PATTERN = "whitelist_pattern"
    DIRECTOR_EXCLUSION = "director_exclusion"
    MINIMUM_CONSENSUS = "minimum_consensus"

@dataclass
class UserPreference:
    """Individual user preference setting"""
    preference_id: str
    user_id: str
    preference_type: PreferenceType
    key: str
    value: Any
    description: str
    created_at: datetime
    updated_at: datetime
    active: bool = True
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class UserConstraint:
    """User-defined constraint for board decisions"""
    constraint_id: str
    user_id: str
    constraint_type: ConstraintType
    rule_expression: str
    description: str
    priority: int  # 1-10, higher is more important
    active: bool
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PreferenceProfile:
    """Complete user preference profile"""
    user_id: str
    display_name: str
    preferences: Dict[str, Any]
    constraints: List[UserConstraint]
    director_weights: Dict[str, float]
    risk_tolerance: float  # 0.0 (risk-averse) to 1.0 (risk-tolerant)
    automation_level: float  # 0.0 (manual) to 1.0 (fully automated)
    last_updated: datetime
    profile_version: str

class UserPreferences:
    """
    Manages user preferences, constraints, and customization settings
    for the Echo Brain Board of Directors system
    """

    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize UserPreferences system

        Args:
            db_config: Database connection parameters
        """
        self.db_config = db_config
        self.preference_cache: Dict[str, PreferenceProfile] = {}
        self.default_preferences = self._get_default_preferences()

        self._initialize_database()

    def _initialize_database(self):
        """Initialize user preferences database tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    preference_id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    preference_type VARCHAR(50) NOT NULL,
                    key VARCHAR(100) NOT NULL,
                    value JSONB NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    active BOOLEAN DEFAULT TRUE,
                    metadata JSONB,
                    UNIQUE(user_id, preference_type, key)
                );
            """)

            # User constraints table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_constraints (
                    constraint_id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    constraint_type VARCHAR(50) NOT NULL,
                    rule_expression TEXT NOT NULL,
                    description TEXT,
                    priority INTEGER DEFAULT 5,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB
                );
            """)

            # User profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id VARCHAR(255) PRIMARY KEY,
                    display_name VARCHAR(255),
                    risk_tolerance FLOAT DEFAULT 0.5,
                    automation_level FLOAT DEFAULT 0.7,
                    profile_version VARCHAR(20) DEFAULT '1.0',
                    last_updated TIMESTAMP DEFAULT NOW(),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # Preference templates table (for defaults and sharing)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS preference_templates (
                    template_id VARCHAR(255) PRIMARY KEY,
                    template_name VARCHAR(255) NOT NULL,
                    description TEXT,
                    template_data JSONB NOT NULL,
                    created_by VARCHAR(255),
                    is_public BOOLEAN DEFAULT FALSE,
                    usage_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_constraints_user_id ON user_constraints(user_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_preferences_type ON user_preferences(preference_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_constraints_type ON user_constraints(constraint_type);")

            conn.close()
            logger.info("User preferences database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize user preferences database: {e}")
            raise

    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default preference values"""
        return {
            "director_weights": {
                "security_director": 1.0,
                "performance_director": 1.0,
                "quality_director": 1.0,
                "ethics_director": 1.0,
                "ux_director": 1.0
            },
            "risk_tolerance": 0.5,
            "automation_level": 0.7,
            "approval_threshold": 0.8,
            "minimum_consensus": 0.7,
            "notification_settings": {
                "email_enabled": True,
                "realtime_updates": True,
                "summary_frequency": "daily"
            },
            "escalation_rules": {
                "high_risk_threshold": 0.8,
                "time_limit_minutes": 30,
                "require_approval_above_risk": 0.9
            }
        }

    def create_user_profile(self, user_id: str, display_name: str = None) -> PreferenceProfile:
        """
        Create a new user preference profile with defaults

        Args:
            user_id: User identifier
            display_name: Optional display name

        Returns:
            PreferenceProfile: New user profile
        """
        try:
            # Check if profile already exists
            existing_profile = self.get_user_profile(user_id)
            if existing_profile:
                return existing_profile

            # Create new profile
            profile = PreferenceProfile(
                user_id=user_id,
                display_name=display_name or f"User {user_id}",
                preferences=self.default_preferences.copy(),
                constraints=[],
                director_weights=self.default_preferences["director_weights"].copy(),
                risk_tolerance=self.default_preferences["risk_tolerance"],
                automation_level=self.default_preferences["automation_level"],
                last_updated=datetime.utcnow(),
                profile_version="1.0"
            )

            # Save to database
            self._save_profile_to_db(profile)

            # Create default preferences
            for pref_key, pref_value in self.default_preferences.items():
                if pref_key not in ["director_weights"]:  # Handle director weights separately
                    preference = UserPreference(
                        preference_id=str(uuid.uuid4()),
                        user_id=user_id,
                        preference_type=self._get_preference_type(pref_key),
                        key=pref_key,
                        value=pref_value,
                        description=f"Default {pref_key} setting",
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    self._save_preference_to_db(preference)

            # Create director weight preferences
            for director_id, weight in self.default_preferences["director_weights"].items():
                preference = UserPreference(
                    preference_id=str(uuid.uuid4()),
                    user_id=user_id,
                    preference_type=PreferenceType.DIRECTOR_WEIGHT,
                    key=director_id,
                    value=weight,
                    description=f"Weight for {director_id}",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                self._save_preference_to_db(preference)

            # Cache profile
            self.preference_cache[user_id] = profile

            logger.info(f"Created user profile for {user_id}")
            return profile

        except Exception as e:
            logger.error(f"Failed to create user profile for {user_id}: {e}")
            raise

    def get_user_profile(self, user_id: str) -> Optional[PreferenceProfile]:
        """
        Get complete user preference profile

        Args:
            user_id: User identifier

        Returns:
            PreferenceProfile or None if not found
        """
        try:
            # Check cache first
            if user_id in self.preference_cache:
                return self.preference_cache[user_id]

            # Load from database
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Get profile basics
            cursor.execute("""
                SELECT * FROM user_profiles WHERE user_id = %s
            """, (user_id,))

            profile_row = cursor.fetchone()
            if not profile_row:
                conn.close()
                return None

            # Get preferences
            cursor.execute("""
                SELECT * FROM user_preferences
                WHERE user_id = %s AND active = TRUE
                ORDER BY preference_type, key
            """, (user_id,))

            preference_rows = cursor.fetchall()

            # Get constraints
            cursor.execute("""
                SELECT * FROM user_constraints
                WHERE user_id = %s AND active = TRUE
                ORDER BY priority DESC, created_at
            """, (user_id,))

            constraint_rows = cursor.fetchall()
            conn.close()

            # Build preferences dict
            preferences = {}
            director_weights = {}

            for row in preference_rows:
                if row['preference_type'] == PreferenceType.DIRECTOR_WEIGHT.value:
                    director_weights[row['key']] = row['value']
                else:
                    preferences[row['key']] = row['value']

            # Build constraints list
            constraints = []
            for row in constraint_rows:
                constraint = UserConstraint(
                    constraint_id=row['constraint_id'],
                    user_id=row['user_id'],
                    constraint_type=ConstraintType(row['constraint_type']),
                    rule_expression=row['rule_expression'],
                    description=row['description'] or "",
                    priority=row['priority'],
                    active=row['active'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=row['metadata']
                )
                constraints.append(constraint)

            # Build complete profile
            profile = PreferenceProfile(
                user_id=profile_row['user_id'],
                display_name=profile_row['display_name'],
                preferences=preferences,
                constraints=constraints,
                director_weights=director_weights,
                risk_tolerance=profile_row['risk_tolerance'],
                automation_level=profile_row['automation_level'],
                last_updated=profile_row['last_updated'],
                profile_version=profile_row['profile_version']
            )

            # Cache profile
            self.preference_cache[user_id] = profile

            return profile

        except Exception as e:
            logger.error(f"Failed to get user profile for {user_id}: {e}")
            return None

    def update_preference(self, user_id: str, preference_type: PreferenceType,
                         key: str, value: Any, description: str = None) -> bool:
        """
        Update or create a user preference

        Args:
            user_id: User identifier
            preference_type: Type of preference
            key: Preference key
            value: Preference value
            description: Optional description

        Returns:
            bool: True if successfully updated
        """
        try:
            preference = UserPreference(
                preference_id=str(uuid.uuid4()),
                user_id=user_id,
                preference_type=preference_type,
                key=key,
                value=value,
                description=description or f"{key} preference",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Save to database (will update if exists due to unique constraint)
            self._save_preference_to_db(preference, upsert=True)

            # Update cache if profile is loaded
            if user_id in self.preference_cache:
                profile = self.preference_cache[user_id]
                if preference_type == PreferenceType.DIRECTOR_WEIGHT:
                    profile.director_weights[key] = value
                else:
                    profile.preferences[key] = value
                profile.last_updated = datetime.utcnow()

            logger.info(f"Updated preference {key} for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update preference for {user_id}: {e}")
            return False

    def add_constraint(self, user_id: str, constraint_type: ConstraintType,
                      rule_expression: str, description: str, priority: int = 5) -> bool:
        """
        Add a user constraint

        Args:
            user_id: User identifier
            constraint_type: Type of constraint
            rule_expression: Rule expression (e.g., SQL-like, regex)
            description: Human-readable description
            priority: Priority level (1-10)

        Returns:
            bool: True if successfully added
        """
        try:
            constraint = UserConstraint(
                constraint_id=str(uuid.uuid4()),
                user_id=user_id,
                constraint_type=constraint_type,
                rule_expression=rule_expression,
                description=description,
                priority=priority,
                active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            self._save_constraint_to_db(constraint)

            # Update cache if profile is loaded
            if user_id in self.preference_cache:
                self.preference_cache[user_id].constraints.append(constraint)
                self.preference_cache[user_id].last_updated = datetime.utcnow()

            logger.info(f"Added constraint {constraint.constraint_id} for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add constraint for {user_id}: {e}")
            return False

    def remove_constraint(self, user_id: str, constraint_id: str) -> bool:
        """
        Remove or deactivate a user constraint

        Args:
            user_id: User identifier
            constraint_id: Constraint identifier

        Returns:
            bool: True if successfully removed
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE user_constraints
                SET active = FALSE, updated_at = NOW()
                WHERE constraint_id = %s AND user_id = %s
            """, (constraint_id, user_id))

            if cursor.rowcount > 0:
                conn.commit()

                # Update cache if profile is loaded
                if user_id in self.preference_cache:
                    profile = self.preference_cache[user_id]
                    profile.constraints = [c for c in profile.constraints if c.constraint_id != constraint_id]
                    profile.last_updated = datetime.utcnow()

                logger.info(f"Removed constraint {constraint_id} for user {user_id}")
                result = True
            else:
                result = False

            conn.close()
            return result

        except Exception as e:
            logger.error(f"Failed to remove constraint {constraint_id} for {user_id}: {e}")
            return False

    def get_effective_director_weights(self, user_id: str, context: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Get effective director weights considering user preferences and context

        Args:
            user_id: User identifier
            context: Optional context for dynamic adjustments

        Returns:
            Dict mapping director_id to weight
        """
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                return self.default_preferences["director_weights"].copy()

            weights = profile.director_weights.copy()

            # Apply context-based adjustments
            if context:
                # Example: Increase security director weight for high-risk tasks
                risk_level = context.get("risk_level", 0.5)
                if risk_level > 0.7 and "security_director" in weights:
                    weights["security_director"] *= (1.0 + risk_level * 0.5)

                # Example: Increase performance director weight for performance-critical tasks
                if context.get("performance_critical", False) and "performance_director" in weights:
                    weights["performance_director"] *= 1.3

            # Normalize weights to ensure they sum to reasonable total
            total_weight = sum(weights.values())
            if total_weight > 0:
                # Don't normalize to 1.0, but keep reasonable bounds
                max_total = len(weights) * 2.0  # Allow up to 2x base weight per director
                if total_weight > max_total:
                    factor = max_total / total_weight
                    weights = {k: v * factor for k, v in weights.items()}

            return weights

        except Exception as e:
            logger.error(f"Failed to get effective director weights for {user_id}: {e}")
            return self.default_preferences["director_weights"].copy()

    def check_constraints(self, user_id: str, task_context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if task context satisfies user constraints

        Args:
            user_id: User identifier
            task_context: Task context to validate

        Returns:
            Tuple of (constraints_satisfied, violation_messages)
        """
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                return True, []

            violations = []

            for constraint in profile.constraints:
                if not constraint.active:
                    continue

                satisfied = self._evaluate_constraint(constraint, task_context)
                if not satisfied:
                    violations.append(f"Constraint violation: {constraint.description}")

            return len(violations) == 0, violations

        except Exception as e:
            logger.error(f"Failed to check constraints for {user_id}: {e}")
            return True, []  # Default to allowing if check fails

    def export_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Export user preferences for backup or sharing

        Args:
            user_id: User identifier

        Returns:
            Dict containing all user preferences
        """
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                return {}

            export_data = {
                "user_id": profile.user_id,
                "display_name": profile.display_name,
                "preferences": profile.preferences,
                "director_weights": profile.director_weights,
                "risk_tolerance": profile.risk_tolerance,
                "automation_level": profile.automation_level,
                "constraints": [
                    {
                        "constraint_type": c.constraint_type.value,
                        "rule_expression": c.rule_expression,
                        "description": c.description,
                        "priority": c.priority
                    }
                    for c in profile.constraints
                ],
                "export_timestamp": datetime.utcnow().isoformat(),
                "profile_version": profile.profile_version
            }

            return export_data

        except Exception as e:
            logger.error(f"Failed to export preferences for {user_id}: {e}")
            return {}

    def import_preferences(self, user_id: str, preferences_data: Dict[str, Any]) -> bool:
        """
        Import user preferences from backup or template

        Args:
            user_id: User identifier
            preferences_data: Preferences data to import

        Returns:
            bool: True if successfully imported
        """
        try:
            # Create or update profile
            self.create_user_profile(user_id, preferences_data.get("display_name"))

            # Update basic preferences
            for key, value in preferences_data.get("preferences", {}).items():
                pref_type = self._get_preference_type(key)
                self.update_preference(user_id, pref_type, key, value)

            # Update director weights
            for director_id, weight in preferences_data.get("director_weights", {}).items():
                self.update_preference(user_id, PreferenceType.DIRECTOR_WEIGHT, director_id, weight)

            # Update profile settings
            if "risk_tolerance" in preferences_data:
                self._update_profile_field(user_id, "risk_tolerance", preferences_data["risk_tolerance"])

            if "automation_level" in preferences_data:
                self._update_profile_field(user_id, "automation_level", preferences_data["automation_level"])

            # Import constraints
            for constraint_data in preferences_data.get("constraints", []):
                self.add_constraint(
                    user_id=user_id,
                    constraint_type=ConstraintType(constraint_data["constraint_type"]),
                    rule_expression=constraint_data["rule_expression"],
                    description=constraint_data["description"],
                    priority=constraint_data.get("priority", 5)
                )

            # Clear cache to force reload
            self.preference_cache.pop(user_id, None)

            logger.info(f"Imported preferences for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to import preferences for {user_id}: {e}")
            return False

    def get_preference_templates(self) -> List[Dict[str, Any]]:
        """Get available preference templates"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute("""
                SELECT template_id, template_name, description, usage_count
                FROM preference_templates
                WHERE is_public = TRUE
                ORDER BY usage_count DESC, template_name
            """)

            templates = cursor.fetchall()
            conn.close()

            return [dict(template) for template in templates]

        except Exception as e:
            logger.error(f"Failed to get preference templates: {e}")
            return []

    def _evaluate_constraint(self, constraint: UserConstraint, context: Dict[str, Any]) -> bool:
        """Evaluate if constraint is satisfied by context"""
        try:
            rule = constraint.rule_expression.lower()

            if constraint.constraint_type == ConstraintType.TIME_LIMIT:
                # Example: "max_time < 30" (minutes)
                if "max_time" in rule:
                    max_time = context.get("estimated_time_minutes", 0)
                    if ">" in rule:
                        threshold = float(rule.split(">")[1].strip())
                        return max_time > threshold
                    elif "<" in rule:
                        threshold = float(rule.split("<")[1].strip())
                        return max_time < threshold

            elif constraint.constraint_type == ConstraintType.BLACKLIST_PATTERN:
                # Example: pattern to avoid in task description
                task_desc = context.get("task_description", "").lower()
                return rule not in task_desc

            elif constraint.constraint_type == ConstraintType.MINIMUM_CONSENSUS:
                # Example: "consensus >= 0.8"
                consensus = context.get("current_consensus", 1.0)
                if ">=" in rule:
                    threshold = float(rule.split(">=")[1].strip())
                    return consensus >= threshold

            # Add more constraint evaluations as needed
            return True  # Default to satisfied if can't evaluate

        except Exception as e:
            logger.error(f"Failed to evaluate constraint: {e}")
            return True  # Default to satisfied on error

    def _get_preference_type(self, key: str) -> PreferenceType:
        """Map preference key to PreferenceType enum"""
        mapping = {
            "risk_tolerance": PreferenceType.RISK_TOLERANCE,
            "automation_level": PreferenceType.AUTOMATION_LEVEL,
            "approval_threshold": PreferenceType.APPROVAL_THRESHOLD,
            "notification_settings": PreferenceType.NOTIFICATION_SETTING,
            "escalation_rules": PreferenceType.ESCALATION_RULE
        }
        return mapping.get(key, PreferenceType.CONTEXT_FILTER)

    def _save_profile_to_db(self, profile: PreferenceProfile):
        """Save profile to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO user_profiles (
                    user_id, display_name, risk_tolerance, automation_level,
                    profile_version, last_updated
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    display_name = EXCLUDED.display_name,
                    risk_tolerance = EXCLUDED.risk_tolerance,
                    automation_level = EXCLUDED.automation_level,
                    profile_version = EXCLUDED.profile_version,
                    last_updated = EXCLUDED.last_updated
            """, (
                profile.user_id, profile.display_name, profile.risk_tolerance,
                profile.automation_level, profile.profile_version, profile.last_updated
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to save profile to database: {e}")

    def _save_preference_to_db(self, preference: UserPreference, upsert: bool = False):
        """Save preference to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            if upsert:
                cursor.execute("""
                    INSERT INTO user_preferences (
                        preference_id, user_id, preference_type, key, value,
                        description, created_at, updated_at, active, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, preference_type, key) DO UPDATE SET
                        value = EXCLUDED.value,
                        description = EXCLUDED.description,
                        updated_at = EXCLUDED.updated_at,
                        active = EXCLUDED.active,
                        metadata = EXCLUDED.metadata
                """, (
                    preference.preference_id, preference.user_id, preference.preference_type.value,
                    preference.key, json.dumps(preference.value), preference.description,
                    preference.created_at, preference.updated_at, preference.active,
                    json.dumps(preference.metadata) if preference.metadata else None
                ))
            else:
                cursor.execute("""
                    INSERT INTO user_preferences (
                        preference_id, user_id, preference_type, key, value,
                        description, created_at, updated_at, active, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    preference.preference_id, preference.user_id, preference.preference_type.value,
                    preference.key, json.dumps(preference.value), preference.description,
                    preference.created_at, preference.updated_at, preference.active,
                    json.dumps(preference.metadata) if preference.metadata else None
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to save preference to database: {e}")

    def _save_constraint_to_db(self, constraint: UserConstraint):
        """Save constraint to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO user_constraints (
                    constraint_id, user_id, constraint_type, rule_expression,
                    description, priority, active, created_at, updated_at, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                constraint.constraint_id, constraint.user_id, constraint.constraint_type.value,
                constraint.rule_expression, constraint.description, constraint.priority,
                constraint.active, constraint.created_at, constraint.updated_at,
                json.dumps(constraint.metadata) if constraint.metadata else None
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to save constraint to database: {e}")

    def _update_profile_field(self, user_id: str, field: str, value: Any):
        """Update a specific profile field"""
        try:
            # SECURITY FIX: Validate field name against allowed columns to prevent SQL injection
            allowed_fields = {
                'automation_level', 'notification_preferences', 'ui_preferences',
                'board_oversight_threshold', 'confidence_threshold', 'preferred_directors',
                'override_history_enabled', 'advanced_features_enabled'
            }

            if field not in allowed_fields:
                logger.error(f"Attempted to update invalid field '{field}' - potential SQL injection attempt")
                raise ValueError(f"Invalid field name: {field}")

            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Use psycopg2.sql.Identifier for safe column name interpolation
            from psycopg2 import sql
            query = sql.SQL("""
                UPDATE user_profiles
                SET {field} = %s, last_updated = NOW()
                WHERE user_id = %s
            """).format(field=sql.Identifier(field))

            cursor.execute(query, (value, user_id))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update profile field {field} for {user_id}: {e}")