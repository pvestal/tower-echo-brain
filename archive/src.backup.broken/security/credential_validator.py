#!/usr/bin/env python3
"""
Secure Credential Validation for Echo Brain
Implements secure credential loading with validation and fallbacks
"""

import os
import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CredentialConfig:
    """Configuration for credential validation"""
    tower_vault_path: str = "/home/patrick/.tower_credentials/vault.json"
    required_db_fields: list = None
    required_jwt_fields: list = None

    def __post_init__(self):
        if self.required_db_fields is None:
            self.required_db_fields = ["host", "database", "user", "password"]
        if self.required_jwt_fields is None:
            self.required_jwt_fields = ["secret"]


class SecureCredentialManager:
    """Secure credential management with Tower vault integration"""

    def __init__(self, config: CredentialConfig = None):
        self.config = config or CredentialConfig()
        self.credentials_cache = {}
        self._validated = False

    def validate_tower_vault(self) -> bool:
        """Validate Tower vault file exists and is readable"""
        try:
            if not os.path.exists(self.config.tower_vault_path):
                logger.warning(f"⚠️ Tower vault not found: {self.config.tower_vault_path}")
                return False

            # Check file permissions
            if not os.access(self.config.tower_vault_path, os.R_OK):
                logger.error(f"❌ Tower vault not readable: {self.config.tower_vault_path}")
                return False

            # Test JSON validity
            with open(self.config.tower_vault_path, 'r') as f:
                json.load(f)

            logger.info("✅ Tower vault validated successfully")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"❌ Tower vault JSON corruption: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Tower vault validation failed: {e}")
            return False

    def load_tower_vault(self) -> Optional[Dict[str, Any]]:
        """Load credentials from Tower vault with validation"""
        try:
            if not self.validate_tower_vault():
                return None

            with open(self.config.tower_vault_path, 'r') as f:
                vault_data = json.load(f)

            logger.info("✅ Tower vault loaded successfully")
            return vault_data

        except Exception as e:
            logger.error(f"❌ Failed to load Tower vault: {e}")
            return None

    def get_database_credentials(self) -> Dict[str, Any]:
        """Get database credentials with secure fallbacks"""

        # Priority 1: Tower vault
        vault_data = self.load_tower_vault()
        if vault_data and "postgresql" in vault_data:
            db_config = vault_data["postgresql"]

            # Validate required fields
            missing_fields = [
                field for field in self.config.required_db_fields
                if field not in db_config or not db_config[field]
            ]

            if not missing_fields:
                logger.info("✅ Database credentials loaded from Tower vault")
                return {
                    "database": db_config["database"],
                    "user": db_config["user"],
                    "host": db_config["host"],
                    "password": db_config["password"],
                    "port": int(db_config.get("port", 5432))
                }
            else:
                logger.warning(f"⚠️ Tower vault missing required database fields: {missing_fields}")

        # Priority 2: Environment variables (NO hardcoded fallbacks)
        db_password = os.environ.get("DB_PASSWORD")
        if not db_password:
            raise ValueError(
                "❌ CRITICAL: Database password not found in Tower vault or DB_PASSWORD environment variable. "
                "Configure Tower vault or set DB_PASSWORD environment variable."
            )

        logger.warning("⚠️ Using environment variables for database credentials - consider configuring Tower vault")
        return {
            "database": os.environ.get("DB_NAME", "echo_brain"),
            "user": os.environ.get("DB_USER", "patrick"),
            "host": os.environ.get("DB_HOST", "localhost"),
            "password": db_password,
            "port": int(os.environ.get("DB_PORT", 5432))
        }

    def get_jwt_secret(self) -> str:
        """Get JWT secret with secure fallbacks"""

        # Priority 1: Tower vault (if JWT section exists)
        vault_data = self.load_tower_vault()
        if vault_data and "jwt" in vault_data and "secret" in vault_data["jwt"]:
            logger.info("✅ JWT secret loaded from Tower vault")
            return vault_data["jwt"]["secret"]

        # Priority 2: Environment variable
        jwt_secret = os.environ.get("JWT_SECRET")
        if not jwt_secret:
            raise ValueError(
                "❌ CRITICAL: JWT secret not found in Tower vault or JWT_SECRET environment variable. "
                "Configure Tower vault or set JWT_SECRET environment variable."
            )

        logger.warning("⚠️ Using JWT_SECRET environment variable - consider adding to Tower vault")
        return jwt_secret

    def get_service_credentials(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get credentials for a specific service"""
        vault_data = self.load_tower_vault()
        if vault_data and service_name in vault_data:
            logger.info(f"✅ {service_name} credentials loaded from Tower vault")
            return vault_data[service_name]

        logger.warning(f"⚠️ {service_name} credentials not found in Tower vault")
        return None

    def validate_all_credentials(self) -> Dict[str, bool]:
        """Validate all critical credentials are available"""
        validation_results = {}

        try:
            # Test database credentials
            db_creds = self.get_database_credentials()
            validation_results["database"] = bool(
                db_creds and all(
                    field in db_creds and db_creds[field]
                    for field in ["host", "database", "user", "password"]
                )
            )
        except Exception as e:
            logger.error(f"❌ Database credential validation failed: {e}")
            validation_results["database"] = False

        try:
            # Test JWT secret
            jwt_secret = self.get_jwt_secret()
            validation_results["jwt"] = bool(jwt_secret and len(jwt_secret) >= 32)
        except Exception as e:
            logger.error(f"❌ JWT secret validation failed: {e}")
            validation_results["jwt"] = False

        # Test Tower vault
        validation_results["tower_vault"] = self.validate_tower_vault()

        return validation_results

    def security_audit_report(self) -> str:
        """Generate security audit report"""
        validation = self.validate_all_credentials()

        report = ["🔒 CREDENTIAL SECURITY AUDIT REPORT", "=" * 50]

        # Overall status
        total_checks = len(validation)
        passed_checks = sum(validation.values())
        status = "✅ SECURE" if passed_checks == total_checks else "⚠️ ISSUES FOUND"

        report.append(f"Overall Status: {status} ({passed_checks}/{total_checks} checks passed)")
        report.append("")

        # Detailed results
        report.append("Credential Validation Results:")
        for check, passed in validation.items():
            status_icon = "✅" if passed else "❌"
            report.append(f"  {status_icon} {check.replace('_', ' ').title()}")

        report.append("")

        # Security recommendations
        report.append("Security Recommendations:")
        if not validation.get("tower_vault", False):
            report.append("  🔧 Configure Tower vault for centralized credential management")
        if not validation.get("database", False):
            report.append("  🔧 Ensure database credentials are properly configured")
        if not validation.get("jwt", False):
            report.append("  🔧 Configure JWT secret (minimum 32 characters)")

        return "\n".join(report)


# Global instance for easy access
credential_manager = SecureCredentialManager()


def get_secure_db_config() -> Dict[str, Any]:
    """Convenience function to get secure database configuration"""
    return credential_manager.get_database_credentials()


def get_secure_jwt_secret() -> str:
    """Convenience function to get secure JWT secret"""
    return credential_manager.get_jwt_secret()


if __name__ == "__main__":
    # Run security audit
    manager = SecureCredentialManager()
    print(manager.security_audit_report())