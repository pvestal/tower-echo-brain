#!/usr/bin/env python3
"""
Credential Rotation Script for Echo Brain
Rotates database passwords and other sensitive credentials
"""

import json
import secrets
import string
import psycopg2
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_secure_password(length=32):
    """Generate a cryptographically secure password"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-="
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def rotate_database_password():
    """Rotate the database password in Tower vault and PostgreSQL"""
    vault_path = "/home/patrick/.tower_credentials/vault.json"

    try:
        # Load current vault
        with open(vault_path, 'r') as f:
            vault_data = json.load(f)

        if 'postgresql' not in vault_data:
            logger.error("❌ PostgreSQL configuration not found in vault")
            return False

        old_config = vault_data['postgresql'].copy()
        new_password = generate_secure_password()

        # Connect with current credentials
        conn = psycopg2.connect(**old_config)
        cursor = conn.cursor()

        # Update password in PostgreSQL
        cursor.execute(f"ALTER USER {old_config['user']} PASSWORD %s;", (new_password,))
        conn.commit()

        # Update vault with new password
        vault_data['postgresql']['password'] = new_password
        vault_data['postgresql']['last_rotated'] = datetime.now().isoformat()

        # Write back to vault
        with open(vault_path, 'w') as f:
            json.dump(vault_data, f, indent=2)

        cursor.close()
        conn.close()

        logger.info("✅ Database password rotated successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Password rotation failed: {e}")
        return False


def audit_credential_security():
    """Audit current credential security status"""
    vault_path = "/home/patrick/.tower_credentials/vault.json"

    audit_results = {
        'vault_exists': False,
        'vault_readable': False,
        'weak_passwords': [],
        'missing_rotation_dates': [],
        'recommendations': []
    }

    try:
        # Check vault existence
        import os
        audit_results['vault_exists'] = os.path.exists(vault_path)
        audit_results['vault_readable'] = os.access(vault_path, os.R_OK)

        if audit_results['vault_exists'] and audit_results['vault_readable']:
            with open(vault_path, 'r') as f:
                vault_data = json.load(f)

            # Check for weak passwords
            for service, config in vault_data.items():
                if isinstance(config, dict):
                    for key, value in config.items():
                        if 'password' in key.lower() and isinstance(value, str):
                            if len(value) < 16:
                                audit_results['weak_passwords'].append(f"{service}.{key}")
                            if value in ['password', '123456', 'admin', 'secret']:
                                audit_results['weak_passwords'].append(f"{service}.{key} (common password)")

                    # Check for rotation tracking
                    if 'password' in config and 'last_rotated' not in config:
                        audit_results['missing_rotation_dates'].append(service)

            # Generate recommendations
            if audit_results['weak_passwords']:
                audit_results['recommendations'].append("Rotate weak passwords")
            if audit_results['missing_rotation_dates']:
                audit_results['recommendations'].append("Add rotation tracking to credentials")

        return audit_results

    except Exception as e:
        logger.error(f"❌ Credential audit failed: {e}")
        return audit_results


if __name__ == "__main__":
    print("🔒 CREDENTIAL ROTATION & SECURITY AUDIT")
    print("=" * 50)

    # Run security audit
    audit = audit_credential_security()

    print("Security Audit Results:")
    print(f"  Vault exists: {'✅' if audit['vault_exists'] else '❌'}")
    print(f"  Vault readable: {'✅' if audit['vault_readable'] else '❌'}")

    if audit['weak_passwords']:
        print(f"  ⚠️ Weak passwords found: {', '.join(audit['weak_passwords'])}")
    else:
        print("  ✅ No weak passwords detected")

    if audit['missing_rotation_dates']:
        print(f"  ⚠️ Missing rotation tracking: {', '.join(audit['missing_rotation_dates'])}")
    else:
        print("  ✅ All credentials have rotation tracking")

    if audit['recommendations']:
        print("  Recommendations:")
        for rec in audit['recommendations']:
            print(f"    • {rec}")

    print("\n" + "="*50)

    # Optional: Uncomment to actually rotate passwords
    # print("Rotating database password...")
    # if rotate_database_password():
    #     print("✅ Password rotation completed successfully")
    # else:
    #     print("❌ Password rotation failed")
