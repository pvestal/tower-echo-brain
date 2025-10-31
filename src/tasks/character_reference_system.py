"""
Character Reference System for Echo Brain
Provides access to canonical character designs and style guides
"""

import os
import glob
import logging
import psycopg2
import hvac
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class CharacterReferenceSystem:
    """Manages character references and style guides for consistent generation"""

    def __init__(self):
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_user = os.getenv('DB_USER', 'patrick')
        self.db_name = os.getenv('DB_NAME', 'anime_production')
        self.reference_base_path = "/opt/tower-anime-production/frames"
        self.vault_client = self._init_vault_client()
        self.db_password = self._get_db_password()

    def _init_vault_client(self) -> Optional[hvac.Client]:
        """Initialize HashiCorp Vault client for secure credential retrieval"""
        try:
            vault_url = os.getenv('VAULT_ADDR', 'http://127.0.0.1:8200')
            client = hvac.Client(url=vault_url)

            # Try to get token from file first, then environment
            token_paths = [
                Path('/opt/vault/.vault-token'),
                Path('/opt/vault/data/vault-token'),
                Path('/home/patrick/.vault-token')
            ]

            vault_token = os.getenv('VAULT_TOKEN')
            if not vault_token:
                for token_path in token_paths:
                    if token_path.exists():
                        vault_token = token_path.read_text().strip()
                        break

            if vault_token:
                client.token = vault_token
                if client.is_authenticated():
                    logger.info("Successfully authenticated with Vault")
                    return client
                else:
                    logger.warning("Vault token invalid or expired")
            else:
                logger.warning("No Vault token found")

        except Exception as e:
            logger.error(f"Failed to initialize Vault client: {e}")

        return None

    def _get_db_password(self) -> str:
        """Securely retrieve database password from Vault or environment"""
        # Try Vault first
        if self.vault_client:
            try:
                # Try different secret paths
                secret_paths = [
                    'secret/tower/database',
                    'secret/anime_production/database',
                    'secret/echo_brain/database'
                ]

                for path in secret_paths:
                    try:
                        response = self.vault_client.secrets.kv.v2.read_secret_version(path=path)
                        if response and 'data' in response and 'data' in response['data']:
                            password = response['data']['data'].get('password')
                            if password:
                                logger.info(f"Retrieved database password from Vault path: {path}")
                                return password
                    except Exception:
                        continue

            except Exception as e:
                logger.warning(f"Failed to retrieve password from Vault: {e}")

        # Fallback to environment variable
        env_password = os.getenv('DB_PASSWORD')
        if env_password:
            logger.info("Using database password from environment variable")
            return env_password

        # Last resort - log security warning
        logger.critical("No secure database password found - using insecure fallback. This is a SECURITY RISK!")
        return 'patrick123'  # Temporary fallback - should be removed in production

    def get_character_with_references(self, character_name: str) -> Dict[str, Any]:
        """Get character data with canonical references and style guide"""
        try:
            # Get character from database with parameterized query
            conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )

            cursor = conn.cursor()
            # Validate input to prevent injection
            if not character_name or not isinstance(character_name, str):
                logger.error(f"Invalid character name provided: {character_name}")
                return None

            # Sanitize character name - only allow alphanumeric, spaces, and common punctuation
            import re
            if not re.match(r'^[a-zA-Z0-9\s\-\.]+$', character_name):
                logger.error(f"Character name contains invalid characters: {character_name}")
                return None

            cursor.execute("""
                SELECT c.id, c.name, c.description, c.image_path, p.name as project_name
                FROM anime_api.characters c
                JOIN anime_api.projects p ON c.project_id = p.id
                WHERE c.name = %s
            """, (character_name,))

            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if not row:
                return None

            # Find canonical reference image
            reference_image = self._find_character_reference(character_name)

            # Get style guide
            style_guide = self._get_character_style_guide(character_name)

            return {
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'image_path': row[3],
                'project_name': row[4],
                'reference_image': reference_image,
                'style_guide': style_guide
            }

        except Exception as e:
            logger.error(f"Error getting character references: {e}")
            return None

    def _find_character_reference(self, character_name: str) -> Optional[str]:
        """Find canonical character design reference image with path validation"""
        # Validate and sanitize character name to prevent path traversal
        if not character_name or not isinstance(character_name, str):
            logger.error(f"Invalid character name for reference lookup: {character_name}")
            return None

        # Remove any potential path traversal characters
        import re
        sanitized_name = re.sub(r'[^a-zA-Z0-9\s\-]', '', character_name)
        if not sanitized_name:
            logger.error(f"Character name became empty after sanitization: {character_name}")
            return None

        char_slug = sanitized_name.lower().replace(' ', '_')

        # Search patterns for character design files (sanitized)
        first_name = sanitized_name.split()[0].lower() if sanitized_name.split() else char_slug
        search_patterns = [
            f"{self.reference_base_path}/**/{char_slug}_character_design*.png",
            f"{self.reference_base_path}/**/{char_slug}_design*.png",
            f"{self.reference_base_path}/**/character_design_{char_slug}*.png",
            f"{self.reference_base_path}/**/{first_name}_character_design*.png"
        ]

        logger.info(f"Searching for character reference: {character_name} -> {char_slug}")

        for pattern in search_patterns:
            logger.info(f"Trying pattern: {pattern}")
            matches = glob.glob(pattern, recursive=True)
            if matches:
                # Validate that the found path is within the allowed directory
                found_path = Path(matches[0]).resolve()
                base_path = Path(self.reference_base_path).resolve()

                try:
                    # Ensure the found file is within the reference base path
                    found_path.relative_to(base_path)
                    logger.info(f"Found reference image for {character_name}: {matches[0]}")
                    return str(found_path)
                except ValueError:
                    logger.error(f"Path traversal attempt detected: {matches[0]} is outside {self.reference_base_path}")
                    continue

        logger.warning(f"No reference image found for {character_name} using patterns: {search_patterns}")
        return None

    def _get_character_style_guide(self, character_name: str) -> Dict[str, str]:
        """Get character-specific style guide from project bible"""

        # Character-specific style guides based on project bible
        style_guides = {
            "Kai Nakamura": {
                "style": "photorealistic anime, hyperrealistic anime",
                "eye_specification": "left eye bright blue natural human eye, right eye cyan cybernetic eye with pink magenta technological glow",
                "hair": "dark black hair with electric neon blue streaks",
                "facial_features": "sharp angular face, cybernetic circuit patterns on right side of face, mechanical joints visible at jawline, detailed facial features",
                "outfit": "black tactical military vest with glowing blue circuit patterns",
                "aesthetic": "cyberpunk aesthetic, futuristic neon lighting, high detail, photorealistic rendering, professional lighting",
                "negative_prompts": "cartoon, chibi, simple anime, flat colors, 2D anime style, manga style"
            },
            "Hiroshi Yamamoto": {
                "style": "photorealistic anime",
                "aesthetic": "mentor figure, hidden agenda expression, detailed"
            },
            "Yuki": {
                "style": "photorealistic anime",
                "aesthetic": "college student, relationship dynamics, detailed"
            }
        }

        return style_guides.get(character_name, {
            "style": "photorealistic anime",
            "aesthetic": "high detail, professional lighting",
            "negative_prompts": "cartoon, chibi, simple"
        })

    def build_character_prompt(self, character_name: str) -> Dict[str, str]:
        """Build optimized prompt for character generation"""
        char_data = self.get_character_with_references(character_name)

        if not char_data:
            return {
                "prompt": f"{character_name}, photorealistic anime, high detail",
                "negative_prompt": "blurry, low quality, bad anatomy, text, watermark"
            }

        style_guide = char_data['style_guide']
        description = char_data['description']

        # Build enhanced prompt from style guide
        prompt_parts = [
            character_name,
            description,
            style_guide.get('style', 'photorealistic anime'),
            style_guide.get('eye_specification', ''),
            style_guide.get('hair', ''),
            style_guide.get('facial_features', ''),
            style_guide.get('outfit', ''),
            style_guide.get('aesthetic', 'high detail, professional lighting')
        ]

        prompt = ", ".join([part for part in prompt_parts if part])

        negative_prompt = style_guide.get('negative_prompts', 'blurry, low quality, bad anatomy, text, watermark')

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "reference_image": char_data.get('reference_image'),
            "character_data": char_data
        }