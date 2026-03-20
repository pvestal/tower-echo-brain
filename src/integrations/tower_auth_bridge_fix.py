import json
from pathlib import Path

async def load_existing_tokens(self):
    """Load existing OAuth tokens from credentials file"""
    try:
        # Read Google credentials directly from file
        creds_file = Path("/opt/tower-auth/credentials/google_credentials.json")
        if creds_file.exists():
            with open(creds_file) as f:
                google_creds = json.load(f)
                
            self.cached_tokens["google"] = {
                "access_token": google_creds.get("token"),
                "refresh_token": google_creds.get("refresh_token"),
                "scopes": google_creds.get("scopes", [])
            }
            
            logger.info(f"✅ Loaded Google OAuth token from credentials file")
            return self.cached_tokens
            
    except Exception as e:
        logger.error(f"Failed to load credentials file: {e}")
    
    return {}
