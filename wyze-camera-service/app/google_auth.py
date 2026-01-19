"""
Google OAuth integration for Wyze Camera Service
Handles Google SSO authentication for users
"""

import os
import json
from typing import Optional
from datetime import datetime, timedelta

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from fastapi import HTTPException, Request as FastAPIRequest
from fastapi.responses import RedirectResponse

class GoogleAuthManager:
    def __init__(self):
        self.client_secrets_file = os.getenv("GOOGLE_CLIENT_SECRETS", "client_secrets.json")
        self.credentials_file = os.getenv("GOOGLE_CREDENTIALS_FILE", "user_credentials.json")
        self.scopes = [
            'openid',
            'https://www.googleapis.com/auth/userinfo.email',
            'https://www.googleapis.com/auth/userinfo.profile'
        ]
        self.redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8100/auth/google/callback")
        self._credentials: Optional[Credentials] = None

    def setup_oauth_flow(self):
        """Setup Google OAuth flow"""
        try:
            # Check if client secrets file exists
            if not os.path.exists(self.client_secrets_file):
                # Create a template client secrets file
                self._create_client_secrets_template()
                raise HTTPException(
                    status_code=500,
                    detail=f"Google client secrets not configured. Please edit {self.client_secrets_file}"
                )

            flow = Flow.from_client_secrets_file(
                self.client_secrets_file,
                scopes=self.scopes,
                redirect_uri=self.redirect_uri
            )
            return flow
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OAuth setup failed: {str(e)}")

    def _create_client_secrets_template(self):
        """Create a template client secrets file"""
        template = {
            "web": {
                "client_id": "your_google_client_id_here.apps.googleusercontent.com",
                "project_id": "your-project-id",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_secret": "your_client_secret_here",
                "redirect_uris": [
                    "http://localhost:8100/auth/google/callback"
                ]
            }
        }

        with open(self.client_secrets_file, 'w') as f:
            json.dump(template, f, indent=2)

    def get_auth_url(self):
        """Get Google OAuth authorization URL"""
        flow = self.setup_oauth_flow()
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        return authorization_url, state

    def handle_oauth_callback(self, request: FastAPIRequest):
        """Handle OAuth callback and exchange code for credentials"""
        try:
            flow = self.setup_oauth_flow()

            # Get the authorization code from the callback
            authorization_response = str(request.url)
            flow.fetch_token(authorization_response=authorization_response)

            credentials = flow.credentials
            self._save_credentials(credentials)
            self._credentials = credentials

            # Get user info
            user_info = self._get_user_info(credentials)

            return {
                "success": True,
                "user": user_info,
                "message": "Google authentication successful"
            }

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"OAuth callback failed: {str(e)}")

    def _save_credentials(self, credentials: Credentials):
        """Save credentials to file"""
        credentials_dict = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes,
            'expiry': credentials.expiry.isoformat() if credentials.expiry else None
        }

        with open(self.credentials_file, 'w') as f:
            json.dump(credentials_dict, f, indent=2)

    def load_credentials(self) -> Optional[Credentials]:
        """Load saved credentials"""
        try:
            if not os.path.exists(self.credentials_file):
                return None

            with open(self.credentials_file, 'r') as f:
                cred_data = json.load(f)

            credentials = Credentials(
                token=cred_data.get('token'),
                refresh_token=cred_data.get('refresh_token'),
                token_uri=cred_data.get('token_uri'),
                client_id=cred_data.get('client_id'),
                client_secret=cred_data.get('client_secret'),
                scopes=cred_data.get('scopes')
            )

            # Set expiry if available
            if cred_data.get('expiry'):
                credentials.expiry = datetime.fromisoformat(cred_data['expiry'])

            # Refresh if expired
            if credentials.expired:
                credentials.refresh(Request())
                self._save_credentials(credentials)

            self._credentials = credentials
            return credentials

        except Exception as e:
            print(f"Error loading credentials: {e}")
            return None

    def _get_user_info(self, credentials: Credentials):
        """Get user information from Google API"""
        try:
            import requests

            headers = {'Authorization': f'Bearer {credentials.token}'}
            response = requests.get(
                'https://www.googleapis.com/oauth2/v2/userinfo',
                headers=headers
            )
            response.raise_for_status()

            return response.json()

        except Exception as e:
            print(f"Error getting user info: {e}")
            return {"email": "unknown", "name": "Unknown User"}

    def get_current_user(self):
        """Get current authenticated user"""
        if not self._credentials:
            credentials = self.load_credentials()
            if not credentials:
                return None

        user_info = self._get_user_info(self._credentials)
        return user_info

    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        credentials = self.load_credentials()
        return credentials is not None and not credentials.expired

    def logout(self):
        """Logout user and clear credentials"""
        if os.path.exists(self.credentials_file):
            os.remove(self.credentials_file)
        self._credentials = None
        return {"success": True, "message": "Logged out successfully"}

# Global auth manager instance
google_auth = GoogleAuthManager()