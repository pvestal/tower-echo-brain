#!/usr/bin/env python3
"""
Google API Client for Takeout Integration
Handles OAuth2 authentication and Google Takeout API interactions
"""

import os
import json
import time
import logging
import requests
from typing import Dict, List, Optional, Any, Iterator
from datetime import datetime, timedelta
from urllib.parse import urlencode
import google.auth.transport.requests
import google.oauth2.credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from vault_auth import VaultAuthManager

logger = logging.getLogger(__name__)

class GoogleTakeoutClient:
    """Google Takeout API client with OAuth2 authentication and retry logic"""

    def __init__(self, vault_manager: VaultAuthManager, config: Dict[str, Any]):
        """
        Initialize Google Takeout client

        Args:
            vault_manager: Vault authentication manager
            config: Configuration dictionary
        """
        self.vault_manager = vault_manager
        self.config = config
        self.credentials: Optional[google.oauth2.credentials.Credentials] = None
        self.session: Optional[requests.Session] = None

        # API configuration
        self.api_endpoint = config['google']['takeout']['api_endpoint']
        self.max_retries = config['google']['takeout']['max_retries']
        self.retry_delay = config['google']['takeout']['retry_delay']
        self.timeout = config['google']['takeout']['timeout']

        self._initialize_session()
        self._initialize_credentials()

    def _initialize_session(self) -> None:
        """Initialize HTTP session with retry strategy"""
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.info("HTTP session initialized with retry strategy")

    def _initialize_credentials(self) -> None:
        """Initialize Google OAuth2 credentials from Vault"""
        try:
            cred_data = self.vault_manager.get_google_credentials()

            if not cred_data:
                logger.error("No Google credentials found in Vault")
                return

            # Create credentials object
            self.credentials = google.oauth2.credentials.Credentials(
                token=cred_data.get('access_token'),
                refresh_token=cred_data.get('refresh_token'),
                client_id=cred_data.get('client_id'),
                client_secret=cred_data.get('client_secret'),
                token_uri=cred_data.get('token_uri', 'https://oauth2.googleapis.com/token'),
                scopes=cred_data.get('scopes', self.config['google']['oauth2']['scopes'])
            )

            # Test and refresh credentials if needed
            if self._refresh_credentials_if_needed():
                logger.info("Google credentials initialized successfully")
            else:
                logger.error("Failed to initialize or refresh Google credentials")

        except Exception as e:
            logger.error(f"Failed to initialize Google credentials: {e}")

    def _refresh_credentials_if_needed(self) -> bool:
        """
        Refresh OAuth2 credentials if they're expired or about to expire

        Returns:
            bool: True if credentials are valid/refreshed successfully
        """
        if not self.credentials:
            return False

        try:
            # Check if credentials are expired or expiring soon
            if self.credentials.expired or (
                self.credentials.expiry and
                self.credentials.expiry <= datetime.utcnow() + timedelta(minutes=5)
            ):
                logger.info("Refreshing expired Google credentials")

                # Refresh credentials
                request = google.auth.transport.requests.Request()
                self.credentials.refresh(request)

                # Store updated credentials back to Vault
                updated_creds = {
                    'access_token': self.credentials.token,
                    'refresh_token': self.credentials.refresh_token,
                    'client_id': self.credentials.client_id,
                    'client_secret': self.credentials.client_secret,
                    'token_uri': self.credentials.token_uri,
                    'scopes': self.credentials.scopes,
                    'expiry': self.credentials.expiry.isoformat() if self.credentials.expiry else None
                }

                self.vault_manager.store_google_credentials(updated_creds)
                logger.info("Credentials refreshed and stored in Vault")

            return True

        except Exception as e:
            logger.error(f"Failed to refresh credentials: {e}")
            return False

    def is_authenticated(self) -> bool:
        """Check if client is properly authenticated"""
        return (
            self.credentials is not None and
            self.credentials.valid and
            not self.credentials.expired
        )

    def create_takeout_archive(self, products: List[str], format_options: Dict[str, str] = None) -> Optional[str]:
        """
        Create a new Google Takeout archive

        Args:
            products: List of Google products to include (e.g., ['Photos', 'Drive', 'Gmail'])
            format_options: Optional format preferences for each product

        Returns:
            str: Archive ID if successful, None otherwise
        """
        if not self.is_authenticated():
            logger.error("Not authenticated with Google")
            return None

        try:
            # Build service for Google Takeout
            service = build('takeout', 'v1', credentials=self.credentials)

            # Prepare archive configuration
            archive_config = {
                'archiveFormat': 'ZIP',
                'products': []
            }

            for product in products:
                product_config = {'product': product}

                if format_options and product in format_options:
                    product_config['formatOptions'] = format_options[product]

                archive_config['products'].append(product_config)

            # Create archive request
            request = service.archives().create(body=archive_config)
            response = request.execute()

            if 'name' in response:
                archive_id = response['name'].split('/')[-1]
                logger.info(f"Created Takeout archive with ID: {archive_id}")
                return archive_id
            else:
                logger.error("No archive ID returned from Takeout API")
                return None

        except HttpError as e:
            logger.error(f"Google API error creating archive: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create Takeout archive: {e}")
            return None

    def get_archive_status(self, archive_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a Takeout archive

        Args:
            archive_id: ID of the archive to check

        Returns:
            Dict containing archive status information
        """
        if not self.is_authenticated():
            logger.error("Not authenticated with Google")
            return None

        try:
            service = build('takeout', 'v1', credentials=self.credentials)

            request = service.archives().get(name=f'archives/{archive_id}')
            response = request.execute()

            status_info = {
                'id': archive_id,
                'state': response.get('state', 'UNKNOWN'),
                'created_time': response.get('createTime'),
                'size_bytes': response.get('sizeBytes', 0),
                'download_urls': []
            }

            # Extract download URLs if available
            if response.get('state') == 'COMPLETED' and 'downloadUrls' in response:
                status_info['download_urls'] = response['downloadUrls']

            logger.info(f"Archive {archive_id} status: {status_info['state']}")
            return status_info

        except HttpError as e:
            logger.error(f"Google API error getting archive status: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get archive status: {e}")
            return None

    def list_archives(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List all Takeout archives for the authenticated user

        Args:
            limit: Maximum number of archives to return

        Returns:
            List of archive information dictionaries
        """
        if not self.is_authenticated():
            logger.error("Not authenticated with Google")
            return []

        try:
            service = build('takeout', 'v1', credentials=self.credentials)

            archives = []
            page_token = None

            while len(archives) < limit:
                request = service.archives().list(
                    pageSize=min(50, limit - len(archives)),
                    pageToken=page_token
                )
                response = request.execute()

                if 'archives' in response:
                    for archive in response['archives']:
                        archive_info = {
                            'id': archive['name'].split('/')[-1],
                            'state': archive.get('state', 'UNKNOWN'),
                            'created_time': archive.get('createTime'),
                            'size_bytes': archive.get('sizeBytes', 0)
                        }
                        archives.append(archive_info)

                page_token = response.get('nextPageToken')
                if not page_token:
                    break

            logger.info(f"Found {len(archives)} archives")
            return archives

        except HttpError as e:
            logger.error(f"Google API error listing archives: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to list archives: {e}")
            return []

    def get_download_urls(self, archive_id: str) -> List[str]:
        """
        Get download URLs for a completed archive

        Args:
            archive_id: ID of the completed archive

        Returns:
            List of download URLs
        """
        status = self.get_archive_status(archive_id)

        if not status:
            logger.error(f"Could not get status for archive {archive_id}")
            return []

        if status['state'] != 'COMPLETED':
            logger.warning(f"Archive {archive_id} is not completed (state: {status['state']})")
            return []

        return status.get('download_urls', [])

    def delete_archive(self, archive_id: str) -> bool:
        """
        Delete a Takeout archive

        Args:
            archive_id: ID of the archive to delete

        Returns:
            bool: True if deletion successful
        """
        if not self.is_authenticated():
            logger.error("Not authenticated with Google")
            return False

        try:
            service = build('takeout', 'v1', credentials=self.credentials)

            request = service.archives().delete(name=f'archives/{archive_id}')
            request.execute()

            logger.info(f"Successfully deleted archive {archive_id}")
            return True

        except HttpError as e:
            logger.error(f"Google API error deleting archive: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete archive: {e}")
            return False

    def wait_for_archive_completion(self, archive_id: str, poll_interval: int = 300, max_wait_time: int = 86400) -> bool:
        """
        Wait for an archive to complete processing

        Args:
            archive_id: ID of the archive to monitor
            poll_interval: Seconds between status checks (default: 5 minutes)
            max_wait_time: Maximum time to wait in seconds (default: 24 hours)

        Returns:
            bool: True if archive completed successfully
        """
        start_time = time.time()

        logger.info(f"Waiting for archive {archive_id} to complete...")

        while time.time() - start_time < max_wait_time:
            status = self.get_archive_status(archive_id)

            if not status:
                logger.error("Failed to get archive status")
                return False

            state = status['state']

            if state == 'COMPLETED':
                logger.info(f"Archive {archive_id} completed successfully")
                return True
            elif state == 'FAILED':
                logger.error(f"Archive {archive_id} failed")
                return False
            elif state in ['IN_PROGRESS', 'PENDING']:
                logger.info(f"Archive {archive_id} still processing (state: {state})")
                time.sleep(poll_interval)
            else:
                logger.warning(f"Unknown archive state: {state}")
                time.sleep(poll_interval)

        logger.error(f"Archive {archive_id} did not complete within {max_wait_time} seconds")
        return False

    def get_available_products(self) -> List[str]:
        """
        Get list of available Google products for Takeout

        Returns:
            List of product names
        """
        # Standard Google products available for Takeout
        # This list should be updated based on Google's current offerings
        return [
            'Photos',
            'Drive',
            'Gmail',
            'Calendar',
            'Contacts',
            'Keep',
            'Maps',
            'YouTube',
            'Chrome',
            'Search',
            'Location History',
            'My Activity',
            'Blogger',
            'Google Pay',
            'Play Store',
            'Hangouts',
            'Fit'
        ]

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Google API client

        Returns:
            Dict containing health status
        """
        status = {
            'authenticated': False,
            'credentials_valid': False,
            'credentials_expiry': None,
            'api_accessible': False
        }

        try:
            status['authenticated'] = self.credentials is not None
            status['credentials_valid'] = self.is_authenticated()

            if self.credentials and self.credentials.expiry:
                status['credentials_expiry'] = self.credentials.expiry.isoformat()

            # Test API accessibility by listing archives
            if self.is_authenticated():
                archives = self.list_archives(limit=1)
                status['api_accessible'] = True
                logger.info("Google Takeout API is accessible")

        except Exception as e:
            logger.error(f"Health check failed: {e}")

        return status


def create_google_client(vault_manager: VaultAuthManager, config: Dict[str, Any]) -> GoogleTakeoutClient:
    """
    Factory function to create GoogleTakeoutClient

    Args:
        vault_manager: Vault authentication manager
        config: Configuration dictionary

    Returns:
        GoogleTakeoutClient instance
    """
    return GoogleTakeoutClient(vault_manager, config)


if __name__ == "__main__":
    # Example usage and testing
    import yaml
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    # Load configuration
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    vault_manager = VaultAuthManager(config)
    google_client = create_google_client(vault_manager, config)

    print("Health Check:", google_client.health_check())
    print("Available Products:", google_client.get_available_products())