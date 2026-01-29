# CREATE NEW OAUTH CLIENT - STEP BY STEP

## Quick Link to Create:
https://console.cloud.google.com/apis/credentials/oauthclient?project=tower-echo-brain

## If that doesn't work, manual steps:

1. **Go here**: https://console.cloud.google.com

2. **Create/Select Project**:
   - Project name: `tower-echo-brain`

3. **Enable APIs** (if not already):
   https://console.cloud.google.com/apis/library
   - Google Calendar API
   - Gmail API
   - Google Photos Library API

4. **Create OAuth Client**:
   https://console.cloud.google.com/apis/credentials

   Click "CREATE CREDENTIALS" → "OAuth client ID"

   **Application type**: Web application
   **Name**: Tower Echo Brain

   **Authorized JavaScript origins**:
   ```
   http://localhost:8088
   http://localhost:8080
   http://127.0.0.1:8088
   http://192.168.50.135:8088
   ```

   **Authorized redirect URIs**:
   ```
   http://localhost:8088/api/auth/oauth/google/callback
   http://localhost:8080/
   http://127.0.0.1:8088/api/auth/oauth/google/callback
   ```

5. **Download JSON** and save as:
   `/opt/tower-auth/credentials/new_oauth.json`

6. **Run**:
   ```bash
   python3 /opt/tower-echo-brain/setup_with_new_oauth.py
   ```

That's it! The integration is ready and waiting.