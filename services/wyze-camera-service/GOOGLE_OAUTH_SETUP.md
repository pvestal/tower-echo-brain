# Google OAuth Setup for Wyze Camera Service

## Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create new project or select existing project: "tower-wyze-camera"
3. Enable the Google+ API or Google Identity API

## Step 2: Configure OAuth Consent Screen

1. Go to **APIs & Services** > **OAuth consent screen**
2. Choose **External** user type
3. Fill out required fields:
   - **App name**: "Tower Wyze Camera Service"
   - **User support email**: patrick.vestal@gmail.com
   - **Developer contact**: patrick.vestal@gmail.com
4. Add scopes:
   - `../auth/userinfo.email`
   - `../auth/userinfo.profile`
   - `openid`
5. Add test users: patrick.vestal@gmail.com

## Step 3: Create OAuth Credentials

1. Go to **APIs & Services** > **Credentials**
2. Click **+ CREATE CREDENTIALS** > **OAuth client ID**
3. Choose **Web application**
4. Name: "Wyze Camera Service Web Client"
5. **Authorized redirect URIs**:
   - `http://localhost:8100/auth/google/callback`
   - `http://192.168.50.135:8100/auth/google/callback`

## Step 4: Download and Configure

1. Download the client configuration JSON file
2. Replace `/opt/tower-echo-brain/wyze-camera-service/client_secrets.json` with downloaded file
3. Or manually update the client_secrets.json with:
   - `client_id`: From Google Cloud Console
   - `client_secret`: From Google Cloud Console

## Step 5: Test Authentication Flow

1. Start the service: `cd /opt/tower-echo-brain/wyze-camera-service && source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8100`
2. Visit: `http://localhost:8100/auth/google/login`
3. Complete Google OAuth flow
4. Check auth status: `http://localhost:8100/auth/status`

## Environment Variables

Update `.env` with any additional OAuth settings:

```env
GOOGLE_CLIENT_SECRETS=client_secrets.json
GOOGLE_CREDENTIALS_FILE=user_credentials.json
GOOGLE_REDIRECT_URI=http://localhost:8100/auth/google/callback
```

## Testing

Once configured:
1. Visit `/auth/google/login` to start OAuth flow
2. Complete Google authentication
3. Service will verify Google user email matches Wyze account
4. Wyze API authentication will proceed with API keys
5. Camera controls will be available

## Security Notes

- Keep `client_secrets.json` secure and don't commit to git
- `user_credentials.json` will be created automatically to store user tokens
- Tokens are automatically refreshed when expired
- Logout clears both Google and Wyze authentication

## Troubleshooting

- **Redirect URI mismatch**: Ensure URIs in Google Console match exactly
- **Client ID not found**: Verify client_secrets.json format and content
- **Unauthorized client**: Check OAuth consent screen configuration
- **Scope errors**: Ensure proper scopes are enabled and consented