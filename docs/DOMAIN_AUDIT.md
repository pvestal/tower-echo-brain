# Domain Hard-coding Audit Report

## Summary
Audit performed: 2026-02-05
Purpose: Verify no hard-coded domains in Echo Brain and Tower services

## Echo Brain Status: ✅ CLEAN

### Source Code
- **No hard-coded domains** found in Python, TypeScript, JavaScript, or Vue files
- Frontend uses relative URLs (`/api`) with environment variable fallback
- Vite config uses `localhost` for development proxy
- Built frontend assets use relative base path `/echo-brain/`

### API Endpoints
- All endpoints properly namespaced under `/api/echo/`
- Work via localhost:8309 directly
- Work via Tower Dashboard proxy at localhost:8080
- No domain dependencies

### Frontend
- Uses relative paths with `/echo-brain/` base
- API calls use relative `/api` prefix
- No hard-coded IPs or domains
- Works via localhost, IP address, or domain name

## Other Tower Services: ⚠️ NEEDS ATTENTION

### Found Hard-coded Domains

1. **Tower Dashboard** (`/opt/tower-dashboard/assets/progress_tracker.js`)
   ```javascript
   this.wsUrl = 'wss://vestal-garcia.duckdns.org/ws/progress';
   ```
   - WebSocket URL should be dynamic based on current host

2. **Tower Auth Service** (`/opt/tower-auth/auth_service.py`)
   - Multiple OAuth redirect URIs hard-coded:
     - Google: `https://vestal-garcia.duckdns.org/api/auth/oauth/google/callback`
     - GitHub: `https://vestal-garcia.duckdns.org/api/auth/oauth/github/callback`
     - Apple: `https://vestal-garcia.duckdns.org/api/auth/oauth/apple/callback`
   - Note: OAuth providers require registered redirect URIs, so this may be unavoidable

3. **Tower Anime Production** (`/opt/tower-anime-production/frontend/vite.config.js`)
   ```javascript
   allowedHosts: ['tower.local', '192.168.50.135', 'localhost', '.duckdns.org']
   ```
   - Has `.duckdns.org` wildcard in allowed hosts

## Nginx Configuration

- Server name: `vestal-garcia.duckdns.org`
- This is expected for SSL certificate matching
- Services work via localhost and IP without nginx

## Testing Results

### Via localhost:8309 (Direct)
- ✅ Health endpoint: Working
- ✅ Brain activity: Working
- ✅ Models list: Working
- ✅ Ask endpoint: Working
- ✅ Search: Working

### Via localhost:8080 (Tower Dashboard Proxy)
- ✅ Echo Brain API at `/api/echo/*`: Working
- ✅ Echo Brain Frontend at `/echo-brain/`: Working
- ✅ Assets loading correctly with relative paths

### Via IP (192.168.50.135)
- ✅ All endpoints accessible
- ✅ No domain requirements

## Recommendations

1. **Echo Brain**: No changes needed - properly domain-agnostic

2. **Tower Dashboard**:
   - Replace hard-coded WebSocket URL with dynamic host detection:
   ```javascript
   this.wsUrl = `wss://${window.location.host}/ws/progress`;
   ```

3. **Tower Auth**:
   - Consider environment variables for OAuth redirect URIs
   - May need multiple OAuth app registrations for different domains

4. **Tower Anime Production**:
   - The `.duckdns.org` wildcard is reasonable for CORS
   - No critical issues

## Conclusion

Echo Brain is completely domain-agnostic and works correctly via:
- Direct port access (localhost:8309)
- Tower Dashboard proxy (localhost:8080)
- IP address access
- Any configured domain name

Other Tower services have some hard-coded domains, primarily for:
- OAuth callbacks (may be unavoidable)
- WebSocket connections (should be fixed)
- CORS allowed hosts (acceptable)