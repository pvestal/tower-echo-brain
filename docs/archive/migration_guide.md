# Migration Guide for tower-echo-brain

## Overview
This service has been migrated to use the Tower Common Library for standardization.

## Files Created
- `main_migrated.py` - New standardized main entry point
- `requirements_migrated.txt` - Updated dependencies with tower-common
- `migration_guide.md` - This guide

## Migration Steps Completed
- Create backup of existing service
- Install tower-common library dependencies
- Generate new main.py using standardized patterns
- Create migration guide for existing routes
- Update requirements.txt
- Update systemd service file
- Test migrated service
- Update nginx configuration if needed

## Manual Tasks Required
- Review and migrate existing API routes
- Update database models if needed
- Migrate authentication logic
- Test all endpoints
- Update documentation

## Service Analysis
- Main files found: []
- API routes found: 65 files
- Database integration: Yes
- Redis integration: Yes
- Authentication: Yes
- Detected port: None

## Next Steps
1. Review the generated `main_migrated.py`
2. Migrate your existing routes to the new structure
3. Test the service with `python main_migrated.py`
4. Update systemd service to use new main file
5. Update nginx configuration if API paths changed

## API Standardization
All services now follow these patterns:
- Health check: `/api/echo-brain/health`
- Documentation: `/api/echo-brain/docs`
- Version info: `/api/echo-brain/version`
- Status: `/api/echo-brain/status`

## Rollback
If issues occur, restore from backup at: `/opt/tower-echo-brain/backup_pre_migration`
