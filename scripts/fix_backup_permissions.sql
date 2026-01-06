-- Fix backup permissions for Echo Brain database
-- Run as postgres superuser

-- Grant schema usage
GRANT USAGE ON SCHEMA public TO patrick;

-- Grant select on all existing tables
GRANT SELECT ON ALL TABLES IN SCHEMA public TO patrick;

-- Grant select on all future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO patrick;

-- Grant select on all sequences
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO patrick;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON SEQUENCES TO patrick;

-- Ensure patrick can read from anime tables
GRANT SELECT ON ALL TABLES IN SCHEMA public TO patrick;

-- For RV visualization schema (if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'rv_visualization') THEN
        GRANT USAGE ON SCHEMA rv_visualization TO patrick;
        GRANT SELECT ON ALL TABLES IN SCHEMA rv_visualization TO patrick;
    END IF;
END $$;

-- Create a backup role with full read access
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'echo_backup_role') THEN
        CREATE ROLE echo_backup_role;
    END IF;
END $$;

GRANT USAGE ON SCHEMA public TO echo_backup_role;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO echo_backup_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO echo_backup_role;
GRANT echo_backup_role TO patrick;

-- Show current table owners
\echo 'Current table ownership:'
SELECT schemaname, tablename, tableowner
FROM pg_tables
WHERE schemaname IN ('public', 'rv_visualization')
ORDER BY schemaname, tableowner, tablename;