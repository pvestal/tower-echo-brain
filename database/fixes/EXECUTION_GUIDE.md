# 🚨 DATABASE EMERGENCY CLEANUP - EXECUTION GUIDE

**⚠️ CRITICAL:** echo_brain database has 5.1GB bloat requiring immediate cleanup

## 🎯 QUICK SUMMARY
- **Problem:** 895,274 file records bloating echo_context_registry table (5.1GB)
- **Solution:** Remove duplicates, empty files, add missing indexes
- **Time:** 2-4 hours total (30-60 min downtime for VACUUM FULL)
- **Expected Result:** 5.1GB → 1.5GB (70% size reduction)

## 🔧 STEP-BY-STEP EXECUTION

### Step 1: Create Backup (5-10 minutes)
```bash
# Create backup before any changes
PGPASSWORD=tower_echo_brain_secret_key_2025 pg_dump -h 192.168.50.135 -U patrick echo_brain > /tmp/echo_brain_backup_$(date +%Y%m%d_%H%M%S).sql

# Verify backup created successfully
ls -lh /tmp/echo_brain_backup_*
```

### Step 2: Check Current Status (2 minutes)
```bash
# Connect to database and check current size
PGPASSWORD=tower_echo_brain_secret_key_2025 psql -h 192.168.50.135 -U patrick -d echo_brain -c "
SELECT
    pg_size_pretty(pg_total_relation_size('echo_context_registry')) as table_size,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE source_type = 'file') as file_records
FROM echo_context_registry;
"
```

### Step 3: Execute Critical Cleanup (30-45 minutes)
```bash
# Run the critical cleanup script
PGPASSWORD=tower_echo_brain_secret_key_2025 psql -h 192.168.50.135 -U patrick -d echo_brain -f /opt/tower-echo-brain/database/fixes/critical-bloat-cleanup.sql
```

**This will:**
- Remove 35,107 empty content files
- Remove ~84,000 duplicate records (keeping newest)
- Archive old low-importance files
- Add missing primary keys
- Optimize indexes

### Step 4: Performance Optimization (15-20 minutes)
```bash
# Run performance optimization script
PGPASSWORD=tower_echo_brain_secret_key_2025 psql -h 192.168.50.135 -U patrick -d echo_brain -f /opt/tower-echo-brain/database/fixes/performance-optimization.sql
```

### Step 5: Implement Retention Policy (10 minutes)
```bash
# Set up automated retention to prevent future bloat
PGPASSWORD=tower_echo_brain_secret_key_2025 psql -h 192.168.50.135 -U patrick -d echo_brain -f /opt/tower-echo-brain/database/fixes/data-retention-policy.sql
```

### Step 6: Verify Results (5 minutes)
```bash
# Check final table size and record count
PGPASSWORD=tower_echo_brain_secret_key_2025 psql -h 192.168.50.135 -U patrick -d echo_brain -c "
SELECT
    'echo_context_registry' as table_name,
    pg_size_pretty(pg_total_relation_size('echo_context_registry')) as new_size,
    COUNT(*) as new_record_count,
    COUNT(*) FILTER (WHERE source_type = 'file') as file_records,
    COUNT(*) FILTER (WHERE source_type = 'conversation') as conversation_records
FROM echo_context_registry;
"

# Check dead tuple count
PGPASSWORD=tower_echo_brain_secret_key_2025 psql -h 192.168.50.135 -U patrick -d echo_brain -c "
SELECT relname, n_dead_tup, n_live_tup
FROM pg_stat_user_tables
WHERE relname = 'echo_context_registry';
"
```

## 🎯 EXPECTED RESULTS

### Before Cleanup:
- Table Size: 5.1GB
- Records: 901,965
- File Records: 895,274
- Duplicates: 119,791
- Empty Files: 35,107

### After Cleanup:
- Table Size: ~1.5GB (70% reduction)
- Records: ~400,000 (unique, valuable content)
- File Records: ~300,000 (deduplicated)
- Duplicates: 0
- Empty Files: 0

## ⚠️ SAFETY CHECKS

### Before Starting:
- [ ] Backup created and verified
- [ ] Echo Brain service can be restarted if needed
- [ ] 2-4 hour maintenance window available
- [ ] Disk space available (need ~6GB free for operations)

### If Something Goes Wrong:
1. **Stop immediately** - Don't continue with broken operations
2. **Restore from backup:**
   ```bash
   PGPASSWORD=tower_echo_brain_secret_key_2025 dropdb -h 192.168.50.135 -U patrick echo_brain
   PGPASSWORD=tower_echo_brain_secret_key_2025 createdb -h 192.168.50.135 -U patrick echo_brain
   PGPASSWORD=tower_echo_brain_secret_key_2025 psql -h 192.168.50.135 -U patrick echo_brain < /tmp/echo_brain_backup_TIMESTAMP.sql
   ```
3. **Contact Claude for alternative approach**

## 📊 MONITORING COMMANDS

### Check Progress During Cleanup:
```bash
# Monitor active queries
PGPASSWORD=tower_echo_brain_secret_key_2025 psql -h 192.168.50.135 -U patrick -d echo_brain -c "
SELECT query, state, query_start
FROM pg_stat_activity
WHERE datname = 'echo_brain' AND state = 'active';
"

# Check current table size
PGPASSWORD=tower_echo_brain_secret_key_2025 psql -h 192.168.50.135 -U patrick -d echo_brain -c "
SELECT pg_size_pretty(pg_total_relation_size('echo_context_registry')) as current_size;
"
```

## 🚀 POST-CLEANUP TASKS

### Immediate (same day):
1. **Restart Echo Brain service** - Clear any cached query plans
2. **Test character queries** - Verify performance improvement
3. **Check error logs** - Ensure no application errors

### This Week:
1. **Monitor daily cleanup** - Verify retention policy working
2. **Review query performance** - Check for <100ms character lookups
3. **Update documentation** - Record new baseline metrics

### Ongoing:
1. **Weekly bloat check** - Prevent future issues
2. **Monthly performance review** - Optimize as needed

## 📞 EMERGENCY CONTACTS

**If cleanup fails or database becomes unresponsive:**
- Stop all operations immediately
- Restore from backup
- Contact Claude Code for emergency consultation

---

**🎯 Ready to Execute?** Follow steps 1-6 in order. Total time: 2-4 hours. Expected result: 70% database size reduction and significantly improved performance.

**Last Updated:** November 19, 2025