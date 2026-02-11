#!/bin/bash
# TOWER REORGANIZATION SCRIPT - THE GREAT UNFUCKING
# Purpose: Clean up the organizational disaster across Tower projects
# Created: 2026-02-11

set -e  # Exit on error

echo "================================================"
echo "🔥 TOWER REORGANIZATION - THE GREAT UNFUCKING 🔥"
echo "================================================"
echo ""
echo "This will:"
echo "1. Move anime scripts from echo-brain to anime-production"
echo "2. Remove 29GB duplicate in /mnt/nvme"
echo "3. Consolidate LoRA Studios"
echo "4. Fix all broken references"
echo ""
read -p "⚠️  READY TO UNFUCK THIS MESS? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. The mess remains."
    exit 1
fi

# Backup critical configs first
echo ""
echo "📦 Creating safety backups..."
mkdir -p /tmp/tower_reorg_backup
cp -r /opt/tower-echo-brain/scripts/*ssot* /tmp/tower_reorg_backup/ 2>/dev/null || true
cp -r /opt/tower-echo-brain/scripts/*anime* /tmp/tower_reorg_backup/ 2>/dev/null || true
echo "✅ Backups saved to /tmp/tower_reorg_backup/"

# STEP 1: Create proper directory structure
echo ""
echo "🏗️  STEP 1: Creating proper structure..."
mkdir -p /opt/tower-anime-production/ssot
mkdir -p /opt/tower-anime-production/ssot/archive

# STEP 2: Move anime-specific scripts from echo-brain to anime-production
echo ""
echo "📦 STEP 2: Moving anime scripts to proper home..."

ANIME_SCRIPTS=(
    "ssot_generation_orchestrator.py"
    "anime_pipeline.py"
    "rebuild_story_bible_ssot.py"
    "ingest_complete_anime_production.py"
    "project_config_ssot.py"
)

for script in "${ANIME_SCRIPTS[@]}"; do
    if [ -f "/opt/tower-echo-brain/scripts/$script" ]; then
        echo "  Moving: $script"
        mv "/opt/tower-echo-brain/scripts/$script" "/opt/tower-anime-production/ssot/" || {
            echo "  ⚠️  Failed to move $script, copying instead"
            cp "/opt/tower-echo-brain/scripts/$script" "/opt/tower-anime-production/ssot/"
        }
    else
        echo "  ⏭️  Skipping: $script (not found)"
    fi
done

# STEP 3: Check which anime-production is newer and keep the best one
echo ""
echo "🔍 STEP 3: Analyzing duplicates..."

OPT_DATE=$(find /opt/tower-anime-production -type f -name "*.py" -exec stat -c %Y {} \; | sort -n | tail -1)
NVME_DATE=$(find /mnt/nvme/opt/tower-anime-production -type f -name "*.py" -exec stat -c %Y {} \; 2>/dev/null | sort -n | tail -1)

if [ -z "$NVME_DATE" ]; then
    echo "  No nvme duplicate found, skipping..."
else
    echo "  /opt version last modified: $(date -d @$OPT_DATE)"
    echo "  /mnt/nvme version last modified: $(date -d @$NVME_DATE)"

    if [ "$OPT_DATE" -ge "$NVME_DATE" ]; then
        echo "  ✅ /opt version is newer or same - keeping it"
        echo "  🗑️  Removing duplicate at /mnt/nvme/opt/tower-anime-production..."
        read -p "  Confirm deletion of 29GB duplicate? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf /mnt/nvme/opt/tower-anime-production
            echo "  ✅ Duplicate removed, 29GB freed!"
        else
            echo "  ⏭️  Skipped deletion"
        fi
    else
        echo "  ⚠️  NVME version is newer! Manual review needed."
        echo "  Consider: rsync -av /mnt/nvme/opt/tower-anime-production/ /opt/tower-anime-production/"
    fi
fi

# STEP 4: Consolidate LoRA Studios
echo ""
echo "🔧 STEP 4: Consolidating LoRA Studios..."

if [ -d "/opt/tower-lora-studio" ] && [ -d "/opt/tower-anime-production/training/lora-studio" ]; then
    echo "  Found both LoRA Studios"
    echo "  Standalone: $(du -sh /opt/tower-lora-studio | cut -f1)"
    echo "  Integrated: $(du -sh /opt/tower-anime-production/training/lora-studio | cut -f1)"

    # Archive the standalone if it's small
    if [ $(du -s /opt/tower-lora-studio | cut -f1) -lt 100000 ]; then  # Less than 100MB
        echo "  📦 Archiving standalone LoRA Studio..."
        mv /opt/tower-lora-studio /opt/tower-lora-studio.archived.$(date +%Y%m%d)
        echo "  ✅ Standalone archived"
    else
        echo "  ⚠️  Standalone is large, manual review needed"
    fi
fi

# STEP 5: Fix import paths in moved scripts
echo ""
echo "🔧 STEP 5: Fixing import paths..."

cd /opt/tower-anime-production/ssot/

# Update imports in the moved scripts
for script in *.py; do
    if [ -f "$script" ]; then
        echo "  Fixing imports in: $script"

        # Fix echo brain imports that should now be local
        sed -i 's|from scripts\.|from \.|g' "$script" 2>/dev/null || true
        sed -i 's|/opt/tower-echo-brain/scripts|/opt/tower-anime-production/ssot|g' "$script" 2>/dev/null || true

        # Fix workflow paths if needed
        sed -i 's|WORKFLOW_DIR = "/opt/ComfyUI/workflows"|WORKFLOW_DIR = "/opt/tower-anime-production/workflows/comfyui"|g' "$script" 2>/dev/null || true
    fi
done

# STEP 6: Create symlinks for compatibility
echo ""
echo "🔗 STEP 6: Creating compatibility symlinks..."

# Symlink workflows to ComfyUI if needed
if [ ! -e "/opt/ComfyUI/workflows" ]; then
    ln -s /opt/tower-anime-production/workflows/comfyui /opt/ComfyUI/workflows
    echo "  ✅ Linked workflows to ComfyUI"
fi

# STEP 7: Update systemd services if they exist
echo ""
echo "🔧 STEP 7: Checking systemd services..."

SERVICES=("tower-anime-production" "tower-echo-brain" "tower-lora-studio")
for service in "${SERVICES[@]}"; do
    if systemctl list-units --all | grep -q "$service"; then
        echo "  Found service: $service"
        SERVICE_FILE="/etc/systemd/system/${service}.service"
        if [ -f "$SERVICE_FILE" ]; then
            # Check if service references old paths
            if grep -q "/opt/tower-echo-brain/scripts.*anime\|ssot" "$SERVICE_FILE"; then
                echo "  ⚠️  Service $service needs path updates"
                echo "     Run: sudo systemctl edit $service"
            fi
        fi
    fi
done

# STEP 8: Verify critical files exist
echo ""
echo "✅ STEP 8: Verification..."

CRITICAL_FILES=(
    "/opt/tower-anime-production/ssot/ssot_generation_orchestrator.py"
    "/opt/tower-anime-production/workflows/comfyui/anime_30sec_rife_workflow_with_lora.json"
    "/opt/tower-anime-production/quality/video_contract.py"
)

ALL_GOOD=true
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ Found: $(basename $file)"
    else
        echo "  ❌ MISSING: $file"
        ALL_GOOD=false
    fi
done

# Summary
echo ""
echo "================================================"
echo "📊 REORGANIZATION SUMMARY"
echo "================================================"

if [ "$ALL_GOOD" = true ]; then
    echo "✅ SUCCESS! The Tower organization has been unfucked!"
else
    echo "⚠️  PARTIAL SUCCESS - Some files need manual attention"
fi

echo ""
echo "📁 New Structure:"
echo "  /opt/tower-anime-production/ssot/     ← Anime SSOT scripts"
echo "  /opt/tower-anime-production/workflows/ ← ComfyUI workflows"
echo "  /opt/tower-anime-production/training/  ← LoRA Studio (integrated)"
echo "  /opt/tower-echo-brain/scripts/        ← Memory/conversation only"
echo ""
echo "🔄 Next Steps:"
echo "  1. Restart any affected services"
echo "  2. Test the SSOT orchestrator from new location"
echo "  3. Verify LoRA training still works"
echo ""
echo "💾 Backups saved in: /tmp/tower_reorg_backup/"
echo "================================================"