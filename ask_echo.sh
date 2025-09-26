#!/bin/bash
# Ask Echo to compare cloud vs local photos

echo "To compare photos, run:"
echo "  python3 /opt/tower-echo-brain/compare_photos.py"
echo ""
echo "To index all 14,301 photos:"
echo "  python3 /opt/tower-echo-brain/photo_indexer_fixed.py"
echo "  (remove [:100] limit first)"
echo ""
echo "To check what's in Google Photos cloud:"
echo "  Need to fix OAuth - Google blocked unverified domain"
echo "  Options:"
echo "    1. Use localhost redirect (needs desktop)"
echo "    2. Create your own Google Cloud project"
echo "    3. Just use the 14,301 files you already have"
