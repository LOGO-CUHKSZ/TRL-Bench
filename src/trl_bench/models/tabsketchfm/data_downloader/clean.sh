#!/bin/bash
# Quick cleanup script - removes all download results

echo "🧹 Cleaning download results..."

# Stop any running downloads
pkill -f "download_manager.py" 2>/dev/null && echo "  ✓ Stopped running downloads"

# Remove downloaded data
rm -rf ../opendata ../opendata_metadata 2>/dev/null && echo "  ✓ Removed data directories"

# Remove progress and logs
rm -f download_progress.json download_errors.log failed_urls.txt 2>/dev/null && echo "  ✓ Removed progress files"

# Remove analysis outputs
rm -f domain_stats.txt url_validation_results*.json url_validation_results*.txt 2>/dev/null && echo "  ✓ Removed analysis files"

# Remove any logs
rm -f *.log 2>/dev/null

echo ""
echo "✅ Cleanup complete! Ready for fresh download."
