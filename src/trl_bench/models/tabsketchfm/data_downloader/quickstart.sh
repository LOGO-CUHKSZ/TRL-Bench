#!/bin/bash
# Quick Start Script for TabSketchFM Data Downloader

set -e

echo "================================================"
echo "TabSketchFM Data Downloader - Quick Start"
echo "================================================"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

echo "Step 1: Installing dependencies..."
pip install -r requirements_downloader.txt

echo ""
echo "Step 2: Analyzing source distribution..."
python3 analyze_sources.py --input ../pretraining_tables.txt

echo ""
echo "Step 3: Validating a sample of URLs (1000)..."
python3 validate_urls.py --sample 1000

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Review the configuration in config.yaml"
echo "2. (Optional) Set API tokens:"
echo "   export SOCRATA_APP_TOKEN='your_token'"
echo "   export CKAN_API_KEY='your_key'"
echo ""
echo "3. Start downloading:"
echo "   # Test with 10 URLs first"
echo "   python3 download_manager.py --limit 10"
echo ""
echo "   # Download everything"
echo "   python3 download_manager.py"
echo ""
echo "4. Monitor progress:"
echo "   cat download_progress.json"
echo "   tail -f download_errors.log"
echo ""
echo "================================================"
