#!/bin/bash
# Quick test of the downloader with valid URLs

echo "Creating test URL list..."
cat > test_urls.txt <<'EOF'
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
https://raw.githubusercontent.com/fivethirtyeight/data/master/alcohol-consumption/drinks.csv
https://people.sc.fsu.edu/~jburkardt/data/csv/airtravel.csv
EOF

echo ""
echo "Testing download manager with 3 known-good URLs..."
python3 download_manager.py --config config.yaml --limit 3 2>&1 | tail -n 30

echo ""
echo "Checking downloaded files..."
find ~/opendata -name "*.csv" -type f | head -n 5

echo ""
echo "Test complete!"
