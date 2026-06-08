#!/bin/bash
# Retry timeout failures with more patient settings

echo "🔄 Preparing to retry timeout failures..."

# Backup current config
cp config.yaml config.yaml.backup

# Create optimized config for slow/timeout URLs
cat > config_retry.yaml << 'EOF'
# Retry Configuration - Optimized for slow/timeout URLs

# Input/Output Paths
urls_file: "../pretraining_tables.txt"
output_dir: "../opendata"
metadata_dir: "../opendata_metadata"
progress_file: "./download_progress.json"
error_log: "./download_errors_retry.log"

# Download Settings - More patient
max_workers: 5  # Reduced from 20 for stability
timeout: 120  # Doubled to 120 seconds
max_retries: 5  # Increased from 3
retry_delay: 10  # Increased delay between retries
chunk_size: 8192

# Rate Limiting - Slower
rate_limits:
  default: 1  # Slower rate
  socrata: 0.5
  ckan: 0.5
  direct: 2

# API Keys
api_keys:
  socrata_app_token: null
  ckan_api_key: null

# File Handling
max_file_size_mb: 5000
allowed_extensions:
  - .csv
  - .xls
  - .xlsx
  - .tsv
  - .txt

# Resume Settings
resume_enabled: true
checkpoint_interval: 50  # Save more frequently

# Metadata Settings
extract_metadata: true
metadata_fields:
  - title
  - description
  - source_url
  - download_date
  - file_size
  - num_rows
  - num_columns
  - column_names

# URL Classification
skip_invalid_urls: true
validate_ssl: false

# Logging
log_level: "INFO"
verbose: true
EOF

echo "✅ Created retry configuration (slower, more patient)"
echo ""
echo "Estimated timeout failures: ~10,000 URLs"
echo "Expected recovery: 2,000-3,000 files"
echo "Estimated time: 6-12 hours"
echo ""
echo "To start retry:"
echo "  python3 download_manager.py --config config_retry.yaml --reset-failed"
echo ""
echo "To restore original config:"
echo "  cp config.yaml.backup config.yaml"
