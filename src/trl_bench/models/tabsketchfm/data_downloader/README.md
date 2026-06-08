# Full-Featured Data Downloader for TabSketchFM

A comprehensive downloader for acquiring the pretraining data for TabSketchFM from the 127,000+ URLs listed in `pretraining_tables.txt`.

## Features

- **Multi-Source Support**: Handles direct CSV/Excel downloads, Socrata API, CKAN API, and generic web scraping
- **Parallel Downloads**: Concurrent downloads with configurable worker pool
- **Rate Limiting**: Respects rate limits for different source types
- **Smart Classification**: Automatically detects URL type and routes to appropriate handler
- **Resume Capability**: Checkpoint-based progress tracking allows resuming interrupted downloads
- **Metadata Extraction**: Automatically generates `.meta` files with dataset information
- **Error Handling**: Robust retry logic and comprehensive error logging
- **Progress Tracking**: Real-time progress bars and detailed statistics

## Architecture

```
data_downloader/
├── download_manager.py       # Main orchestrator
├── url_classifier.py          # URL type detection
├── progress_tracker.py        # Progress and resume capability
├── config.yaml                # Configuration settings
├── downloaders/
│   ├── base.py               # Base downloader class
│   ├── direct.py             # Direct CSV/Excel downloads
│   ├── socrata.py            # Socrata API integration
│   ├── ckan.py               # CKAN API integration
│   └── generic.py            # Web scraping for landing pages
└── utils/
    ├── validate_urls.py      # URL validation utility
    └── analyze_sources.py    # Source analysis utility
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements_downloader.txt
```

2. (Optional) Set up API tokens for better rate limits:
```bash
export SOCRATA_APP_TOKEN="your_token_here"
export CKAN_API_KEY="your_key_here"
```

## Configuration

Edit `config.yaml` to customize settings:

```yaml
# Key settings
output_dir: "~/opendata"              # Where to save CSV files
metadata_dir: "~/opendata_metadata"   # Where to save .meta files
max_workers: 10                       # Parallel download threads
timeout: 60                           # Request timeout (seconds)
max_file_size_mb: 500                 # Skip files larger than this
```

See `config.yaml` for all available options.

## Usage

### Basic Usage

Download all datasets:
```bash
python download_manager.py
```

### Advanced Usage

**Download with custom config:**
```bash
python download_manager.py --config my_config.yaml
```

**Download a subset (for testing):**
```bash
# Download first 100 URLs
python download_manager.py --limit 100

# Download URLs 1000-2000
python download_manager.py --start 1000 --limit 1000
```

**Resume a previous download:**
```bash
# Progress is automatically saved and resumed
python download_manager.py
```

**Retry failed downloads:**
```bash
python download_manager.py --reset-failed
```

## Utility Scripts

### 1. Analyze Sources

Analyze the distribution of data sources:
```bash
python analyze_sources.py --input ../pretraining_tables.txt
```

Output:
- URL type distribution
- Domain statistics
- File format breakdown
- Download strategy requirements

### 2. Validate URLs

Check which URLs are still accessible:
```bash
# Validate a sample of 1000 URLs
python validate_urls.py --sample 1000

# Validate all URLs (takes a long time!)
python validate_urls.py --workers 50
```

Output:
- `url_validation_results.json` - Full validation results
- `url_validation_results_accessible.txt` - List of accessible URLs

## Download Strategies

The downloader automatically classifies URLs and uses the appropriate strategy:

| Strategy | Description | Count (~) |
|----------|-------------|-----------|
| **direct** | Direct CSV/Excel file downloads | ~55,000 |
| **socrata** | Socrata open data portals (API) | ~2,200 |
| **ckan** | CKAN open data portals (API) | ~3,000 |
| **scrape** | Landing pages requiring web scraping | ~65,000 |
| **ftp** | FTP downloads (not yet implemented) | ~2 |
| **skip** | Invalid or broken URLs | ~1,800 |

## Output Structure

Downloaded data is organized by domain:

```
opendata/
├── data_gov_au/
│   ├── dataset1.csv
│   └── dataset2.csv
├── socrata_com/
│   ├── dataset3.csv
│   └── dataset4.csv
└── ...

opendata_metadata/
├── data_gov_au/
│   ├── dataset1.csv.meta
│   └── dataset2.csv.meta
├── socrata_com/
│   ├── dataset3.csv.meta
│   └── dataset4.csv.meta
└── ...
```

## Metadata Format

Each downloaded file has a corresponding `.meta` JSON file:

```json
{
  "source_url": "https://...",
  "download_date": "2025-11-19T10:30:00",
  "file_path": "/path/to/data.csv",
  "file_size": 12345,
  "domain": "example.com",
  "download_method": "direct",
  "api_metadata": {
    "name": "Dataset Name",
    "description": "Dataset description",
    "columns": ["col1", "col2", ...]
  }
}
```

## Progress Tracking

Progress is saved to `download_progress.json`:

```json
{
  "total_urls": 127934,
  "processed": 5000,
  "successful": 4500,
  "failed": 300,
  "skipped": 200,
  "completed_urls": { ... },
  "failed_urls": { ... }
}
```

Failed URLs are also saved to `failed_urls.txt` for easy retry.

## Performance Tips

1. **Adjust worker count** based on your bandwidth and CPU:
   - Fast connection: `max_workers: 20-50`
   - Slow connection: `max_workers: 5-10`

2. **Enable checkpointing** for large downloads:
   ```yaml
   checkpoint_interval: 100  # Save every 100 downloads
   ```

3. **Filter by source type** by modifying the URL list:
   ```bash
   # Extract only direct CSV URLs
   grep "\.csv" pretraining_tables.txt > direct_csvs.txt
   ```

4. **Test on a sample first**:
   ```bash
   python download_manager.py --limit 10
   ```

## Troubleshooting

### SSL Certificate Errors

Many old URLs have SSL issues. SSL verification is disabled by default:
```yaml
validate_ssl: false
```

### Timeout Errors

Increase timeout for slow servers:
```yaml
timeout: 120  # 2 minutes
```

### Rate Limiting

Reduce worker count or adjust rate limits:
```yaml
max_workers: 5
rate_limits:
  default: 1  # 1 request per second
```

### Memory Issues

Process in smaller batches:
```bash
# Process 10,000 URLs at a time
for i in {0..120000..10000}; do
    python download_manager.py --start $i --limit 10000
done
```

## Expected Results

Based on the URL analysis:

- **Total URLs**: 127,934
- **Expected Success Rate**: 40-60% (many URLs are 10+ years old)
- **Estimated Data Size**: 100GB - 1TB (varies greatly)
- **Download Time**:
  - With 10 workers: 2-5 days
  - With 50 workers: 12-24 hours
  - Depends heavily on network speed and source responsiveness

## Notes

- The downloader handles HTTP(S) sources; a couple of FTP-only and authentication-gated sources are skipped.
- Some source URLs from 2011-2018 may no longer resolve.

## Next Steps

After downloading:

1. **Preprocess the data** using the main TabSketchFM scripts:
   ```bash
   python tabsketchfm/batch_fastdata_opendata.py \
       --input_dir ~/opendata \
       --metadata_dir ~/opendata_metadata \
       --output_dir ~/opendata_processed
   ```

2. **Create train/test/val splits** as described in the main README

3. **Pretrain the model** using the processed data

## Contributing

To add support for new source types:

1. Create a new downloader in `downloaders/`
2. Inherit from `BaseDownloader`
3. Implement the `download()` method
4. Add URL detection logic to `url_classifier.py`
5. Register in `download_manager.py`

## License

Same as TabSketchFM main project (CC BY-NC-ND 4.0)

## Support

For issues or questions:
1. Check the error log: `download_errors.log`
2. Review failed URLs: `failed_urls.txt`
3. Run validation: `python validate_urls.py --sample 100`
4. Open an issue on the TabSketchFM repository
