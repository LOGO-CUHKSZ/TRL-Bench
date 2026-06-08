# Multi-GPU Support for Embedding Extraction

## Overview

The `extract_embeddings_unified.py` script **automatically** uses all available GPUs using PyTorch's DataParallel. No special flags needed!

## Changes Made

1. **Added `get_base_model()` helper function** (line 161-173)
   - Unwraps DataParallel to access the original model
   - Ensures correct attribute access for model methods

2. **Automatic GPU detection**
   - Detects all available GPUs
   - Automatically uses DataParallel when 2+ GPUs are found
   - No manual configuration required

3. **Updated extraction functions**
   - `extract_from_pretrained_model()` uses `get_base_model()` (line 225)
   - `extract_from_finetuned_model()` uses `get_base_model()` (line 353)
   - Both correctly access nested model attributes through the wrapper

4. **Automatic DataParallel wrapping** (line 438-441)
   - Automatically wraps model when 2+ GPUs are detected
   - Zero configuration needed

## Usage

The script automatically detects and uses all available GPUs. **No special flags needed!**

### Basic usage (auto-detects GPUs):
```bash
python scripts/embedding_extraction/extract_embeddings_unified.py \
    --model_name_or_path checkpoints/model.ckpt \
    --data_dir spider_join_processed \
    --output_file embeddings.pkl
```

### With increased batch size for better GPU utilization:
```bash
python scripts/embedding_extraction/extract_embeddings_unified.py \
    --model_name_or_path checkpoints/model.ckpt \
    --data_dir spider_join_processed \
    --output_file embeddings.pkl \
    --batch_size 512
```

**Note:** When using multiple GPUs, consider increasing `--batch_size` proportionally:
- 2 GPUs → try batch_size 512 (2x default)
- 4 GPUs → try batch_size 1024 (4x default)
- Adjust based on available GPU memory

## Expected Output

### Single GPU:
```
Using device: cuda
Single GPU detected

🎯 Mode: Single table processing (pretrained model)
...
```

### Multi-GPU (automatic):
```
Using device: cuda
✅ Detected 4 GPUs - will use all GPUs with DataParallel

🔧 Wrapping model with DataParallel for 4 GPUs...
✅ Model wrapped - all GPUs will be utilized

🎯 Mode: Single table processing (pretrained model)
...
```

## Implementation Details

### DataParallel Behavior
- Automatically splits batches across available GPUs
- Each GPU processes its portion independently
- Results are gathered back to GPU 0 in original order

### Results Consistency
- Embeddings are **functionally identical** to single-GPU runs
- Possible floating-point differences at ~1e-7 or 1e-8 level (negligible)
- Deterministic when using `model.eval()` and `torch.no_grad()`

### Performance Considerations
- Recommended to increase `--batch_size` proportionally to GPU count
- Example: 2 GPUs → 2x batch size, 4 GPUs → 4x batch size
- Monitor GPU memory usage to find optimal batch size

## Troubleshooting

### "RuntimeError: module must have its parameters and buffers on device cuda:0"
- Ensure model is moved to device before wrapping with DataParallel
- Fixed in current implementation (line 438)

### Lower than expected speedup
- Increase batch size (default 256 may be too small)
- Check if data loading is the bottleneck (monitor GPU utilization)

### Out of memory errors
- Reduce `--batch_size` parameter
- Some models may not fit on all GPUs simultaneously
