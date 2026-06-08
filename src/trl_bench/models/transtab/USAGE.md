# TransTab Row Embeddings

Self-supervised row embeddings using TransTab's Vertical-Partition Contrastive Learning (VPCL).

Reference: Wang & Sun, 2022 — "TransTab: Learning Transferable Tabular Transformers Across Tables"

## Directory Mode (DLTE pipeline)

```bash
python models/transtab/generate_embeddings_directory.py \
    --input_dir /path/to/csvs \
    --output_path /path/to/output.pkl \
    --checkpoint_base_dir /path/to/checkpoints \
    --num_epoch 50 --hidden_dim 128 --num_layer 2
```

## Data Mode (canonical dataset)

### Train
```bash
python models/transtab/train_transtab.py \
    --data_dir data/my_dataset \
    --checkpoint_dir models/transtab/checkpoints \
    --num_epoch 50
```

### Generate Embeddings
```bash
python models/transtab/generate_embeddings.py \
    --data_dir data/my_dataset \
    --checkpoint_dir models/transtab/checkpoints \
    --embedding_dir models/transtab/embeddings
```

## Key Differences from ts3l Models

- Uses TransTab's own API (`transtab.build_contrastive_learner`, `transtab.train`, `transtab.build_encoder`)
- Handles raw DataFrames directly — no `SSLPreprocessor`, but applies `MinMaxScaler` to numerical columns (matching TransTab's own `load_data()`)
- Preprocessing: median NaN fill + MinMaxScaler for numerical (scaled to [0,1] on training data; val/test may exceed this range), `"__MISSING__"` sentinel for categorical, int cast for binary
- Checkpoint format: `ckpt_best.pt` + `training_config.pkl` (not `{model}_self_supervised.ckpt`)
- Distinguishes categorical, numerical, and binary columns
