"""
Unified embedding extraction script that works with both pretrained and finetuned models.

This script extracts three types of embeddings from TabSketchFM models:
1. CLS embedding: The [CLS] token representation
2. Table embedding: Mean-pooled representation of all table tokens
3. Column embeddings: Per-column mean-pooled representations

Usage:
    # With pretrained TabSketchFM (single table processing)
    python extract_embeddings_unified.py \
        --model_name_or_path logs/tabsketchfm-pretrain/checkpoints/epoch=10-step=27786.ckpt \
        --data_dir spider_join_processed_dataset \
        --output_file data_lake_embeddings.pkl

    # With finetuned model (cross-encoder, requires pairing)
    python extract_embeddings_unified.py \
        --model_name_or_path finetuned_model.pt \
        --model_type finetuned \
        --data_dir spider_join_processed_dataset \
        --output_file data_lake_embeddings.pkl

    # With raw BERT (single table processing)
    python extract_embeddings_unified.py \
        --model_name_or_path bert-base-uncased \
        --data_dir spider_join_processed_dataset \
        --output_file data_lake_embeddings.pkl
"""

import os
import sys
import pickle
import bz2
import json
import torch
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BertModel

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from tabsketchfm.models.tabsketchfm import TabSketchFM
from tabsketchfm.data_processing.tabular_tokenizer import TableSimilarityTokenizer, fake_tablename_metadata
from tabsketchfm.utils.datamodule import PretrainDataModule, FinetuneDataModule


def extract_table_id(table_name: str) -> str:
    """
    Extract table_id from a table name/path.

    Removes directory path and file extension to get canonical identifier.

    Args:
        table_name: Table filename or path (e.g., 'path/to/table.csv', 'table.json')

    Returns:
        Clean table_id (e.g., 'table')
    """
    # Get basename (remove directory path)
    basename = os.path.basename(table_name)
    # Remove common extensions
    for ext in ['.csv', '.json', '.tsv', '.parquet', '.bz2']:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
    return basename


def find_table_col(toks, seq_states, table_start, table_end, inputs):
    """
    Extract three types of embeddings from hidden states.

    Args:
        toks: Tokenizer
        seq_states: Hidden states from model [seq_len, 768]
        table_start: Start position of table in sequence
        table_end: End position of table in sequence
        inputs: Input token IDs

    Returns:
        table_embedding: Mean-pooled table representation [768]
        col_embeddings: Per-column embeddings {col_id: [768], ...}
        cls_embedding: CLS token representation [768]
    """
    cls_embedding = seq_states[0].cpu().tolist()

    special_tokens = {toks.cls_token, toks.sep_token, toks.pad_token}
    tokens = toks.convert_ids_to_tokens(inputs)

    mask = []
    num_sep = 0
    col_states = {}

    for i in range(table_start, table_end):
        if tokens[i] in special_tokens:
            mask.append(False)
            if tokens[i] == toks.sep_token and i != table_start:
                num_sep += 1
            continue
        else:
            mask.append(True)
            if num_sep not in col_states:
                col_states[num_sep] = []
            col_states[num_sep].append(seq_states[i])

    seq_states_table = seq_states[table_start:table_end]

    # Move mask to same device as seq_states
    mask = torch.tensor(mask, device=seq_states.device).unsqueeze(-1)

    sz = seq_states_table.size()[1]
    seq_states_table = seq_states_table.masked_select(mask)
    seq_states_table = torch.reshape(seq_states_table, (-1, sz))

    # Mean pool all table tokens
    table_embedding = torch.mean(seq_states_table, dim=0).cpu().tolist()

    # Mean pool per column
    col_embeddings = {}
    for i in col_states:
        t = torch.stack(col_states[i], dim=0)
        col_embeddings[i] = torch.mean(t, dim=0).cpu().tolist()

    return table_embedding, col_embeddings, cls_embedding


def load_model(model_name_or_path, model_type, device):
    """
    Load model from either a checkpoint or model name.

    Args:
        model_name_or_path: Path to checkpoint or HuggingFace model name
        model_type: 'pretrained' or 'finetuned'
        device: torch device

    Returns:
        Loaded model
    """
    is_checkpoint = model_name_or_path.endswith('.ckpt') and os.path.isfile(model_name_or_path)
    is_pt_file = model_name_or_path.endswith('.pt') and os.path.isfile(model_name_or_path)

    if model_type == 'finetuned':
        # Load finetuned model (full model object)
        print(f"\n🔧 Loading finetuned model from: {model_name_or_path}")
        if is_pt_file:
            model = torch.load(model_name_or_path, map_location=device)
        elif is_checkpoint:
            from tabsketchfm.models.tabsketchfm_finetune import FinetuneTabSketchFM
            from transformers import AutoConfig

            # Load checkpoint to get hyperparameters
            ckpt = torch.load(model_name_or_path, map_location='cpu')
            hparams = ckpt.get('hyper_parameters', {})
            base_model = hparams.get('model_name_or_path', 'bert-base-uncased')
            num_labels = hparams.get('num_labels', 2)

            # Create config for model
            config = AutoConfig.from_pretrained(base_model)
            config.task_specific_params = {'hash_input_size': config.hidden_size}  # Enable TabularBertModel
            config.num_labels = num_labels

            print(f"   Base model: {base_model}, num_labels: {num_labels}")
            model = FinetuneTabSketchFM.load_from_checkpoint(
                model_name_or_path,
                map_location=device,
                config=config
            )
        else:
            raise ValueError(f"Finetuned model must be a .pt or .ckpt file, got: {model_name_or_path}")
        print("✅ Finetuned model loaded (cross-encoder)")
        return model, 'finetuned'

    else:  # pretrained
        if is_checkpoint:
            # Load pretrained TabSketchFM checkpoint
            print(f"\n🔧 Loading pretrained TabSketchFM from: {model_name_or_path}")
            model = TabSketchFM.load_from_checkpoint(model_name_or_path, map_location=device)
            print("✅ Pretrained TabSketchFM loaded")
        else:
            # Create fresh model with BERT weights
            print(f"\n🔧 Creating TabSketchFM with: {model_name_or_path}")
            model = TabSketchFM(
                model_name_or_path=model_name_or_path,
                learning_rate=2e-5,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1e-8
            )
            # Load pretrained BERT weights
            print("   Loading pretrained BERT weights...")
            pretrained_bert = BertModel.from_pretrained(model_name_or_path)
            model.model.bert.encoder.load_state_dict(pretrained_bert.encoder.state_dict())
            print(f"✅ Fresh TabSketchFM with {model_name_or_path} weights loaded")

        return model, 'pretrained'


def get_base_model(model):
    """
    Get the base model (no-op for single GPU version).

    Args:
        model: Model instance

    Returns:
        Base model
    """
    return model


def extract_from_pretrained_model(model, data_dir, batch_size, device):
    """
    Extract embeddings from pretrained model (processes single tables).

    Args:
        model: TabSketchFM pretrained model (may be wrapped with DataParallel)
        data_dir: Directory with preprocessed tables
        batch_size: Batch size for processing
        device: torch device

    Returns:
        Dictionary mapping table filenames to their embeddings
    """
    # Get all preprocessed files
    all_tables = []
    table_files = []

    for f in os.listdir(data_dir):
        if not f.endswith('.json.bz2'):
            continue
        filepath = os.path.join(data_dir, f)
        with bz2.open(filepath, 'rt') as inp:
            data = json.load(inp)
            table_name = data['table_metadata']['file_name']
            all_tables.append(table_name)
            table_files.append({
                'json': f,
                'table': data,
                'table_name': table_name
            })

    print(f"\n📂 Found {len(all_tables)} tables")
    print(f"   Unique tables: {len(set(all_tables))}")

    # Setup tokenizer
    config = AutoConfig.from_pretrained('bert-base-uncased')
    config.max_position_embeddings = 512
    config.task_specific_params = {'hash_input_size': config.hidden_size}
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = TableSimilarityTokenizer(
        tokenizer=bert_tokenizer,
        config=config,
        table_metadata_func=fake_tablename_metadata
    )

    embeddings = []
    model.eval()

    # Get base model (unwrap DataParallel if necessary)
    base_model = get_base_model(model)

    print("\n🔄 Extracting embeddings from single tables...")
    with torch.no_grad():
        for i in tqdm(range(0, len(table_files), batch_size)):
            batch = table_files[i:i+batch_size]

            # Tokenize each table individually
            tokenized_batch = []
            for item in batch:
                tokenized = tokenizer.tokenize_function(item['table'])
                tokenized_batch.append(tokenized)

            # Stack batch
            batch_data = {}
            for key in tokenized_batch[0].keys():
                batch_data[key] = torch.stack([t[key] for t in tokenized_batch]).to(device)

            # Forward pass through pretrained model
            # base_model.model is TabularBertForMaskedLM
            # base_model.model.bert is TabularBertModel
            outputs = base_model.model.bert(**batch_data, return_dict=True, output_hidden_states=True)

            # Extract embeddings for each table in batch
            for batch_idx, item in enumerate(batch):
                hidden_state = outputs.last_hidden_state[batch_idx]  # [512, 768]
                input_ids = batch_data['input_ids'][batch_idx]

                # The entire sequence is one table (no pairing, so table_start=0)
                # Find where the actual table content ends (before padding)
                attention_mask = batch_data['attention_mask'][batch_idx]
                seq_length = attention_mask.sum().item()

                # Extract three types of embeddings
                table_emb, col_embs, cls_emb = find_table_col(
                    bert_tokenizer,
                    hidden_state,
                    table_start=0,
                    table_end=seq_length,
                    inputs=input_ids
                )

                embeddings.append({
                    'table_id': extract_table_id(item['table_name']),
                    'table': item['table_name'],
                    'table_embedding': table_emb,
                    'column_embeddings': col_embs,  # Unified format (plural)
                    'cls_embedding': cls_emb,
                })

    return embeddings


def extract_from_finetuned_model(model, data_dir, batch_size, device):
    """
    Extract embeddings from finetuned model (requires self-pairing).

    Args:
        model: FinetuneTabSketchFM finetuned model
        data_dir: Directory with preprocessed tables
        batch_size: Batch size for processing
        device: torch device

    Returns:
        Dictionary mapping table filenames to their embeddings
    """
    # Get all preprocessed files
    all_tables = []

    for f in os.listdir(data_dir):
        if not f.endswith('.json.bz2'):
            continue
        with bz2.open(os.path.join(data_dir, f), 'rt') as inp:
            data = json.load(inp)
            all_tables.append(data['table_metadata']['file_name'])

    print(f"\n📂 Found {len(all_tables)} tables")
    all_tables = list(set(all_tables))
    print(f"   Unique tables: {len(all_tables)}")

    # Create self-pairs (table with itself)
    l_itself = []
    for obj in all_tables:
        if ':' in obj:
            table = obj.split(':')[0] + '.csv'
            col = obj.split(':')[1]
            o = {
                'table1': {'filename': table, 'col1': col},
                'table2': {'filename': table, 'col1': col},
                'label': 1
            }
        else:
            o = {
                'table1': {'filename': obj},
                'table2': {'filename': obj},
                'label': 1
            }
        l_itself.append(o)

    # Setup tokenizer
    config = AutoConfig.from_pretrained('bert-base-uncased')
    config.max_position_embeddings = 512
    config.task_specific_params = {'hash_input_size': config.hidden_size}
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = TableSimilarityTokenizer(
        tokenizer=bert_tokenizer,
        config=config,
        table_metadata_func=fake_tablename_metadata
    )

    # Create data module with self-pairs
    data_module = FinetuneDataModule(
        tokenizer=tokenizer,
        data_dir=data_dir,
        dataset={'test': l_itself},
        shuffle=False,
        concat=True,
        extract_embedding=True,
        train_batch_size=batch_size,
        val_batch_size=batch_size
    )

    data_module.setup(None)
    iterator = data_module.test_dataloader()

    embeddings = []
    model.eval()

    # Get base model (unwrap DataParallel if necessary)
    base_model = get_base_model(model)

    print("\n🔄 Extracting embeddings from table pairs (self-paired)...")
    with torch.no_grad():
        for idx, features in tqdm(enumerate(iterator), total=len(iterator)):
            input_features = features[0]
            num_t1_cols = features[3].cpu().tolist()
            table1_start = features[5].cpu().tolist()
            table1_end = features[6].cpu().tolist()
            table_indices = features[2].cpu().tolist()

            # Move input features to device
            input_features = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in input_features.items()}

            # Forward pass
            model_predictions = base_model.model(**input_features, return_dict=True, output_hidden_states=True)

            # Get first layer hidden states (after embedding layer)
            hidden_states = model_predictions.hidden_states.hidden_states[1]

            local_batch_size = len(table1_start)

            for i in range(local_batch_size):
                table_idx = (idx * batch_size) + i
                if table_idx >= len(all_tables):
                    break

                # Extract embeddings for table1 portion only (table2 is identical)
                table_emb, col_embs, cls_emb = find_table_col(
                    bert_tokenizer,
                    hidden_states[i],
                    table1_start[i],
                    table1_end[i],
                    input_features['input_ids'][i]
                )

                embeddings.append({
                    'table_id': extract_table_id(all_tables[table_idx]),
                    'table': all_tables[table_idx],
                    'table_embedding': table_emb,
                    'column_embeddings': col_embs,  # Unified format (plural)
                    'cls_embedding': cls_emb,
                })

    return embeddings


def main():
    parser = ArgumentParser(description="Extract embeddings from TabSketchFM models")
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='Path to checkpoint (.ckpt/.pt) or HuggingFace model name')
    parser.add_argument('--model_type', type=str, default='pretrained',
                        choices=['pretrained', 'finetuned'],
                        help='Type of model: pretrained (single table) or finetuned (cross-encoder)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing preprocessed .json.bz2 files')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output pickle file for embeddings')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for extraction')

    args = parser.parse_args()

    # Set device (single GPU or CPU only)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: CPU")

    # Auto-detect model type only if using default
    # User-specified --model_type takes precedence
    if args.model_type == 'pretrained':  # Check if it's the default
        if args.model_name_or_path.endswith('.pt'):
            print("📋 Detected .pt file → assuming finetuned model")
            args.model_type = 'finetuned'
        elif args.model_name_or_path.endswith('.ckpt'):
            print("📋 Detected .ckpt file → using pretrained model type")
    else:
        print(f"📋 Using explicitly specified model type: {args.model_type}")

    # Load model
    model, actual_type = load_model(args.model_name_or_path, args.model_type, device)
    model = model.to(device)
    model.eval()

    print("\n📋 Note: For multi-GPU extraction, use extract_embeddings_multigpu.py instead")

    # Extract embeddings based on model type
    if actual_type == 'pretrained':
        print("\n🎯 Mode: Single table processing (pretrained model)")
        print("   Each table is processed independently")
        embeddings = extract_from_pretrained_model(model, args.data_dir, args.batch_size, device)
    else:
        print("\n🎯 Mode: Self-pairing (finetuned cross-encoder)")
        print("   Each table is paired with itself to extract embeddings")
        embeddings = extract_from_finetuned_model(model, args.data_dir, args.batch_size, device)

    # Save embeddings
    print(f"\n💾 Saving embeddings to: {args.output_file}")
    with open(args.output_file, 'wb') as f:
        pickle.dump(embeddings, f)

    # Print summary
    print("\n" + "="*60)
    print("📊 EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total tables: {len(embeddings)}")
    if len(embeddings) > 0:
        sample = embeddings[0]
        print(f"CLS embedding shape: {len(sample['cls_embedding'])}")
        print(f"Table embedding shape: {len(sample['table_embedding'])}")
        print(f"Number of columns: {len(sample['column_embeddings'])}")
        print(f"Example column embedding shape: {len(list(sample['column_embeddings'].values())[0])}")
    print("="*60)
    print(f"✅ Done! Embeddings saved to: {args.output_file}")


if __name__ == '__main__':
    main()
