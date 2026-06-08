#!/usr/bin/env python3
"""Test a finetuned TabSketchFM checkpoint on the test set."""

import json
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from tabsketchfm import TableSimilarityTokenizer, FinetuneDataModule, FinetuneTabSketchFM
from transformers import AutoConfig, AutoTokenizer
import torch.utils.data

def main():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to processed data directory')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset JSON file')
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
    parser.add_argument('--task_type', type=str, default='classification')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--dataloader_num_workers', type=int, default=16)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--preprocessed_data', type=int, default=1)

    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")

    # Initialize BERT config and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.max_position_embeddings = args.max_seq_length
    bert_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Initialize tabular tokenizer
    config.task_specific_params = {'hash_input_size': config.hidden_size}
    tokenizer = TableSimilarityTokenizer(tokenizer=bert_tokenizer, config=config)

    # Initialize data module
    data_module = FinetuneDataModule(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        dataset=args.dataset,
        pad_to_max_length=False,
        preprocessing_num_workers=4,
        overwrite_cache=False,
        max_seq_length=args.max_seq_length,
        mlm_probability=0.15,
        train_batch_size=32,  # Not used for testing
        val_batch_size=args.val_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        run_on_sample=False,
        concat=True  # Important: concatenate table pairs
    )

    # Custom collate function to properly batch dictionary data
    def collate_fn(batch):
        # batch is a list of (data_dict, label) tuples
        data_dicts = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Stack tensors for each key in the data dictionary
        batched_data = {}
        for key in data_dicts[0].keys():
            first_val = data_dicts[0][key]
            if isinstance(first_val, tuple):
                # If the value is a tuple (data1, data2), stack each separately
                data1_list = [d[key][0] for d in data_dicts]
                data2_list = [d[key][1] for d in data_dicts]
                batched_data[key] = (torch.stack(data1_list), torch.stack(data2_list))
            else:
                # If the value is a tensor, stack normally
                batched_data[key] = torch.stack([d[key] for d in data_dicts])

        # Stack labels
        batched_labels = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels)

        return batched_data, batched_labels

    # Monkey-patch the test_dataloader method to use custom collate_fn
    from torch.utils.data import DataLoader
    original_test_dataloader = data_module.test_dataloader

    def patched_test_dataloader():
        # Call setup if not already done
        if not hasattr(data_module, 'data_splits'):
            data_module.setup('test')

        # Get the dataset
        test_files = data_module.data_splits['test']
        from tabsketchfm import TableSimilarityDataset
        test_dataset = TableSimilarityDataset(
            data_dir=data_module.data_dir,
            table_similarity=test_files,
            transform=data_module.tokenizer.tokenize_function,
            concat=data_module.concat,
            cols_equal=data_module.cols_equal,
            preprocessed_data=data_module.preprocessed_data,
            extract_embedding=data_module.extract_embedding,
            send_idx=data_module.send_idx
        )

        return DataLoader(
            dataset=test_dataset,
            shuffle=data_module.shuffle,
            batch_size=data_module.val_batch_size,  # Use val_batch_size for testing
            drop_last=data_module.drop_last,
            num_workers=data_module.dataloader_num_workers,
            collate_fn=collate_fn  # Add custom collate function
        )

    data_module.test_dataloader = patched_test_dataloader

    # Load model from checkpoint
    print(f"Loading model from checkpoint...")
    model = FinetuneTabSketchFM.load_from_checkpoint(
        args.checkpoint,
        model_name_or_path=args.model_name_or_path,
        config=config,
        num_labels=args.num_labels,
        model_type=args.task_type
    )

    # Initialize trainer for testing
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False
    )

    print("\n" + "="*60)
    print("Running test evaluation...")
    print("="*60 + "\n")

    # Run test (trainer will call data_module.setup() automatically)
    results = trainer.test(model, datamodule=data_module)

    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    for key, value in results[0].items():
        print(f"{key}: {value:.4f}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
