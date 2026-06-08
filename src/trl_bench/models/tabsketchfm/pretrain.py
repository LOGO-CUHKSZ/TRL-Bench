import random
from argparse import ArgumentParser
import pytorch_lightning as pl
from tabsketchfm import Tokenizer, PretrainDataModule, TabSketchFM
from transformers import AutoConfig, AutoTokenizer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch, numpy as np, random




def auto_lr_find(args, lmmodel, tabular_tokenizer):
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1
    )
    sample_data_module = PretrainDataModule(
        model_name_or_path=args.model_name_or_path,
        tokenizer=tabular_tokenizer,
        data_dir=args.data_dir,
        pad_to_max_length=args.pad_to_max_length,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
        max_seq_length=args.max_seq_length,
        mlm_probability=args.mlm_probability,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        run_on_sample=args.run_on_sample,
        sample_size = args.sample_size
    )
    lr_finder = trainer.tuner.lr_find(lmmodel, datamodule=sample_data_module, num_training=100)
    print('lr_finder.results:', lr_finder.results)
   
    new_lr = lr_finder.suggestion()
    print('LR suggestion: ', new_lr)
    if new_lr:
        print('Updating LR based on auto finder: ', new_lr)
        lmmodel.hparams.lr = new_lr
        print(lmmodel.hparams.lr)


def cli_main():

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str,
                        default="bert-base-uncased")
    parser.add_argument('--data_dir', type=str,
                        default='../sample_extracted_test_data')
    parser.add_argument('--dataset', type=str, help='name of the file that contains the train test splits for the data')
    parser.add_argument('--run_on_sample', action='store_true', default=False)
    parser.add_argument('--sample_size', type=int, default=32)
    parser.add_argument('--pad_to_max_length', action='store_true', default=False)
    parser.add_argument('--preprocessing_num_workers', type=int, default=4)
    parser.add_argument('--overwrite_cache', action='store_true', default=False)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--max_token_types', type=int, default=5)
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--dataloader_num_workers', type=int, default=16)
    parser.add_argument('--auto_find_lr', action='store_true', default=False)
    parser.add_argument('--run_local', action='store_true', default=False)
    parser.add_argument('--save_bert_model', action='store_true', default=False)
    parser.add_argument('--bert_model_path', type=str, default='tmp/hf_bert_checkpoint', help="Path to save the trained TabSketchFM model")
    parser.add_argument('--cols_equal', action='store_true', default=False)
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--preprocessed_data', type=int, default=1)

    # Trainer arguments (manually added for PyTorch Lightning 2.x compatibility)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='auto')
    parser.add_argument('--max_epochs', type=int, default=40)
    parser.add_argument('--min_epochs', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.add_argument('--precision', type=str, default='32-true')
    parser.add_argument('--default_root_dir', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to a Lightning checkpoint to resume training from (use "last" or "best" for shortcuts)')

    # Wandb logging arguments
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='tabsketchfm-pretrain', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity/team name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')

    parser = TabSketchFM.add_model_specific_args(parser)
    args = parser.parse_args()
    
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    pl.seed_everything(args.random_seed)
    print("SEEDED ALL")


    # ------------
    # data
    # ------------
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.max_position_embeddings = args.max_seq_length
    toks = AutoTokenizer.from_pretrained(args.model_name_or_path)


    # ------------
    # Data Module
    # ------------


    tokenizer = Tokenizer(tokenizer=toks, config=config)

    data_module = PretrainDataModule(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        dataset=args.dataset,
        pad_to_max_length=args.pad_to_max_length,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
        max_seq_length=args.max_seq_length,
        mlm_probability=args.mlm_probability,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        run_on_sample=args.run_on_sample,
        sample_size=args.sample_size,
        cols_equal = args.cols_equal,
        concat=True,
        preprocessed_data = bool(args.preprocessed_data==1)
    )

    # ------------
    # model
    # ------------

    model = TabSketchFM(
        model_name_or_path=args.model_name_or_path,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon
    )

    # ------------
    # training
    # ------------
    early_stop_callback = EarlyStopping(
        monitor="valid_loss",
        min_delta=0.0,
        patience=5,
        verbose=True,
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    print('Parameters:')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    # Setup logger
    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            save_dir=args.default_root_dir or './logs'
        )
        print(f"✅ W&B logging enabled: {args.wandb_project}")
    else:
        print("ℹ️  W&B logging disabled (use --use_wandb to enable)")

    # PyTorch Lightning 2.x compatible Trainer initialization
    trainer_ddp = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        max_steps=args.max_steps,
        log_every_n_steps=args.log_every_n_steps,
        precision=args.precision,
        default_root_dir=args.default_root_dir,
        gradient_clip_val=0.5,
        callbacks=[early_stop_callback, lr_monitor],
        logger=logger
    )

    if args.auto_find_lr:
        auto_lr_find(args, model, tokenizer)
        exit(0)

    ckpt_path = args.ckpt_path.strip() if args.ckpt_path else None
    if ckpt_path and ckpt_path.lower() == "none":
        ckpt_path = None

    trainer_ddp.fit(model, data_module, ckpt_path=ckpt_path)
    if args.save_bert_model:
        model.model.bert.save_pretrained(args.bert_model_path)
        tokenizer.save_tokenizer_checkpoint(args.bert_model_path)
        
    #shows metrics for last epoch only
    print(trainer_ddp.logged_metrics)
    
    
    # ------------
    # testing
    # ------------
    trainer_ddp.test(model, dataloaders=data_module.test_dataloader())


if __name__ == '__main__':
    cli_main()
