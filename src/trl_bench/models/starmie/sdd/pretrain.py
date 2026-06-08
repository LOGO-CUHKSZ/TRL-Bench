import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
import mlflow
import pandas as pd
import os

from .utils import evaluate_column_matching, evaluate_clustering
from .model import BarlowTwinsSimCLR
from .dataset import PretrainTableDataset

from tqdm import tqdm
from torch.utils import data
from transformers import get_linear_schedule_with_warmup
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW
from typing import List

_MLFLOW_CONFIGURED = False


def configure_mlflow_logging(hp=None):
    """Ensure MLflow logs go to a custom directory instead of ./mlruns."""
    global _MLFLOW_CONFIGURED
    if _MLFLOW_CONFIGURED:
        return

    # Give precedence to explicit tracking URIs so we do not override user intent.
    explicit_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if explicit_uri:
        _MLFLOW_CONFIGURED = True
        return

    candidate_dir = None

    # Allow environment overrides without touching MLflow internals.
    env_dir = os.environ.get("MLFLOW_TRACKING_DIR")
    if env_dir:
        candidate_dir = env_dir

    # Fall back to hp overrides
    if candidate_dir is None and hp is not None:
        candidate_dir = getattr(hp, "mlflow_dir", None)
    if candidate_dir is None and hp is not None:
        candidate_dir = getattr(hp, "mlflow_tracking_dir", None)

    # Use the checkpoint directory if nothing else is provided.
    if candidate_dir is None:
        checkpoint_root = getattr(hp, "checkpoint_dir", None) if hp is not None else None
        if checkpoint_root is None and hp is not None:
            checkpoint_root = getattr(hp, "logdir", None)
        root_dir = checkpoint_root
        data_path = getattr(hp, "data_path", None) if hp is not None else None
        dataset_name = os.path.basename(os.path.normpath(data_path)) if data_path else ""
        if root_dir:
            if dataset_name:
                candidate_dir = os.path.join(root_dir, dataset_name, "mlflow_runs")
            else:
                candidate_dir = os.path.join(root_dir, "mlflow_runs")
        else:
            candidate_dir = os.path.join(os.getcwd(), "mlflow_runs")

    candidate_dir = os.path.abspath(candidate_dir)
    os.makedirs(candidate_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{candidate_dir}")
    _MLFLOW_CONFIGURED = True


def train_step(train_iter, model, optimizer, scheduler, scaler, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (BarlowTwinsSimCLR): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        scaler (GradScaler): gradient scaler for fp16 training
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    for i, batch in enumerate(train_iter):
        x_ori, x_aug, cls_indices = batch
        optimizer.zero_grad()

        if hp.fp16:
            with torch.cuda.amp.autocast():
                loss = model(x_ori, x_aug, cls_indices, mode='simclr')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            loss = model(x_ori, x_aug, cls_indices, mode='simclr')
            loss.backward()
            optimizer.step()

        scheduler.step()
        if i % 10 == 0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")
        del loss


def train(trainset, hp):
    """Train and evaluate the model

    Args:
        trainset (PretrainTableDataset): the training set
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)
    Returns:
        The pre-trained table model
    """
    configure_mlflow_logging(hp)

    checkpoint_root = getattr(hp, "checkpoint_dir", None) or getattr(hp, "logdir", None)
    if checkpoint_root is None:
        raise ValueError("A checkpoint directory is required. Pass --checkpoint_dir when launching training.")

    # Derive resume checkpoint path. ``--checkpoint_subdir`` pins the subdir so
    # the saved .pt path is independent of the input tables-dir basename (which
    # is only 'datalake' for union_search inputs); default preserves the old
    # basename behavior for direct callers.
    dataset_name = getattr(hp, "checkpoint_subdir", None) or \
        os.path.basename(os.path.normpath(hp.data_path))
    resume_dir = os.path.join(checkpoint_root, dataset_name)
    suffix = '_'+str(hp.augment_op)+'_'+str(hp.sample_meth)+'_'+str(hp.table_order)+'_'+str(hp.run_id)
    if hp.single_column:
        suffix += 'singleCol'
    resume_path = os.path.join(resume_dir, 'resume' + suffix + '.pt')

    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    if hp.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    start_epoch = 1
    if os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        resume_ckpt = torch.load(resume_path, weights_only=False)
        model.load_state_dict(resume_ckpt['model'])
        optimizer.load_state_dict(resume_ckpt['optimizer'])
        scheduler.load_state_dict(resume_ckpt['scheduler'])
        if scaler is not None and 'scaler' in resume_ckpt:
            scaler.load_state_dict(resume_ckpt['scaler'])
        start_epoch = resume_ckpt['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        del resume_ckpt

    for epoch in range(start_epoch, hp.n_epochs+1):
        # train
        model.train()
        train_step(train_iter, model, optimizer, scheduler, scaler, hp)

        # save resume checkpoint every epoch
        if hp.save_model:
            if not os.path.exists(resume_dir):
                os.makedirs(resume_dir)
            resume_ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'hp': hp,
            }
            if scaler is not None:
                resume_ckpt['scaler'] = scaler.state_dict()
            torch.save(resume_ckpt, resume_path)

        # save the last checkpoint
        if hp.save_model and epoch == hp.n_epochs:
            # Use the last component of data_path as the subdirectory name
            directory = os.path.join(checkpoint_root, dataset_name)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # save the checkpoints for each component
            if hp.single_column:
                ckpt_path = os.path.join(checkpoint_root, dataset_name, 'model_'+str(hp.augment_op)+'_'+str(hp.sample_meth)+'_'+str(hp.table_order)+'_'+str(hp.run_id)+'singleCol.pt')
            else:
                ckpt_path = os.path.join(checkpoint_root, dataset_name, 'model_'+str(hp.augment_op)+'_'+str(hp.sample_meth)+'_'+str(hp.table_order)+'_'+str(hp.run_id)+'.pt')

            ckpt = {'model': model.state_dict(),
                    'hp': hp}
            torch.save(ckpt, ckpt_path)

            # clean up resume checkpoint after successful final save
            if os.path.exists(resume_path):
                os.remove(resume_path)

        # intrinsic evaluation with column matching (if evaluation data exists)
        # Train column matching models using the learned representations
        eval_files_exist = all([
            os.path.exists(os.path.join(hp.data_path, f'{split}.csv'))
            for split in ["train", "valid", "test"]
        ]) and os.path.exists(os.path.join(hp.data_path, 'tables'))

        if eval_files_exist:
            metrics_dict = evaluate_pretrain(model, trainset)
            # log metrics
            mlflow.log_metrics(metrics_dict)
            print("epoch %d: " % epoch + ", ".join(["%s=%f" % (k, v) \
                                    for k, v in metrics_dict.items()]))

            # Train column matching models using the learned representations
            metrics_dict = evaluate_column_clustering(model, trainset)
            # log metrics
            mlflow.log_metrics(metrics_dict)
            print("epoch %d: " % epoch + ", ".join(["%s=%f" % (k, v) \
                                    for k, v in metrics_dict.items()]))
        else:
            print(f"epoch %d: Skipping evaluation (evaluation data not found in {hp.data_path})" % epoch)



def inference_on_tables(tables: List[pd.DataFrame],
                        model: BarlowTwinsSimCLR,
                        unlabeled: PretrainTableDataset,
                        batch_size=128,
                        total=None):
    """Extract column vectors from a table.

    Args:
        tables (List of DataFrame): the list of tables
        model (BarlowTwinsSimCLR): the model to be evaluated
        unlabeled (PretrainTableDataset): the unlabeled dataset
        batch_size (optional): batch size for model inference

    Returns:
        List of np.array: the column vectors
    """
    total=total if total is not None else len(tables)
    batch = []
    results = []
    for tid, table in tqdm(enumerate(tables), total=total):
        x, _ = unlabeled._tokenize(table)

        batch.append((x, x, []))
        if tid == total - 1 or len(batch) == batch_size:
            # model inference
            with torch.no_grad():
                x, _, _ = unlabeled.pad(batch)
                # all column vectors in the batch
                column_vectors = model.inference(x)
                ptr = 0
                for xi in x:
                    current = []
                    for token_id in xi:
                        if token_id == unlabeled.tokenizer.cls_token_id:
                            current.append(column_vectors[ptr].cpu().numpy())
                            ptr += 1
                    results.append(current)

            batch.clear()

    return results


def load_checkpoint(ckpt, ds_path_override: str | None = None):
    """Load a model from a checkpoint.
        ** If you would like to run your own benchmark, update the ds_path here

    The checkpoint dict stores a baked-in ``hp.data_path`` (or ``hp.task``)
    that points at the original training-data directory on the upstream
    author's filesystem. If you've moved or renamed that directory pass
    ``ds_path_override`` so the dataset constructor uses your local CSV
    directory instead of the unresolvable bake-in.

    Args:
        ckpt (str): the model checkpoint.
        ds_path_override (str, optional): override for the checkpoint-stored
            data path (the directory of CSVs the tokenizer + max-len logic
            walks at init time). When ``None``, falls back to the
            checkpoint's stored ``data_path`` / ``task`` path.

    Returns:
        BarlowTwinsSimCLR: the pre-trained model
        PretrainDataset: the dataset for pre-training the model
    """
    hp = ckpt['hp']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm)
    model = model.to(device)
    # ``strict=False`` accommodates HuggingFace transformers' v4.x ``position_ids``
    # buffer rename: older Starmie checkpoints (PVLDB 2023 epoch) carry
    # ``bert.embeddings.position_ids`` which the newer modeling code no
    # longer registers as a state-dict entry. Both old + new shape on the
    # actual model weights, so a non-strict load preserves correctness.
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if unexpected:
        # Filter to only the known harmless rename; anything else should
        # surface as a hard failure.
        leftover = [k for k in unexpected if not k.endswith(".position_ids")]
        if leftover:
            raise RuntimeError(
                f"starmie load_checkpoint: unexpected keys in checkpoint "
                f"state_dict that are not the known ``position_ids`` legacy "
                f"buffer: {leftover}"
            )

    # Backward compatibility: handle both old (task) and new (data_path) checkpoints.
    # If the caller supplied an explicit override (e.g. for inference where the
    # checkpoint's baked-in path is unresolvable on the current host), honor it.
    if ds_path_override is not None:
        ds_path = ds_path_override
    elif hasattr(hp, 'data_path'):
        # New checkpoint format
        ds_path = hp.data_path
    elif hasattr(hp, 'task'):
        # Old checkpoint format - map task names to paths
        if "santos" in hp.task:
            ds_path = 'data/%s/datalake' % hp.task
            if hp.task == "santosLarge":
                ds_path = 'data/santos-benchmark/real-benchmark/datalake'
        elif "tus" in hp.task:
            ds_path = 'data/table-union-search-benchmark/small/benchmark'
            if hp.task == "tusLarge":
                ds_path = 'data/table-union-search-benchmark/large/benchmark'
        elif hp.task == "wdc":
            ds_path = 'data/wdc/0'
        else:
            ds_path = 'data/%s/tables' % hp.task
    else:
        raise ValueError("Checkpoint must have either 'data_path' or 'task' attribute")

    dataset = PretrainTableDataset.from_hp(ds_path, hp)

    return model, dataset


def evaluate_pretrain(model: BarlowTwinsSimCLR,
                      unlabeled: PretrainTableDataset):
    """Evaluate pre-trained model.

    Args:
        model (BarlowTwinsSimCLR): the model to be evaluated
        unlabeled (PretrainTableDataset): the unlabeled dataset

    Returns:
        Dict: the dictionary of metrics (e.g., valid_f1)
    """
    table_path = os.path.join(model.hp.data_path, 'tables')

    # encode each dataset
    featurized_datasets = []
    for dataset in ["train", "valid", "test"]:
        ds_path = os.path.join(model.hp.data_path, f'{dataset}.csv')
        ds = pd.read_csv(ds_path)

        def encode_tables(table_ids, column_ids):
            tables = []
            for table_id, col_id in zip(table_ids, column_ids):
                table = pd.read_csv(os.path.join(table_path, \
                                    "table_%d.csv" % table_id))
                if model.hp.single_column:
                    table = table[[table.columns[col_id]]]
                tables.append(table)
            vectors = inference_on_tables(tables, model, unlabeled,
                                          batch_size=128)

            # assert all columns exist
            for vec, table in zip(vectors, tables):
                assert len(vec) == len(table.columns)

            res = []
            for vec, cid in zip(vectors, column_ids):
                if cid < len(vec):
                    res.append(vec[cid])
                else:
                    # single column
                    res.append(vec[-1])
            return res

        # left tables
        l_features = encode_tables(ds['l_table_id'], ds['l_column_id'])

        # right tables
        r_features = encode_tables(ds['r_table_id'], ds['r_column_id'])

        features = []
        Y = ds['match']
        for l, r in zip(l_features, r_features):
            feat = np.concatenate((l, r, np.abs(l - r)))
            features.append(feat)

        featurized_datasets.append((features, Y))

    train, valid, test = featurized_datasets
    return evaluate_column_matching(train, valid, test)


def evaluate_column_clustering(model: BarlowTwinsSimCLR,
                               unlabeled: PretrainTableDataset):
    """Evaluate pre-trained model on a column clustering dataset.

    Args:
        model (BarlowTwinsSimCLR): the model to be evaluated
        unlabeled (PretrainTableDataset): the unlabeled dataset

    Returns:
        Dict: the dictionary of metrics (e.g., purity, number of clusters)
    """
    table_path = os.path.join(model.hp.data_path, 'tables')

    # encode each dataset
    featurized_datasets = []
    ds_path = os.path.join(model.hp.data_path, 'test.csv')
    ds = pd.read_csv(ds_path)
    table_ids, column_ids = ds['table_id'], ds['column_id']

    # encode all tables
    def table_iter():
        for table_id, col_id in zip(table_ids, column_ids):
            table = pd.read_csv(os.path.join(table_path, \
                                "table_%d.csv" % table_id))
            if model.hp.single_column:
                table = table[[table.columns[col_id]]]
            yield table

    vectors = inference_on_tables(table_iter(), model, unlabeled,
                                    batch_size=128, total=len(table_ids))

    # # assert all columns exist
    # for vec, table in zip(vectors, tables):
    #     assert len(vec) == len(table.columns)

    column_vectors = []
    for vec, cid in zip(vectors, column_ids):
        if cid < len(vec):
            column_vectors.append(vec[cid])
        else:
            # single column
            column_vectors.append(vec[-1])

    return evaluate_clustering(column_vectors, ds['class'])
