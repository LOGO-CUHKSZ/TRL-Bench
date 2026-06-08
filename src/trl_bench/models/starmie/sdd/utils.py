import torch
import numpy as np
import sklearn.metrics as metrics

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from tqdm import tqdm
from collections import deque, Counter


def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (TableModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            if len(batch) == 4:
                x1, x2, x12, y = batch
                logits = model(x1, x2, x12)
            else:
                x, y = batch
                logits = model(x)

            # print(probs)
            probs = logits.softmax(dim=1)[:, 1]

            # print(logits)
            # pred = logits.argmax(dim=1)
            all_probs += probs.cpu().numpy().tolist()
            # all_p += pred.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1
    else:
        best_th = 0.5
        f1 = 0.0 # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th



def evaluate_column_matching(train, valid, test):
    """Run classification algorithms on feature vectors.
    """
    # datasets = pickle.load(open(feature_path, "rb"))
    # train, valid, test = datasets

    ml_models = {
        "LR": LogisticRegression,
        "SVM": LinearSVC,
        "GB": XGBClassifier, # GradientBoostingClassifier,
        "RF": RandomForestClassifier
    }

    mname = "GB"
    Model = ml_models[mname]
    
    # standardization
    pipe = make_pipeline(StandardScaler(), Model())

    # training
    pipe.fit(np.nan_to_num(train[0]), train[1])

    # eval
    results = {}
    for ds, ds_name in zip([valid, test], ['valid', 'test']):
        X, y = ds
        y_pred = pipe.predict(np.nan_to_num(X))
        f1 = metrics.f1_score(y, y_pred)
        p = metrics.precision_score(y, y_pred)
        r = metrics.recall_score(y, y_pred)

        for var in ["f1", "p", "r"]:
            results[ds_name + "_" + var] = eval(var)
    
    return results


def blocked_matmul(mata, matb,
                   threshold=None,
                   k=None,
                   batch_size=512,
                   exclude_self=False):
    """Find the most similar pairs of vectors from two matrices (top-k or threshold)

    Args:
        mata (np.ndarray): the first matrix
        matb (np.ndarray): the second matrix
        threshold (float, optional): if set, return all pairs of cosine
            similarity above the threshold
        k (int, optional): if set, return for each row in matb the top-k
            most similar vectors in mata
        batch_size (int, optional): the batch size of each block
        exclude_self (bool): if True, filter pairs where idx_a == idx_b

    Returns:
        list of tuples: the pairs of similar vectors' indices and the similarity
    """
    mata = np.array(mata)
    matb = np.array(matb)
    results = []

    # Guard against k >= n (argpartition requires kth < n)
    if k is not None:
        max_k = mata.shape[0] - (1 if exclude_self else 0)
        if min(k, max_k) <= 0:
            return results

    for start in tqdm(range(0, len(matb), batch_size)):
        block = matb[start:start+batch_size]
        sim_mat = np.matmul(mata, block.transpose())
        if k is not None:
            n = sim_mat.shape[0]
            k_fetch = min(k + 1, n) if exclude_self else min(k, n)
            kth = min(k_fetch, n) - 1
            indices = np.argpartition(-sim_mat, kth, axis=0)
            col_counts = {}
            for row in indices[:k_fetch]:
                for idx_b, idx_a in enumerate(row):
                    idx_b += start
                    if exclude_self and idx_a == idx_b:
                        continue
                    cnt = col_counts.get(idx_b, 0)
                    if cnt >= k:
                        continue
                    col_counts[idx_b] = cnt + 1
                    results.append((idx_a, idx_b, sim_mat[idx_a][idx_b-start]))
        elif threshold is not None:
            indices = np.argwhere(sim_mat >= threshold)
            for idx_a, idx_b in indices:
                idx_b += start
                if exclude_self and idx_a == idx_b:
                    continue
                results.append((idx_a, idx_b, sim_mat[idx_a][idx_b-start]))
    return results


def connected_components(pairs, cluster_size=50):
    """Helper function for computing the connected components
    """
    edges = {}
    for left, right, _ in pairs:
        if left not in edges:
            edges[left] = []
        if right not in edges:
            edges[right] = []
            
        edges[left].append(right)
        edges[right].append(left)
    
    print('num nodes =', len(edges))
    all_ccs = []
    used = set([])
    for start in edges:
        if start in used:
            continue
        used.add(start)
        cc = [start]
        
        queue = deque([start])
        while len(queue) > 0:
            u = queue.popleft()
            for v in edges[u]:
                if v not in used:
                    cc.append(v)
                    used.add(v)
                    queue.append(v)
            
            if len(cc) >= cluster_size:
                break
        
        all_ccs.append(cc)
        # print(cc)
    return all_ccs


def evaluate_clustering(vectors, labels):
    """Evaluate column clustering on input column vectors.

    Note: This is Starmie's in-training evaluator. The authoritative benchmark
    evaluator is downstream_tasks/column_clustering/evaluate_clustering.py,
    which additionally sorts edges by similarity and tunes cluster_size via
    binary search. This version uses fixed cluster_size=50 and unsorted edges.
    """
    # L2 normalize for cosine similarity
    vectors = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

    # top 20 matching columns (excluding self-pairs)
    pairs = blocked_matmul(vectors, vectors,
                           k=20,
                           batch_size=4096,
                           exclude_self=True)

    # run column clustering algorithm
    ccs = connected_components(pairs)

    # compute purity (micro-average: sum of dominant counts / total)
    if not ccs:
        return {"num_clusters": 0, "avg_cluster_size": 0.0,
                "purity": 0.0, "nmi": 0.0, "ari": 0.0}

    correct = 0
    total = 0
    for cc in ccs:
        cnt = Counter()
        for column_id in cc:
            cnt[labels[column_id]] += 1
        correct += cnt.most_common(1)[0][1]
        total += len(cc)
    purity = correct / total if total > 0 else 0.0

    # compute NMI and ARI
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    pred_labels = []
    true_labels = []
    for cluster_id, cc in enumerate(ccs):
        for column_id in cc:
            pred_labels.append(cluster_id)
            true_labels.append(labels[column_id])
    nmi = normalized_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')
    ari = adjusted_rand_score(true_labels, pred_labels)

    return {"num_clusters": len(ccs),
            "avg_cluster_size": np.mean([len(cc) for cc in ccs]),
            "purity": purity,
            "nmi": nmi,
            "ari": ari}
