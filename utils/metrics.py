from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc, precision_recall_curve
import numpy as np
import math

def mrr_score(y_true, y_score):
    """Computing mrr score metric.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss.
    FIXME:
        refactor this with the reco metrics and make it explicit.
    """
    res = {}
    for metric in metrics:
        if metric == "auc":
            tmp_labels, tmp_preds = [], []
            for l, p in zip(labels, preds):
                tmp_labels += l
                tmp_preds += p
            auc = roc_auc_score(np.asarray(tmp_labels), np.asarray(tmp_preds))
            res["auc"] = round(auc, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric == "group_auc":
            auc_list = []
            for each_labels, each_preds in zip(labels, preds):
                try:
                    x = roc_auc_score(each_labels, each_preds)
                    auc_list.append(x)
                except:
                    print("There are only zero labels")
                    auc_list.append(0.0)
            group_auc = np.mean(
                auc_list
            )
            res["group_auc"] = round(group_auc, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res

def cm(y_pred, y_target, k=5000000):

    precision, recall, _ = precision_recall_curve(y_target, y_pred)
    Z = np.stack((y_target, y_pred), axis=1).astype(np.float64)
    (n_sample, n_feature) = Z.shape
    # metrics
    loglikelihood = (np.sum(Z[:, 0] * np.log(Z[:, 1])) + 
    np.sum((1 - Z[:, 0]) * np.log(1 - Z[:, 1]))) / n_sample
    overallp = np.sum(Z[:, 0]) / n_sample
    H = -(overallp * np.log(overallp + 0.000001) + (1 - overallp) * np.log(1 - overallp + 0.000001))
    rig = (loglikelihood + H) / H

    return {
        "ROC AUC": roc_auc_score(y_target, y_pred),
        "MRR": mrr_score(y_target, y_pred),
        "nDCG": ndcg_score(y_target, y_pred, k=y_pred.shape[0] // 10, k=k),
        "Precison-Recall AUC": auc(recall, precision),
        "RIG": rig
    }

def compute_metrics(p):
    y_pred = p.predictions[0].flatten()
    y_target = p.label_ids
    return cm(y_pred, y_target)
