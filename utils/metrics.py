from pydoc import cli
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

def compute_metrics(p):
    y_pred = p.predictions.flatten()
    precision, recall, _ = precision_recall_curve(p.label_ids, y_pred)
    ysize = p.label_ids.shape[0]
    clickp = sum(p.label_ids) / ysize
    yEntropy = -(clickp * math.log(clickp) + (1 - clickp) * math.log(1 - clickp))

    lsum = []
    # implement the same as pytorch: log output show be greater or equal than -100
    for i in range(ysize):
        if p.label_ids[i] == 1:
            if y_pred[i] == 0:
                lsum.append(-100)
            else:
                max(lsum.append(math.log(y_pred[i])), -100)
        else:
            if y_pred[i] == 1:
                lsum.append(-100)
            else:
                max(lsum.append(math.log(1 - y_pred[i])), -100)

    llxy = sum(lsum)

    return {
        "ROC AUC": roc_auc_score(p.label_ids, y_pred),
        "MRR": mrr_score(p.label_ids, y_pred),
        "nDCG": ndcg_score(p.label_ids, y_pred, k=y_pred.shape[0] // 10),
        "Precison-Recall AUC": auc(recall, precision),
        "RIG": (llxy / ysize + yEntropy) / yEntropy
    }