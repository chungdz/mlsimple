from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def compute_metrics(p):
    return {
        "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
        "f1_score": f1_score(p.label_ids, np.argmax(p.predictions, axis=1))
    }