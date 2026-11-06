"""Calibration and ranking metrics, on the host (numpy)."""
import numpy as np


def softmax(logits, T=1.0):
    z = logits / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def reliability(conf, correct, bins=10):
    """Bins of confidence: [{lo, hi, count, confidence (mean), accuracy}], and the expected calibration error:
    the count-weighted mean |accuracy - confidence|"""
    conf, correct = np.asarray(conf, np.float64), np.asarray(correct, np.float64)
    edges = np.linspace(0, 1, bins + 1)
    out, ece = [], 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        n = int(m.sum())
        if n:
            c, a = float(conf[m].mean()), float(correct[m].mean())
            ece += n / len(conf) * abs(a - c)
        else:
            c = a = None
        out.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "count": n, "confidence": c, "accuracy": a})
    return out, float(ece)


def multiclass(probs, labels, bins=10):
    """Accuracy, NLL, Brier score and ECE (on the top choice) of probabilities [n, J] against labels [n]"""
    n = len(labels)
    p_true = probs[np.arange(n), labels]
    pred = probs.argmax(1)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1
    rel, ece = reliability(probs.max(1), pred == labels, bins)
    return {"accuracy": float((pred == labels).mean()), "nll": float(-np.log(np.maximum(p_true, 1e-12)).mean()),
            "brier": float(((probs - onehot) ** 2).sum(1).mean()), "ece": ece, "n": n}, rel


def fit_temperature(logits, labels):
    """The temperature T minimizing the NLL of softmax(logits / T): a grid, then a finer one"""
    def nll(T):
        p = softmax(logits, T)
        return -np.log(np.maximum(p[np.arange(len(labels)), labels], 1e-12)).mean()
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 120))
    best = grid[np.argmin([nll(t) for t in grid])]
    fine = best * np.exp(np.linspace(-0.06, 0.06, 41))
    return float(fine[np.argmin([nll(t) for t in fine])])


def auc(scores, y):
    """Area under the ROC curve (the Mann-Whitney statistic, ties averaged)"""
    y = np.asarray(y, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if not npos or not nneg:
        return None
    order = np.argsort(scores, kind="mergesort")
    s = np.asarray(scores)[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def best_f1(scores, y):
    """(threshold, F1) maximizing F1"""
    y = np.asarray(y, bool)
    order = np.argsort(-scores)
    s, yy = scores[order], y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(~yy)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int(y.sum()), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    return float(s[i]), float(f1[i])


def f1_at(scores, y, th):
    y = np.asarray(y, bool)
    p = scores >= th
    tp = int((p & y).sum())
    prec = tp / max(int(p.sum()), 1)
    rec = tp / max(int(y.sum()), 1)
    return 2 * prec * rec / max(prec + rec, 1e-12)
