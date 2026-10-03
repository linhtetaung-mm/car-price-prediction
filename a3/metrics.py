"""Classification metrics from counts, with sklearn's zero_division=0 convention."""
import numpy as np

LABELS = np.arange(4)


def confusion_matrix(y_true, y_pred, labels=LABELS):
    """Rows are actual classes; columns are predicted classes."""
    y_true, y_pred, labels = map(np.asarray, (y_true, y_pred, labels))
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or not y_true.size:
        raise ValueError("Labels must be non-empty, matching one-dimensional arrays")
    if labels.ndim != 1 or len(np.unique(labels)) != len(labels):
        raise ValueError("labels must be unique and one-dimensional")
    if not np.isin(y_true, labels).all() or not np.isin(y_pred, labels).all():
        raise ValueError("An observation has a label outside the supplied classes")
    return np.array([[np.sum((y_true == a) & (y_pred == b)) for b in labels]
                     for a in labels], dtype=int)


def _divide(numerator, denominator):
    # Undefined precision/recall is zero, even if a class is never predicted.
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                     where=denominator != 0)


def classification_report(y_true, y_pred, labels=LABELS):
    cm = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm)
    support = cm.sum(axis=1)  # Number of ACTUAL examples of each class.
    precision = _divide(tp, cm.sum(axis=0))
    recall = _divide(tp, support)
    f1 = _divide(2 * precision * recall, precision + recall)
    report = {str(c): {"precision": float(p), "recall": float(r),
                      "f1-score": float(f), "support": int(s)}
              for c, p, r, f, s in zip(labels, precision, recall, f1, support)}
    report["accuracy"] = float(tp.sum() / support.sum())
    for name, weights in [("macro avg", np.ones(len(labels)) / len(labels)),
                          ("weighted avg", support / support.sum())]:
        report[name] = {"precision": float(weights @ precision),
                        "recall": float(weights @ recall),
                        "f1-score": float(weights @ f1), "support": int(support.sum())}
    return report


class ClassificationMetrics:
    """Expose the assignment's requested functions directly on the model class."""
    @staticmethod
    def accuracy(y_true, y_pred):
        return classification_report(y_true, y_pred)["accuracy"]

    @staticmethod
    def precision(y_true, y_pred):
        r = classification_report(y_true, y_pred)
        return np.array([r[str(c)]["precision"] for c in LABELS])

    @staticmethod
    def recall(y_true, y_pred):
        r = classification_report(y_true, y_pred)
        return np.array([r[str(c)]["recall"] for c in LABELS])

    @staticmethod
    def f1_score(y_true, y_pred):
        r = classification_report(y_true, y_pred)
        return np.array([r[str(c)]["f1-score"] for c in LABELS])

    @staticmethod
    def macro_precision(y_true, y_pred):
        return classification_report(y_true, y_pred)["macro avg"]["precision"]

    @staticmethod
    def macro_recall(y_true, y_pred):
        return classification_report(y_true, y_pred)["macro avg"]["recall"]

    @staticmethod
    def macro_f1(y_true, y_pred):
        return classification_report(y_true, y_pred)["macro avg"]["f1-score"]

    @staticmethod
    def weighted_precision(y_true, y_pred):
        return classification_report(y_true, y_pred)["weighted avg"]["precision"]

    @staticmethod
    def weighted_recall(y_true, y_pred):
        return classification_report(y_true, y_pred)["weighted avg"]["recall"]

    @staticmethod
    def weighted_f1(y_true, y_pred):
        return classification_report(y_true, y_pred)["weighted avg"]["f1-score"]
