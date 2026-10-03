"""Task 1: calculate the classification report without sklearn metrics."""
import numpy as np

LABELS = np.arange(4)


def confusion_matrix(y_true, y_pred, labels=LABELS):
    """Count actual classes in rows and predicted classes in columns."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.asarray(labels)

    if y_true.ndim != 1 or y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Labels must be non-empty, matching one-dimensional arrays")
    if labels.ndim != 1 or len(np.unique(labels)) != len(labels):
        raise ValueError("labels must be unique and one-dimensional")
    if not np.isin(y_true, labels).all() or not np.isin(y_pred, labels).all():
        raise ValueError("An observation has a label outside the supplied classes")

    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for row, actual_class in enumerate(labels):
        for column, predicted_class in enumerate(labels):
            matches = (y_true == actual_class) & (y_pred == predicted_class)
            matrix[row, column] = np.sum(matches)
    return matrix


def classification_report(y_true, y_pred, labels=LABELS):
    """Return per-class scores, accuracy, macro averages and weighted averages."""
    matrix = confusion_matrix(y_true, y_pred, labels)
    support = matrix.sum(axis=1)
    total = support.sum()

    precision = np.zeros(len(labels))
    recall = np.zeros(len(labels))
    f1 = np.zeros(len(labels))
    report = {}

    # Work on one class at a time, treating the other classes as negative.
    for index, label in enumerate(labels):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive

        # Keep zero when a denominator is zero, like sklearn's zero_division=0.
        if true_positive + false_positive > 0:
            precision[index] = true_positive / (true_positive + false_positive)
        if true_positive + false_negative > 0:
            recall[index] = true_positive / (true_positive + false_negative)
        if precision[index] + recall[index] > 0:
            f1[index] = (
                2 * precision[index] * recall[index]
                / (precision[index] + recall[index])
            )

        report[str(label)] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1-score": float(f1[index]),
            "support": int(support[index]),
        }

    report["accuracy"] = float(np.trace(matrix) / total)
    report["macro avg"] = {
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1-score": float(np.mean(f1)),
        "support": int(total),
    }

    # A class with more actual samples contributes more to the weighted score.
    weights = support / total
    report["weighted avg"] = {
        "precision": float(np.sum(weights * precision)),
        "recall": float(np.sum(weights * recall)),
        "f1-score": float(np.sum(weights * f1)),
        "support": int(total),
    }
    return report


class ClassificationMetrics:
    """Methods inherited by LogisticRegression, all using the calculation above."""

    @staticmethod
    def accuracy(y_true, y_pred):
        report = classification_report(y_true, y_pred)
        return report["accuracy"]

    @staticmethod
    def precision(y_true, y_pred):
        report = classification_report(y_true, y_pred)
        return np.array([report[str(label)]["precision"] for label in LABELS])

    @staticmethod
    def recall(y_true, y_pred):
        report = classification_report(y_true, y_pred)
        return np.array([report[str(label)]["recall"] for label in LABELS])

    @staticmethod
    def f1_score(y_true, y_pred):
        report = classification_report(y_true, y_pred)
        return np.array([report[str(label)]["f1-score"] for label in LABELS])

    @staticmethod
    def macro_precision(y_true, y_pred):
        report = classification_report(y_true, y_pred)
        return report["macro avg"]["precision"]

    @staticmethod
    def macro_recall(y_true, y_pred):
        report = classification_report(y_true, y_pred)
        return report["macro avg"]["recall"]

    @staticmethod
    def macro_f1(y_true, y_pred):
        report = classification_report(y_true, y_pred)
        return report["macro avg"]["f1-score"]

    @staticmethod
    def weighted_precision(y_true, y_pred):
        report = classification_report(y_true, y_pred)
        return report["weighted avg"]["precision"]

    @staticmethod
    def weighted_recall(y_true, y_pred):
        report = classification_report(y_true, y_pred)
        return report["weighted avg"]["recall"]

    @staticmethod
    def weighted_f1(y_true, y_pred):
        report = classification_report(y_true, y_pred)
        return report["weighted avg"]["f1-score"]
