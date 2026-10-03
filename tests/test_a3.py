"""Required input/output tests plus independent checks of the ML mathematics."""
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report as sklearn_report
from a3.data import FEATURES, price_classes
from a3.metrics import classification_report
from a3.model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]


def sample_input():
    return pd.DataFrame([dict(year=2017., km_driven=40000., owner=1., mileage=20.,
                             engine=1200., max_power=80., seats=5., brand="Maruti",
                             fuel="Petrol", seller_type="Individual", transmission="Manual")], columns=FEATURES)


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = joblib.load(ROOT / "models/best_a3_model.joblib")

    def test_model_takes_expected_input(self):
        """Task 3 test (1): a valid car row works, including missing/unseen values."""
        row = sample_input()
        predicted = self.pipeline.predict(row)
        self.assertIn(predicted[0], [0, 1, 2, 3])
        row.loc[0, "mileage"] = np.nan
        row.loc[0, "brand"] = "Previously unseen manufacturer"
        self.assertIn(self.pipeline.predict(row)[0], [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            self.pipeline.predict(row.drop(columns=["engine"]))

    def test_model_output_has_expected_shape(self):
        """Task 3 test (2): one class per row, four probabilities per row."""
        for size in [1, 5]:
            rows = pd.concat([sample_input()] * size, ignore_index=True)
            predictions = self.pipeline.predict(rows)
            probabilities = self.pipeline.predict_proba(rows)
            self.assertEqual(predictions.shape, (size,))
            self.assertEqual(probabilities.shape, (size, 4))
            self.assertTrue(np.issubdtype(predictions.dtype, np.integer))
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
            np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=1e-12)
            np.testing.assert_array_equal(predictions, probabilities.argmax(axis=1))


class MathematicalTests(unittest.TestCase):
    def test_metrics_match_sklearn_including_absent_classes(self):
        cases = [([0,0,0,1,1,2,3,3], [0,1,0,1,2,2,0,3]),
                 ([0,0,1], [0,0,0]), ([0,1,2,3], [1,2,3,0])]
        for true, pred in cases:
            own = classification_report(true, pred)
            reference = sklearn_report(true, pred, labels=[0,1,2,3], output_dict=True, zero_division=0)
            for label in ["0", "1", "2", "3", "macro avg", "weighted avg"]:
                for metric in ["precision", "recall", "f1-score", "support"]:
                    self.assertAlmostEqual(own[label][metric], reference[label][metric], places=12)
            self.assertAlmostEqual(own["accuracy"], np.mean(np.array(true) == pred))
            model = LogisticRegression()
            for method, group, metric in [("macro_precision","macro avg","precision"),
                                           ("macro_recall","macro avg","recall"),
                                           ("macro_f1","macro avg","f1-score"),
                                           ("weighted_precision","weighted avg","precision"),
                                           ("weighted_recall","weighted avg","recall"),
                                           ("weighted_f1","weighted avg","f1-score")]:
                self.assertAlmostEqual(getattr(model, method)(true, pred), reference[group][metric])
            for method, metric in [("precision","precision"),("recall","recall"),("f1_score","f1-score")]:
                np.testing.assert_allclose(getattr(model, method)(true,pred), [reference[str(c)][metric] for c in range(4)])

    def test_l2_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(7)
        X, y = rng.normal(size=(7,3)), np.array([0,1,2,3,1,0,2])
        W, b = rng.normal(size=(3,4)), rng.normal(size=4)
        for l2 in [0., .2]:
            model = LogisticRegression(l2=l2)
            _, dw, db = model._loss_gradient(X,y,W,b)
            eps = 1e-6
            for index in np.ndindex(W.shape):
                plus, minus = W.copy(), W.copy()
                plus[index] += eps
                minus[index] -= eps
                numeric = (model._loss_gradient(X,y,plus,b)[0] - model._loss_gradient(X,y,minus,b)[0]) / (2*eps)
                self.assertAlmostEqual(dw[index], numeric, places=6)
            for index in range(4):
                plus, minus = b.copy(), b.copy()
                plus[index] += eps
                minus[index] -= eps
                numeric = (model._loss_gradient(X,y,W,plus)[0] - model._loss_gradient(X,y,W,minus)[0]) / (2*eps)
                self.assertAlmostEqual(db[index], numeric, places=6)
        baseline = LogisticRegression(l2=0)._loss_gradient(X,y,W,b)
        ridge = LogisticRegression(l2=.2)._loss_gradient(X,y,W,b)
        np.testing.assert_allclose(ridge[1] - baseline[1], .4*W)
        np.testing.assert_allclose(ridge[2], baseline[2])

    def test_softmax_is_stable_and_model_learns(self):
        X = np.repeat(np.eye(4), 10, axis=0)
        y = np.repeat(np.arange(4), 10)
        model = LogisticRegression(lr=.3).fit(X,y)
        np.testing.assert_array_equal(model.predict(X), y)
        self.assertLess(model.loss_history_[-1], model.loss_history_[0])
        logits = np.array([[10000., 10001., -10000., 0.]])
        p = np.exp(model._log_softmax(logits))
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(axis=1),1.)
        with self.assertRaises(ValueError):
            model.predict(np.ones((2,5)))

    def test_price_boundary_rules(self):
        np.testing.assert_array_equal(price_classes([1,300000,300001,600000,600001,1000000,1000001]), [0,0,1,1,2,2,3])
        with self.assertRaises(ValueError):
            price_classes([0, np.nan])


if __name__ == "__main__":
    unittest.main()
