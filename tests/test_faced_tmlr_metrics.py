import numpy as np

from tmlr.metrics import classification_metrics, selection_value


def test_metrics_known_example_and_selection_names():
    targets = [0, 0, 1, 1, 2, 2]
    predictions = [0, 1, 1, 1, 2, 0]
    metrics = classification_metrics(targets, predictions, losses=[0.5] * 6, num_classes=3)
    assert np.isclose(metrics["accuracy"], 4 / 6)
    assert len(metrics["per_class_f1"]) == 3
    assert len(metrics["confusion_matrix"]) == 3
    assert selection_value(metrics, "balanced_accuracy") == metrics["balanced_accuracy"]
    assert selection_value(metrics, "macro_f1") == metrics["macro_f1"]


def test_singleton_accumulation_shape_is_safe():
    metrics = classification_metrics([2], [2], losses=[0.2], num_classes=3)
    assert metrics["sample_count"] == 1
    assert metrics["per_class_support"] == [0, 0, 1]
