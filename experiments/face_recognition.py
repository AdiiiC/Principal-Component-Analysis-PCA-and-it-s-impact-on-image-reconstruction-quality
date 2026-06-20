"""
Face Recognition using PCA features.
"""

import numpy as np
import pandas as pd
from typing import List
from pca_core import PCAEngine


def run_face_recognition_experiment(
    data_flat: np.ndarray,
    labels: np.ndarray,
    components_range: List[int] = None,
    classifier: str = "svm",
    test_size: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate face recognition accuracy vs number of PCA components.

    Args:
        data_flat: (N, D) flattened images
        labels: (N,) subject IDs
        components_range: List of component counts to test
        classifier: 'svm', 'knn', or 'logistic'
        test_size: Fraction of data for testing
        random_state: Random seed

    Returns:
        DataFrame with n_components vs accuracy
    """
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    if components_range is None:
        components_range = list(range(5, 101, 5))

    X_train, X_test, y_train, y_test = train_test_split(
        data_flat, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    results = []

    for k in components_range:
        # Fit PCA
        engine = PCAEngine(n_components=k)
        engine.fit(X_train)

        # Transform to PCA space
        Z_train = engine.transform(X_train)
        Z_test = engine.transform(X_test)

        # Train classifier
        if classifier == "svm":
            clf = SVC(kernel="rbf", random_state=random_state)
        elif classifier == "knn":
            clf = KNeighborsClassifier(n_neighbors=3)
        elif classifier == "logistic":
            clf = LogisticRegression(max_iter=1000, random_state=random_state)
        else:
            raise ValueError(f"Unknown classifier: {classifier}")

        clf.fit(Z_train, y_train)
        y_pred = clf.predict(Z_test)

        results.append({
            "n_components": k,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        })

    return pd.DataFrame(results)
