import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Data_pipeline.Data_transformation import preprocessor, x_train, y_train, x_test, y_test
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    recall_score, precision_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import joblib

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", SVC(probability=True))
])

params = {
    "model__C"     : [0.1, 1, 10, 100],
    "model__kernel": ["linear", "rbf"],
    "model__gamma" : ["scale", "auto"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipe,
    params,
    cv=cv,
    scoring="recall",
    n_jobs=-1
)
grid.fit(x_train, y_train)
best_model = grid.best_estimator_
joblib.dump(best_model, "finalixed_model.pkl")
