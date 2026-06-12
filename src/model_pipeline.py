import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from Data_pipeline.Data_transformation import preprocessor, x_train, y_train, x_test, y_test
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ============================================================
# Output directories
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# Model definitions & hyperparameter search spaces
# ============================================================
models = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=42),
        # Tách 2 grid vì l1 không dùng được với lbfgs
        "params": [
            {
                # Grid 1: l2 penalty — dùng được cả lbfgs lẫn liblinear
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs", "liblinear"]
            },
            {
                # Grid 2: l1 penalty — chỉ dùng liblinear hoặc saga
                # l1 có tác dụng feature selection tự động (zeroing out)
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__penalty": ["l1"],
                "model__solver": ["liblinear", "saga"]
            }
        ]
    },

    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "params": {
            "model__max_depth": [None, 3, 5, 7, 10],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4, 8],  # tránh lá quá nhỏ → giảm overfitting
            "model__criterion": ["gini", "entropy"]
        }
    },

    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "params": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2, 4],       # regularization cho lá
            "model__max_features": ["sqrt", "log2", 0.5] # hyperparameter quan trọng nhất của RF
        }
    },

    "KNN": {
        "estimator": KNeighborsClassifier(),
        "params": {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15], # thêm 15 để test vùng smooth hơn
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"]  # ảnh hưởng lớn đến kết quả
        }
    },

    "SVM": {
        "estimator": SVC(probability=True, random_state=42),
        # Tách 2 grid vì linear kernel không dùng gamma
        "params": [
            {
                # Grid 1: Linear kernel — chỉ cần tune C
                "model__C": [0.1, 1, 10, 100],
                "model__kernel": ["linear"]
            },
            {
                # Grid 2: RBF kernel — cần tune cả C lẫn gamma
                # Thêm giá trị gamma cụ thể vì scale/auto không phải lúc nào cũng tối ưu
                "model__C": [0.1, 1, 10, 100],
                "model__kernel": ["rbf"],
                "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1]
            }
        ]
    }
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ============================================================
# Training & evaluation
# ============================================================
best_estimators = {}
results = []
roc_data = {}
clf_reports = {}

print("Starting model training and hyperparameter tuning using GridSearchCV...")

for name, config in models.items():
    print(f"\nTuning {name}...")
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", config["estimator"])
    ])

    # We optimize for F1-score to balance precision and recall
    grid = GridSearchCV(
        pipe,
        config["params"],
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_
    best_estimators[name] = best_model

    # Predict on test set
    y_pred = best_model.predict(x_test)
    y_prob = best_model.predict_proba(x_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "Best Params": grid.best_params_
    })

    roc_data[name] = {
        "fpr": roc_curve(y_test, y_prob)[0],
        "tpr": roc_curve(y_test, y_prob)[1],
        "auc": auc
    }

    # Classification report cho từng model
    clf_reports[name] = classification_report(
        y_test, y_pred, target_names=["No Disease", "Disease"]
    )

    print(f"Best Params for {name}: {grid.best_params_}")
    print(f"Test Recall: {rec:.4f} | Test F1: {f1:.4f} | Test Accuracy: {acc:.4f}")

# ============================================================
# Results comparison
# ============================================================
df_results = pd.DataFrame(results)
print("\n" + "="*80)
print("                               MODEL COMPARISON")
print("="*80)
print(df_results.to_string(index=False, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]))
print("="*80)

# Identify the best model based on F1-Score
best_row = df_results.loc[df_results["F1-Score"].idxmax()]
best_model_name = best_row["Model"]
best_estimator = best_estimators[best_model_name]

print(f"\nOverall Best Model based on F1-Score: {best_model_name} (F1-Score: {best_row['F1-Score']:.4f}, Recall: {best_row['Recall']:.4f})")

# ============================================================
# Save best model → models/
# ============================================================
model_path = os.path.join(MODELS_DIR, "best_model.pkl")
joblib.dump(best_estimator, model_path)
print(f"\nSaved best model ({best_model_name}) to: {model_path}")

# ============================================================
# Save reports → reports/
# ============================================================

# 1. CSV — bảng so sánh metrics
csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
df_results.to_csv(csv_path, index=False)
print(f"Saved comparison table to: {csv_path}")

# 2. TXT — classification report chi tiết cho từng model
txt_path = os.path.join(REPORTS_DIR, "classification_reports.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("CLASSIFICATION REPORTS — HEART DISEASE PREDICTION\n")
    f.write("=" * 70 + "\n\n")
    for name, report in clf_reports.items():
        clean_params = {k.replace("model__", ""): v
                        for k, v in results[[r["Model"] for r in results].index(name)]["Best Params"].items()}
        f.write(f"--- {name} ---\n")
        f.write(f"Best Params: {clean_params}\n\n")
        f.write(report + "\n\n")
    f.write("=" * 70 + "\n")
    f.write(f"Best Model: {best_model_name} "
            f"(F1-Score: {best_row['F1-Score']:.4f}, Recall: {best_row['Recall']:.4f})\n")
    f.write("=" * 70 + "\n")
print(f"Saved classification reports to: {txt_path}")

# 3. PNG — ROC Curves
plt.figure(figsize=(10, 8))
for name, data in roc_data.items():
    plt.plot(data["fpr"], data["tpr"], label=f"{name} (AUC = {data['auc']:.4f})", lw=2)
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.50)")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves Comparison", fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
roc_path = os.path.join(REPORTS_DIR, "roc_curves.png")
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved ROC curves plot to: {roc_path}")

# 4. PNG — Confusion Matrices + Bar Chart
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (name, model) in enumerate(best_estimators.items()):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)

    # Custom display labels matching heart disease target (0: No disease, 1: Disease)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=axes[idx], cmap=plt.cm.Blues, colorbar=False, values_format='d')

    # Clearer display parameters
    axes[idx].set_title(f"{name}", fontsize=14, fontweight='bold', pad=10)
    # Simplify param display for title
    clean_params = {k.replace("model__", ""): v for k, v in results[idx]["Best Params"].items()}
    axes[idx].set_xlabel(f"Predicted Label\nParams: {clean_params}", fontsize=10)

# 6th Subplot: F1-Score & Recall Comparison Bar Plot
ax_bar = axes[5]
x = np.arange(len(results))
width = 0.35

rects1 = ax_bar.bar(x - width/2, [r["F1-Score"] for r in results], width, label='F1-Score', color='#1f77b4')
rects2 = ax_bar.bar(x + width/2, [r["Recall"] for r in results], width, label='Recall', color='#ff7f0e')

ax_bar.set_ylabel('Score', fontsize=12)
ax_bar.set_title('F1-Score & Recall Comparison', fontsize=14, fontweight='bold', pad=10)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([r["Model"] for r in results], rotation=15, ha='right', fontsize=9)
ax_bar.set_ylim(0, 1.1)
ax_bar.legend(loc='lower left')
ax_bar.grid(True, axis='y', linestyle='--', alpha=0.6)

# Add value labels on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax_bar.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
cm_path = os.path.join(REPORTS_DIR, "confusion_matrices.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved confusion matrices to: {cm_path}")

print(f"\n{'='*80}")
print(f"All reports saved to: {REPORTS_DIR}")
print(f"Best model saved to:  {model_path}")
print(f"{'='*80}")
