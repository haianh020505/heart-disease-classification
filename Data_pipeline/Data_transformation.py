import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

csv_path = os.path.join(os.path.dirname(__file__), "heart.csv")
data = pd.read_csv(csv_path)

# Remove duplicate rows to prevent data leakage between train and test sets
# Original dataset has 1025 rows but only 302 unique rows (723 duplicates!)
n_before = len(data)
data = data.drop_duplicates().reset_index(drop=True)
n_after = len(data)
print(f"[Data] Removed {n_before - n_after} duplicate rows: {n_before} -> {n_after}")

target = "target"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_transformer = Pipeline(steps = [
    ("imputer", SimpleImputer(missing_values=-1, strategy="median")),
    ("scaler", StandardScaler())
])

gender                  = sorted(x_train["sex"].unique().tolist())
fasting_blood_sugar     = sorted(x_train["fbs"].unique().tolist())
exercise_induced_angina = sorted(x_train["exang"].unique().tolist())
slope                   = sorted(x_train["slope"].unique().tolist())
number_of_major_vessels = sorted(x_train["ca"].unique().tolist())

ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[
        gender,
        fasting_blood_sugar,
        exercise_induced_angina,
        slope,
        number_of_major_vessels
    ]))
])
nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["age", "restbps", "chol", "thalach", "oldpeak"]),
    ("ord_feature", ord_transformer, ["sex", "fbs", "exang", "slope", "ca"]),
    ("nom_feature", nom_transformer, ["chest_pain","restecg", "thal"]),
])

