import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- Preprocessing ---
# Q1

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Q1: Mean of each scaled training column:")
print(X_train_scaled.mean(axis=0))

# Fit the scaler on X_train only to prevent data leakage
# from the test data.


# Q2

encoder = OneHotEncoder(sparse_output=False)

season_encoded = encoder.fit_transform(df[["season"]])

print("\nQ2: Encoded season shape:")
print(season_encoded.shape)

print("Season category names:")
print(encoder.get_feature_names_out())

# The season column becomes 4 columns because the
# dataset contains four seasons: fall, spring, summer, winter.


# Q3

features = numeric + ["season"]

X_all = df[features]

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["season"])
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train_all, y_train_all)

y_pred = pipeline.predict(X_test_all)

accuracy = accuracy_score(y_test_all, y_pred)

print("\nQ3: Pipeline test accuracy:")
print(accuracy)

# The pipeline automatically scales numeric features,
# encodes season, and passes the processed data to
# Logistic Regression for training and prediction.


# --- KNN ---
# Q1

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

knn_pipeline.fit(X_train, y_train)

knn_predictions = knn_pipeline.predict(X_test)

print("\nKNN Q1: Test Accuracy:")
print(accuracy_score(y_test, knn_predictions))

print("\nKNN Q1: Classification Report:")
print(classification_report(y_test, knn_predictions))


# Q2

k_values = [1, 3, 5, 7, 9, 11, 15, 21]

cv_results = {}

print("\nKNN Q2: Mean 5-Fold Cross-Validation Accuracy:")

for k in k_values:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    mean_score = scores.mean()
    cv_results[k] = mean_score

    print(f"k={k}: {mean_score:.4f}")

best_k = max(cv_results, key=cv_results.get)

print(f"\nBest k based on CV accuracy: {best_k}")

# I would choose the k with the highest mean cross-validation
# accuracy because it performed best across the training folds.


# Q3

# KNN without scaling
knn_raw = KNeighborsClassifier(n_neighbors=5)

knn_raw.fit(X_train, y_train)

raw_predictions = knn_raw.predict(X_test)

raw_accuracy = accuracy_score(y_test, raw_predictions)

# KNN with scaling
knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

knn_scaled.fit(X_train, y_train)

scaled_predictions = knn_scaled.predict(X_test)

scaled_accuracy = accuracy_score(y_test, scaled_predictions)

print("\nKNN Q3: Accuracy Comparison:")
print(f"Without scaling: {raw_accuracy:.4f}")
print(f"With scaling:    {scaled_accuracy:.4f}")


# Yes, scaling helped. Accuracy increased from 0.9250
# to 0.9583. KNN uses distance to find the nearest
# neighbors, so features with larger numerical ranges
# can dominate the calculations. Scaling puts features
# on comparable scales.


# --- Logistic Regression ---
# Q1

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

lr_pipeline.fit(X_train, y_train)

lr_predictions = lr_pipeline.predict(X_test)

print("\nLogistic Regression Q1:")

print("Accuracy:", accuracy_score(y_test, lr_predictions))
print("Precision:", precision_score(y_test, lr_predictions))
print("Recall:", recall_score(y_test, lr_predictions))
print("F1 Score:", f1_score(y_test, lr_predictions))


# Q2

coefficients = lr_pipeline.named_steps["classifier"].coef_[0]

print("\nLogistic Regression Q2: Feature Coefficients")

for feature, coefficient in zip(numeric, coefficients):
    print(f"{feature}: {coefficient:.4f}")


# Wind speed (-2.4438) and precipitation (-1.8891)
# push most strongly toward "skip" because they have
# the largest negative coefficients.
# This matches my intuition because strong winds and
# heavy rain usually make running less enjoyable.



# Q3

probabilities = lr_pipeline.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Q3: First 5 Test Days")

for i, probability in enumerate(probabilities[:5], start=1):
    print(f"Day {i}: {probability:.2f}")


# A probability near 0.5 means the model is uncertain
# about whether the weather is good for running.
# For example, Day 2 has a probability of 0.47,
# so the model is not very confident in its prediction.


# --- Evaluation ---
# Q1

cm = confusion_matrix(y_test, lr_predictions)

print("\nEvaluation Q1: Confusion Matrix")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["skip", "good"]
)

disp.plot(cmap="Blues")

plt.title("Logistic Regression Confusion Matrix")
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.close()

print("Confusion matrix saved to outputs/logreg_confusion_matrix.png")


# The model made 9 false positives and 7 false negatives.
# False positives would matter more for a running app
# because the app might recommend running when the
# weather is actually bad or potentially unsafe.



# --- joblib ---
# Q1

joblib.dump(lr_pipeline, "models/warmup_model.pkl")

loaded = joblib.load("models/warmup_model.pkl")

assert (loaded.predict(X_test) == lr_pipeline.predict(X_test)).all()

print("\njoblib Q1: Predictions match.")

# Saving only the LogisticRegression step would leave
# out the StandardScaler. The model was trained on
# scaled data, so predicting with raw data could
# produce incorrect results.


