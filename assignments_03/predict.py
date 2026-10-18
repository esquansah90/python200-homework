
# Part 2: Task 7 - Load and Predict

import json
import joblib
import pandas as pd
from pathlib import Path


model_path = Path("models/weather_classifier.pkl")
metadata_path = Path("models/weather_classifier_metadata.json")

if not model_path.exists() or not metadata_path.exists():
    raise FileNotFoundError(
        "Model files are missing. Run train_classifier.py first."
    )

model = joblib.load(model_path)

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)


print("Task 7: Weather Classifier Predictions")
print("City:", metadata["city"])
print("Features:", metadata["features"])
print(f"Test Accuracy: {metadata['test_accuracy']:.2%}")

# Four hypothetical weather days
# Values: max temp (C), min temp (C), rain (mm), wind (km/h)
days = [
    [20, 10, 0, 12],    # Clearly good day
    [35, 25, 15, 45],   # Clearly bad day
    [26, 10, 2.9, 29],  # Borderline good day
    [27, 12, 3.1, 31],  # Borderline bad day
]


new_weather = pd.DataFrame(
    days,
    columns=metadata["features"]
)


predictions = model.predict(new_weather)
probabilities = model.predict_proba(new_weather)


good_class_index = list(model.classes_).index(1)

print("\nPredictions for hypothetical weather days:")

for i, (prediction, probability) in enumerate(
    zip(predictions, probabilities), start=1
):
    label = "good" if prediction == 1 else "skip"
    good_probability = probability[good_class_index]

    print(f"\nDay {i}:")
    for feature, value in new_weather.iloc[i - 1].items():
        print(f"  {feature}: {value}")

    print(f"  Prediction: {label}")
    print(f"  Probability of good: {good_probability:.2f}")


# Task 8: Reflection

# Question 1: How confident is the model about borderline days?
#
# For Day 3, the model predicted "skip" with a 0.00
# probability of "good." However, Day 3 actually
# satisfies all the rules for a good running day.
# This shows that the model can make mistakes near
# the decision boundaries.
#
# My KNN model uses k=1, so its probability estimates
# are either 0.00 or 1.00. They do not provide a
# reliable measure of uncertainty.
#
# If a model gave a borderline day a probability
# of 0.52, I would consider that uncertain rather
# than confidently recommending a run. I could
# suggest checking the weather again before going.

# Question 2: What happens if predict.py runs before
# train_classifier.py?
#
# If predict.py runs before training, the saved model
# and metadata files may not exist. The program
# cannot load a model that has not been created.
#
# I added a file check that raises FileNotFoundError
# with the message:
# "Model files are missing. Run train_classifier.py first."
#
# This helps the user understand what went wrong
# and how to fix the problem.

# Question 3: What would change for a Week 10
# automated cloud pipeline?
#
# I would replace the hardcoded weather examples
# with new weather data from an API or cloud storage.
# I would also store the model in a cloud-accessible
# location and schedule predictions automatically.
#
# I would add logging and error handling so that
# problems with missing data or model files could
# be detected. The pipeline could then provide
# updated running recommendations without someone
# manually running the script.
