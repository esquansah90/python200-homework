
"""Use the saved model to predict running weather."""

from weather_model import WeatherClassifier


SAMPLE_DAYS = [
    {
        "temperature_2m_max": 18,
        "temperature_2m_min": 10,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 12,
    },
    {
        "temperature_2m_max": 2,
        "temperature_2m_min": -5,
        "precipitation_sum": 12.0,
        "wind_speed_10m_max": 40,
    },
    {
        "temperature_2m_max": 22,
        "temperature_2m_min": 13,
        "precipitation_sum": 4.0,
        "wind_speed_10m_max": 18,
    },
    {
        "temperature_2m_max": 25,
        "temperature_2m_min": 15,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 33,
    },
]


def main() -> None:
    """Load the model and print predictions for sample days."""

    classifier = WeatherClassifier(
        "models/weather_classifier.pkl"
    )

    results = classifier.predict(SAMPLE_DAYS)

    for day, result in zip(SAMPLE_DAYS, results):
        print(
            f"High: {day['temperature_2m_max']}C, "
            f"Low: {day['temperature_2m_min']}C, "
            f"Rain: {day['precipitation_sum']}mm, "
            f"Wind: {day['wind_speed_10m_max']} km/h "
            f"-> {result.label} "
            f"(Probability of good: {result.probability:.2f})"
        )


# Without this guard, importing this script into another
# Python file would automatically run main() if it were
# called directly at the top level.
if __name__ == "__main__":
    main()


# Reflection

# 1. Selecting features in a fixed order prevents the model
# from mixing up weather values. For example, if wind speed
# and temperature were switched, the model could make
# incorrect predictions.

# 2. Training a small model in the test fixture makes the
# tests independent of the real saved model. This makes
# testing more reliable and easier to repeat.

# 3. My component is ready for Week 10 because it can
# process multiple weather records in one call. Before
# using it in production, I would add stronger input
# validation to catch invalid values and missing data.
