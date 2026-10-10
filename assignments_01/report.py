import json
from pathlib import Path

from weatherkit import WeatherResponse
from weatherkit.records import to_readings
from weatherkit.summarize import DailyAggregator


def main() -> None:
    """Load, validate, summarize, and report the weather data."""
    json_path = Path(__file__).parent / "weather_raw.json"

    with open(json_path, "r") as file:
        raw_data = json.load(file)

    weather = WeatherResponse.model_validate(raw_data)

    readings = to_readings(weather)

    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)
    incomplete = aggregator.incomplete_days(readings)

    print(
        f"{'Date':<12}"
        f"{'High':>8}"
        f"{'Low':>8}"
        f"{'Precip':>10}"
        f"{'Range':>10}"
    )
    print("-" * 48)

    for summary in summaries:
        print(
            f"{summary.date:<12}"
            f"{summary.temp_max:>8.1f}"
            f"{summary.temp_min:>8.1f}"
            f"{summary.precipitation_sum:>10.1f}"
            f"{summary.temp_range():>10.1f}"
        )

    if incomplete:
        print("\nWarning: Incomplete days dropped:", ", ".join(incomplete))
    else:
        print("\nWarning: No incomplete days were dropped.")


# Without this guard, main() would run automatically if another Python file
# imported report.py. The guard makes main() run only when report.py is
# executed directly, so its functions can be imported without printing
# the report automatically.

if __name__ == "__main__":
    main()


# Reflection Questions
#
# 1. WeatherResponse currently rejects the entire file if one temperature
# is null. This can be useful when the pipeline requires complete and
# accurate data, because it prevents incomplete data from being processed.
# However, if a weather sensor temporarily misses one reading, I might
# rather keep the rest of the valid data instead of rejecting the whole file.
# To tolerate missing temperatures, I could change the schema from
# list[float] to list[float | None]. The pipeline would then need to decide
# how to handle or skip the missing values.
#
# 2. DailyAggregator uses a default min_hours of 24. If the pipeline runs
# at noon, the current day may only have about 12 hourly readings. The
# aggregator could treat that day as incomplete and leave it out even though
# nothing is actually wrong with the data; the day simply is not finished.
# incomplete_days() helps by identifying which dates were left out because
# they did not have enough observations. This makes it easier to tell that
# data is missing or that a day may not be complete yet.
#
# 3. Keeping weatherkit as a package makes the code easier to reuse in
# another pipeline. In Week 10, the pipeline can import the classes and
# functions it needs from weatherkit instead of copying code out of one
# large file. For example, it can import WeatherResponse or DailyAggregator
# and use them directly.