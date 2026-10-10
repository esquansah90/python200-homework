from dataclasses import dataclass, FrozenInstanceError, field
from pydantic import BaseModel, Field, ValidationError, model_validator
import pytest

# --- Classes ---

# Q1

class Thermometer:
    def __init__(self, location, readings=None):
        self.location = location
        self.readings = readings if readings is not None else []

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)

    # Q2
    def __repr__(self):
        return (
            f"Thermometer(location='{self.location}', "
            f"n_readings={len(self.readings)}, "
            f"average={self.average()})"
        )


# Q1 - Create and test a Thermometer

my_thermometer = Thermometer("Atlanta")

my_thermometer.add(25)
my_thermometer.add(28)
my_thermometer.add(30)
my_thermometer.add(27)

print("Average:", my_thermometer.average())
print("Hottest:", my_thermometer.hottest())

# average() needs to check if the readings list is empty because
# there would be no values to calculate an average from.
# Without the check, Python would try to divide by zero and cause an error.


# Q2 - Test __repr__

print(my_thermometer)

second_thermometer = Thermometer("Miami", [30, 31, 32, 33])

print([my_thermometer, second_thermometer])

# Without __repr__, Python displays the object's class and memory location.
# This is not very helpful when debugging because it does not show
# useful information about what is stored inside the object.


# Q3

class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        return [
            reading
            for reading in thermometer.readings
            if reading > self.threshold
        ]


alert_one = TemperatureAlert(26.0)
alert_two = TemperatureAlert(29.0)

print("Above 26:", alert_one.breaches(my_thermometer))
print("Above 29:", alert_two.breaches(my_thermometer))

# The threshold is stored on TemperatureAlert so we can create an alert
# with a specific threshold and reuse it without entering the threshold again.
# If we had twenty thermometers, we could use the same alert to check all
# of them using the same threshold.


# --- Dataclasses, Type Hints, and Docstrings ---

# Q1

@dataclass(frozen=True)
class Station:
    """Represents a weather station and its location information."""

    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float


station_a = Station(
    "ATL001",
    "Atlanta Station",
    33.75,
    -84.39,
    320.0
)

station_b = Station(
    "ATL001",
    "Atlanta Station",
    33.75,
    -84.39,
    320.0
)

print("Stations equal:", station_a == station_b)

# The two Station objects are equal because dataclasses compare the values
# of their fields. With the original hand-written class, they would be
# considered different objects because it did not define __eq__.

# Q2

try:
    station_a.name = "New Station Name"
except FrozenInstanceError as e:
    print("Frozen error:", e)


station_c = Station(
    "MIA001",
    "Miami Station",
    25.76,
    -80.19,
    2.0
)

station_set = {station_a, station_b, station_c}

print("Number of unique stations:", len(station_set))

# frozen=True makes Station objects immutable and also makes them hashable.
# Being hashable allows Station objects to be stored in a set.
# Since station_a and station_b have identical values, the set treats
# them as duplicates, so the set contains only two unique stations.

# Q3

# Error from using stations: list[Station] = []:
# Traceback (most recent call last):
#  File "c:\Users\esqua\OneDrive\CTD-Repos\python200-homework\assignments_01\warmup_01.py", line 154, in <module>
#    @dataclass
#       ^^^^^^^^^
#  File "C:\Users\esqua\AppData\Local\Python\pythoncore-3.14-64\Lib\dataclasses.py", line 1450, in dataclass
#    return wrap(cls)
#  File "C:\Users\esqua\AppData\Local\Python\pythoncore-3.14-64\Lib\dataclasses.py", line 1440, in wrap
#    return _process_class(cls, init, repr, eq, order, unsafe_hash,
#                          frozen, match_args, kw_only, slots,
#                          weakref_slot)
#  File "C:\Users\esqua\AppData\Local\Python\pythoncore-3.14-64\Lib\dataclasses.py", line 1066, in _process_class
#    cls_fields.append(_get_field(cls, name, type, kw_only))
#                      ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
#  File "C:\Users\esqua\AppData\Local\Python\pythoncore-3.14-64\Lib\dataclasses.py", line 917, in _get_field
#    raise ValueError(f'mutable default {type(f.default)} for field '
#                     f'{f.name} is not allowed: use default_factory')
#ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory
#(python200-homework) 
#
# Python refuses [] as the default because a list is mutable.
# If the same list were shared between StationBatch objects, changing the
# list in one object could affect another object. default_factory=list
# creates a new empty list for each StationBatch object.

@dataclass
class StationBatch:
    """Represents a group of weather stations in a region."""

    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Add a station to the batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the highest elevation, or None if empty."""
        if not self.stations:
            return None

        return max(self.stations, key=lambda station: station.elevation)

batch = StationBatch("Southeast")

batch.add(station_a)
batch.add(station_b)
batch.add(station_c)

print("Stations in batch:", batch.stations)
print("Highest station:", batch.highest())

# --- Pydantic ---

# Q1

class Reading(BaseModel):
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def check_failed_sensor(self):
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("Possible failed sensor")
        return self
    


valid_reading = Reading(
    station_id="ATL001",
    timestamp="2026-10-03T12:00:00",
    temperature_c=25.5,
    humidity=65.0
)

print("Valid reading:", valid_reading)

# Q2

# 1. Missing required field
try:
    Reading(
        station_id="ATL001",
        temperature_c=25.0,
        humidity=60.0
    )
except ValidationError as e:
    print("Missing field error:")
    print(e)


# 2. Temperature is too high
try:
    Reading(
        station_id="ATL001",
        timestamp="2026-10-03T12:00:00",
        temperature_c=150.0,
        humidity=60.0
    )
except ValidationError as e:
    print("Temperature error:")
    print(e)


# 3. Humidity cannot be converted to a number
try:
    Reading(
        station_id="ATL001",
        timestamp="2026-10-03T12:00:00",
        temperature_c=25.0,
        humidity="very humid"
    )
except ValidationError as e:
    print("Humidity error:")
    print(e)


# Test Pydantic type conversion
converted_reading = Reading(
    station_id="ATL001",
    timestamp="2026-10-03T12:00:00",
    temperature_c="21.5",
    humidity=40
)

print("Converted reading:", converted_reading)
print("Temperature type:", type(converted_reading.temperature_c))
print("Humidity type:", type(converted_reading.humidity))

# Pydantic accepts "21.5" because the string can be converted into a float.
# It rejects "very humid" because that string cannot be converted into a float.
# In other words, Pydantic can convert compatible values to the required type,
# but it raises a validation error when the value cannot be converted.

# Q3

try:
    Reading(
        station_id="A",
        temperature_c="not a number",
        humidity=50.0
    )
except ValidationError as e:
    for error in e.errors():
        print("Location:", error["loc"])
        print("Message:", error["msg"])


# Three errors were reported: the station_id is too short,
# the timestamp is missing, and temperature_c is not a valid number.
# Reporting all the errors at once is more useful because we can see
# everything that needs to be fixed instead of fixing one error at a time.

# Q4

valid_sensor_reading = Reading(
    station_id="ATL001",
    timestamp="2026-10-03T12:00:00",
    temperature_c=20.0,
    humidity=50.0
)

print("Valid sensor reading:", valid_sensor_reading)


try:
    bad_sensor_reading = Reading(
        station_id="ATL001",
        timestamp="2026-10-03T12:00:00",
        temperature_c=-50.0,
        humidity=0.0
    )
except ValidationError as e:
    print("Failed sensor error:")
    print(e)


# This rule cannot be expressed with Field constraints alone because
# it depends on the values of two fields together: humidity and temperature_c.
# Field constraints are used to validate individual fields.

# --- pytest ---

# Q1

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32


def test_celsius_to_fahrenheit():
    """Test Celsius to Fahrenheit conversions."""
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

    # pytest.approx is used because floating-point calculations can have
    # tiny rounding differences, so an exact == comparison may fail.


# Q2

def mean(values: list[float]) -> float:
    """Return the mean of a list of values."""
    if not values:
        raise ValueError("values cannot be empty")

    return sum(values) / len(values)


def test_mean_of_empty_raises():
    """Test that mean raises ValueError for an empty list."""
    with pytest.raises(ValueError, match="empty"):
        mean([])

    # pytest.raises(ValueError) only checks that a ValueError happened.
    # Using match= also checks that the error message contains the expected
    # word, which helps confirm that the error happened for the right reason.

# Q3

@pytest.mark.parametrize(
    "values, expected",
    [
        ([5.0], 5.0),
        ([10.0, 20.0, 30.0], 20.0),
        ([-5.0, -10.0, -15.0], -10.0),
        ([-10.0, 10.0], 0.0),
    ],
)
def test_mean_values(values, expected):
    """Test mean with several different lists of values."""
    assert mean(values) == pytest.approx(expected)

# 6 passed in 0.18s

# One parametrized test is better than four nearly identical test functions
# because it avoids repeated code and makes the tests easier to read and maintain.

# Q4

# Failure output after changing 9 / 5 to 9 / 4:
# assert celsius_to_fahrenheit(100) == 212
# E       assert 257.0 == 212
# E        +  where 257.0 = celsius_to_fahrenheit(100)
# FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 257.0 == 212
# 1 failed, 5 passed in 0.23s

# Pytest showed that when the input was 100, the function returned 257.0,
# but the expected value was 212. This is more useful than only saying
# "assertion failed" because it shows the actual value and expected value,
# which makes it easier to find what went wrong.