
# Week 7 Mini-Project

import os
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, tool

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key not found.")

print("API key loaded successfully.")

DATA_DIR = Path("happiness_project")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

df = None

print("Project setup complete.")



csv_files = sorted(DATA_DIR.glob("world_happiness_*.csv"))

print(f"Found {len(csv_files)} happiness CSV files.")

for file in csv_files:
    print(file.name)


# Task 1 - Tool 1

COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",
    "GDP per capita": "gdp_per_capita",
}


@tool
def load_happiness_data() -> dict:
    """Load and combine the World Happiness CSV files from 2015 to 2024.

    Returns:
        A dictionary containing the combined dataset's shape
        and column names, or an error if no files are found.
    """
    global df

    frames = []

    for path in sorted(DATA_DIR.glob("world_happiness_*.csv")):
        year_df = pd.read_csv(
            path,
            sep=";",
            decimal=","
        )

        year_df = year_df.rename(columns=COLUMN_NAMES)

        year_df["year"] = int(path.stem.split("_")[-1])

        frames.append(year_df)

    if not frames:
        return {"error": "No happiness CSV files found."}

    df = pd.concat(frames, ignore_index=True)

    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }


print("\n--- Task 1: Load Happiness Data ---")
result = load_happiness_data()
print(result)


# Task 1 - Tool 2

@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a column in the happiness dataset.

    Args:
        column: Name of the column to summarize.

    Returns:
        A dictionary containing summary statistics,
        or an error if the data or column is unavailable.
    """
    global df

    if df is None:
        return {"error": "No data loaded. Load the happiness data first."}

    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist."}

    return df[column].describe().to_dict()


print("\n--- Task 1: Summarize Column ---")
result = summarize_column("happiness_score")
print(result)


# Task 1 - Tool 3

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Calculate the Pearson correlation between two numeric columns.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        A dictionary containing the column names, Pearson
        correlation coefficient, and p-value, or an error.
    """
    global df

    if df is None:
        return {"error": "No data loaded. Load the happiness data first."}

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both columns do not exist."}

    if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
        return {"error": "Both columns must be numeric."}

    data = df[[col1, col2]].dropna()

    if len(data) < 2:
        return {"error": "Not enough valid data to calculate correlation."}

    if data[col1].nunique() < 2 or data[col2].nunique() < 2:
        return {"error": "Both columns must contain varying values."}

    r, p = pearsonr(data[col1], data[col2])

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4)
    }


print("\n--- Task 1: Compute Correlation ---")

result = compute_correlation(
    "gdp_per_capita",
    "happiness_score"
)

print(result)


# Task 1 - Tool 4

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Find the top N countries for a column in a specific year.

    Args:
        column: Numeric column used to rank countries.
        year: Year to filter the happiness dataset.
        n: Number of top countries to return.

    Returns:
        A dictionary containing the top countries and their values,
        or an error message for invalid input.
    """
    global df

    if df is None:
        return {"error": "No data loaded. Load the happiness data first."}

    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist."}

    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": "The ranking column must be numeric."}

    if n < 1:
        return {"error": "n must be at least 1."}

    year_data = df[df["year"] == year]

    if year_data.empty:
        return {"error": f"No data found for year {year}."}

    top_countries = (
        year_data
        .sort_values(by=column, ascending=False)
        .head(n)
    )

    results = top_countries[
        ["country", column]
    ].to_dict(orient="records")

    return {
        "year": year,
        "column": column,
        "top_countries": results
    }


print("\n--- Task 1: Top Countries ---")

result = get_top_n_countries(
    column="happiness_score",
    year=2020,
    n=5
)

print(result)


# Task 2

model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini"
)

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.

Use the available tools for loading data, summarizing columns,
computing correlations, and ranking countries.

Write Python code directly only when the tools are not sufficient,
such as when creating custom plots or calculations.

When creating plots:
- Save them in the outputs folder.
- Never use plt.show().
- Close plots after saving them.

Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[
        load_happiness_data,
        summarize_column,
        compute_correlation,
        get_top_n_countries
    ],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=[
        "pandas",
        "matplotlib.pyplot",
        "scipy.stats"
    ],
    max_steps=8
)

print("\n--- Task 2: CodeAgent Created ---")
print("World Happiness CodeAgent is ready!")


# Task 3

# Query 1
query_1 = "Load the happiness data and tell me its shape and column names."

print("\n--- Task 3: Query 1 ---")

response_1 = agent.run(query_1, reset=False)

print("\nAgent Response:")
print(response_1)


# Task 3 - Query 2

query_2 = "Summarize the happiness_score column."

print("\n--- Task 3: Query 2 ---")

response_2 = agent.run(query_2, reset=False)

print("\nAgent Response:")
print(response_2)


# Task 3 - Query 3

query_3 = (
    "What is the correlation between gdp_per_capita "
    "and happiness_score? Is it statistically significant?"
)

print("\n--- Task 3: Query 3 ---")

response_3 = agent.run(query_3, reset=False)

print("\nAgent Response:")
print(response_3)


# Task 3 - Query 4

query_4 = "Show me the top 5 happiest countries in 2020."

print("\n--- Task 3: Query 4 ---")

response_4 = agent.run(query_4, reset=False)

print("\nAgent Response:")
print(response_4)


# Task 3 - Query 5


query_5 = (
    "Using the pandas DataFrame named df, plot the average "
    "happiness_score over the years as a line chart, "
    "with one line per region. Group the real dataset by "
    "year and region. Do not use placeholder or sample data. "
    "Save the plot to outputs/happiness_by_region.png."
)


print("\n--- Task 3: Query 5 ---")

response_5 = agent.run(
    query_5,
    reset=False,
    additional_args={"df": df}
)

print("\nAgent Response:")
print(response_5)

plot_path = OUTPUT_DIR / "happiness_by_region.png"

if plot_path.exists():
    print(f"Plot successfully saved to: {plot_path}")
else:
    print("Plot was not found. Check the agent execution logs.")


# Task 4

# Question 1
question_1 = "Which 5 countries had the highest GDP per capita in 2024?"

print("\n--- Task 4: Question 1 ---")

answer_1 = agent.run(question_1, reset=False)

print("\nAgent Response:")
print(answer_1)

# Tool usage: The agent should use get_top_n_countries().


# Task 4 - Question 2

question_2 = (
    "Using the pandas DataFrame named df, calculate the average "
    "happiness_score for 2015 and 2024. Show both averages and "
    "calculate how much the average changed between those years. "
    "Use the real dataset and write Python code for the calculation."
)

print("\n--- Task 4: Question 2 ---")

answer_2 = agent.run(
    question_2,
    reset=False,
    additional_args={"df": df}
)

print("\nAgent Response:")
print(answer_2)

# Code generation: The agent should write pandas code to
# calculate and compare average happiness scores for two years.


# --- Task 5: Reflection ---
#
# 1. The agent reported a Pearson correlation of 0.6313
#    between GDP per capita and happiness score, with a
#    statistically significant p-value. The result shows
#    a positive relationship. The agent interpreted the
#    significance correctly, but could have explained
#    the strength and direction more clearly.
#
# 2. I was surprised that the agent initially struggled
#    with the data-loading tool because it expected a
#    DataFrame instead of a dictionary. However, it
#    corrected itself. I was also impressed that it
#    generated Python code to create a regional line chart.
#
# 3. An additional tool to compare happiness scores
#    between two years would make the agent more useful.
#    It could calculate the average score for each year
#    and show whether happiness increased or decreased.
