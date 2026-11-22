

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel, tool

load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    raise ValueError("OpenAI API key not found.")

client = OpenAI()
print("OpenAI client created.")
# Q1

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


celsius_tool_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a temperature from Celsius to Fahrenheit.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "The temperature in Celsius."
                }
            },
            "required": ["celsius"]
        }
    }
}


print("\n--- Q1 ---")
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


# Q2

def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }
]


def run_agent(user_prompt: str) -> str:
    """Run a minimal ReAct-style agent for a single user prompt."""

    SYSTEM_PROMPT = """You are a helpful assistant.
    Only use a tool when it is available and relevant to the question.
    Use get_current_time only for questions about the current time.
    Answer directly if no relevant tool is available."""

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    
    if first_message.tool_calls:
        print("Agentic mode engaged...")

        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name

            if function_name == 'get_current_time':
                tool_result = get_current_time()

            elif function_name == 'celsius_to_fahrenheit':
                arguments = json.loads(tool_call.function.arguments)
                tool_result = celsius_to_fahrenheit(arguments['celsius'])

            else:
                tool_result = f'Error: unknown tool {function_name}.'

            print('Tool called:', function_name)
            print('Tool result:', tool_result)

        
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )

        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''

    else:
        print("No tools needed....")

    return first_message.content or ''


# Prediction:
# I predict that the agent will not call a tool because
# get_current_time is not needed to convert temperatures.
# I expect only one API call because the model can
# calculate the conversion on its own.

print("\n--- Q2 ---")

answer = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print(answer)

# My initial prediction was correct. The agent answered
# directly without using a tool and made only one API call.
# After changing the system prompt, the agent incorrectly
# called the time tool. This shows how system prompts
# can influence tool selection.

# Q3

tools.append(celsius_tool_schema)

print("\n--- Q3 ---")

response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)

# The agent called celsius_to_fahrenheit because the question
# required a temperature conversion. It made two API calls
# and returned 98.6 degrees Fahrenheit.


response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)

# The agent did not call a tool because the boiling point
# of water is general knowledge. It answered directly
# using only one API call.


# --- Lesson 03: Data Analysis Tool Agent ---
# Q4

RESOURCES_DIR = Path("resources")


class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None


    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # Tools

    def list_csv_files(self):
        """List available CSV files in resources/."""
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """Load a CSV file and make it the active dataset."""
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """Return column names for the currently loaded CSV."""
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """Return basic summary statistics for one or more columns."""
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """Return summary statistics for a single column."""
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """Plot data from the active CSV."""
        error = self._ensure_loaded()
        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"

        if x == y:
            x = None

        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."

        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"

        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()

        return f"Plotted {y} vs {x} as a {plot_type}."


    def compute_correlation(self, col1: str, col2: str) -> dict:
        """Calculate Pearson correlation between two columns."""
        error = self._ensure_loaded()
        if error:
            return error

        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {"error": "One or both columns do not exist."}

        r, p = pearsonr(self.df[col1], self.df[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
        }


print("\n--- Q4 ---")
print("Class defined")

csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation,
}


tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {
                        "type": "string",
                        "description": "Column name for y-axis."
                    },
                    "x": {
                        "type": "string",
                        "description": "Optional column name for x-axis."
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Calculate Pearson correlation and p-value between two numeric columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "Name of the first column."
                    },
                    "col2": {
                        "type": "string",
                        "description": "Name of the second column."
                    },
                },
                "required": ["col1", "col2"],
            },
        },
    },
]


def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """Run one ReAct cycle using a tool-based agent."""

    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """Convert a tool result into a conversation message."""
        content = json.dumps(result, default=str) if not isinstance(result, str) else result

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

        return tool_message

    for loop_idx in range(max_tool_rounds):

        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        assistant_entry = {
            "role": "assistant",
            "content": msg.content
        }

        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                tc.model_dump() for tc in msg.tool_calls
            ]

        messages.append(assistant_entry)

        if not msg.tool_calls:
            return msg.content


        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)

            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {
                        "error": f"Tool '{name}' failed: {type(e).__name__}: {e}"
                    }

            messages.append(
                observe_tool_result(tool_call.id, result)
            )

    return "I hit the tool-round limit. Try a simpler request."

# Q5

SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]

print("\n--- Q5 ---")

result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh."
)

print(result)

# Q6

print("\n--- Q6 ---")

# system: Provides instructions that guide the agent's behavior.
# user: Contains the question or task given to the agent.
# assistant: Contains the model's tool requests or final response.
# tool: Contains the results returned by the executed Python tools.

print(json.dumps(messages, indent=2, default=str))

# --- Lesson 04: smolagents ---
# Q7

print("\n--- Q7 ---")

csv_manager = csv_backend


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Calculate the Pearson correlation between two numeric CSV columns.

    Args:
        col1: Name of the first column.
        col2: Name of the second column.

    Returns:
        A dictionary containing the column names, Pearson correlation
        coefficient, and p-value, or an error message.
    """
    return csv_manager.compute_correlation(col1, col2)


print(compute_correlation.description)

# In Q4, I manually wrote a JSON schema describing the function,
# its parameters, and their types.
# Smolagents automatically generates this information using the
# function name, type hints, and docstring.
# The developer must provide clear descriptions, type hints,
# and explanations of the arguments for the tool to work well.


# Q8

print("\n--- Q8 ---")

@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dictionary containing the available CSV filenames.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/.

    Args:
        filename: Name of the CSV file to load.

    Returns:
        A dictionary with the loading result and column names.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Get the column names of the loaded CSV.

    Returns:
        A list of column names or an error dictionary.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Summarize the selected columns or all columns.

    Args:
        columns: Optional list of columns to summarize.

    Returns:
        A dictionary of summary statistics.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column using summary statistics.

    Args:
        column: Name of the column to describe.

    Returns:
        A dictionary of statistics or an error.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot columns from the loaded CSV.

    Args:
        y: Column name for the y-axis.
        x: Optional column name for the x-axis.
        plot_type: Type of plot, either line or scatter.

    Returns:
        A success message or an error.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)


TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

print("All 7 smolagents tools registered successfully.")


model = OpenAIServerModel(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_id="gpt-4o-mini"
)

tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
    instructions=(
        "You are a CSV data assistant. "
        "Use the available tools to analyze data. "
        "Do not guess results."
    ),
    max_steps=8
)


CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

Rules:
- Prefer provided tools for simple tasks.
- For plot styling such as colors, write matplotlib code directly.
- Never call plt.show() because it can cause threading errors.
- Instead, save plots using plt.savefig() and close them with plt.close().
- Save plots in the current directory.
- Use csv_manager.df to access the loaded dataset.
- Only claim success if the requested task was completed.
"""

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=[
        "pandas",
        "matplotlib.pyplot",
        "numpy"
    ],
    max_steps=8
)

print("Both agents created successfully.")


prompt = (
    "Load bike_commute.csv. "
    "Plot avg_heart_rate vs duration_min "
    "as a scatter plot with green dots."
)

print("\n--- ToolCallingAgent ---")
response_tool = tool_agent.run(prompt)
print("ToolCallingAgent response:", response_tool)

print("\n--- CodeAgent ---")
response_code = code_agent.run(
    prompt,
    additional_args={"csv_manager": csv_manager}
)
print("CodeAgent response:", response_code)

# Q8 Reflection:
# ToolCallingAgent created a scatter plot with blue dots instead of green.
# It was limited by the predefined plot_data tool.
# CodeAgent generated custom matplotlib code and created green dots.
# However, it encountered threading and timeout errors when displaying plots.
# ToolCallingAgent is useful for controlled, predefined tasks.
# CodeAgent is useful for flexible tasks requiring custom Python code.

# Q9 Reflection:
# ToolCallingAgent is better when tasks can be completed using
# predefined tools because it is more controlled and predictable.
#
# CodeAgent is better when tasks require custom calculations,
# visualizations, or functionality not available in existing tools.
#
# One major risk of CodeAgent is that it executes AI-generated code.
# This code could contain errors, access unintended resources,
# or perform unsafe operations. Therefore, code execution should
# be restricted and monitored.