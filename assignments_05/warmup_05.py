
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
MODEL = "gpt-4o-mini"


def complete(messages, **kwargs):
    return client.chat.completions.create(model=MODEL, messages=messages, **kwargs)


def ask(prompt, **kwargs):
    return complete([{"role": "user", "content": prompt}], **kwargs)


# --- Completions API ---
# API Q1
response = ask("What is one thing that makes Python a good language for beginners?")
print("API Q1 response:", response.choices[0].message.content)
print("API Q1 model:", response.model)
print("API Q1 total tokens:", response.usage.total_tokens)

# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
for temperature in [0, 0.7, 1.5]:
    response = ask(prompt, temperature=temperature)
    print(f"API Q2 temperature={temperature}:", response.choices[0].message.content)
# Lower temperatures tend to be more consistent than higher ones.
# For the most reproducible response among these, use temperature 0.

# API Q3
response = ask(
    "Give me a one-sentence fun fact about pandas (the animal, not the library).",
    n=3,
    temperature=1.0,
)
for index, choice in enumerate(response.choices, start=1):
    print(f"API Q3 completion {index}:", choice.message.content)

# API Q4

response = ask("Explain how neural networks work in detail.", max_tokens=15)
print("API Q4 limited response:", response.choices[0].message.content)
print("API Q4 finish reason:", response.choices[0].finish_reason)
# The response was cut short because the token limit was 15.
# Token limits help control response length and API costs.

# --- System Messages and Personas ---
# System Q1
question = "I don't understand what a list comprehension is."
for name, persona in [
    ("Patient tutor", "You are a patient, encouraging Python tutor. Explain things simply and end with encouragement."),
    ("Pirate", "You are a playful pirate who explains Python with short nautical metaphors."),
]:
    response = complete([
        {"role": "system", "content": persona},
        {"role": "user", "content": question},
    ])
    print(f"System Q1 ({name}):", response.choices[0].message.content)
# The patient tutor explained the concept clearly with examples.
# The pirate used pirate language and a fun storytelling style.
# The system message changed how the AI responded.

# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"},
]
response = complete(messages)
print("System Q2:", response.choices[0].message.content)
# The AI knew Jordan's name because it was included
# in the conversation history sent with the request.

# --- Prompt Engineering ---
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow.",
]


def classify_reviews(examples=""):
    prompt = (
        "Classify each numbered review as positive, negative, or mixed. "
        "Return one line per review in the format 'Review 1: positive'.\n"
        f"{examples}\n" +
        "\n".join(f"Review {i}: {review}" for i, review in enumerate(reviews, 1))
    )
    return ask(prompt).choices[0].message.content


# Prompt Q1 — Zero-shot
print("Prompt Q1 zero-shot:\n", classify_reviews())

# Prompt Q2 — One-shot
one_example = 'Example:\nReview: "Fast shipping but the item arrived damaged."\nSentiment: mixed\n'
print("Prompt Q2 one-shot:\n", classify_reviews(one_example))
# Adding one example did not change the results or format.
# Both zero-shot and one-shot gave the same answers.
# However, the example made the expected format clearer.

# Prompt Q3 — Few-shot
three_examples = (
    'Examples:\nReview: "Great support and easy setup."\nSentiment: positive\n'
    'Review: "Nothing works and the app freezes."\nSentiment: negative\n'
    'Review: "Fast shipping but the item arrived damaged."\nSentiment: mixed\n'
)
print("Prompt Q3 few-shot:\n", classify_reviews(three_examples))
# All three approaches gave the same results.
# I would use zero-shot for simple tasks, 
# one-shot when one example is helpful, 
# and few-shot for more complex tasks that need several examples.

# Prompt Q4 — Chain of Thought
math_prompt = (
    "A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later "
    "takes a new job that pays $7,500 more per year than her post-raise salary. "
    "Show the calculation steps and clearly label the final annual salary."
)
print("Prompt Q4 calculation:\n", ask(math_prompt).choices[0].message.content)
# Asking the AI to solve the problem step by step
# helps it organize the calculations and makes
# mistakes easier to identify.

# Prompt Q5 — Structured output
review = ("I've been using this tool for three months. It handles large datasets well, "
          "but the UI is clunky and the export options are limited.")
json_prompt = (
    "Analyze the following review. Respond ONLY with valid JSON, no markdown. "
    'Use keys "sentiment" (positive, negative, or mixed), "confidence" '
    '(a number from 0 to 1), and "reason" (one sentence).\n'
    f"Review: {review}"
)
raw = ask(json_prompt).choices[0].message.content or ""
print("Prompt Q5 raw JSON:", raw)
try:
    data = json.loads(raw)
    print("Prompt Q5 sentiment:", data["sentiment"])
    print("Prompt Q5 confidence:", data["confidence"])
    print("Prompt Q5 reason:", data["reason"])
except (json.JSONDecodeError, KeyError, TypeError) as error:
    print("Prompt Q5 parsing failed:", error)
    print("Prompt Q5 raw response for debugging:", raw)

# Prompt Q6 — Delimiters
user_text = ("First boil a pot of water. Once boiling, add a handful of salt and the "
             "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice.")

def format_steps(text):
    prompt = f'''You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{text}```'''
    return ask(prompt).choices[0].message.content

print("Prompt Q6 instructions:", format_steps(user_text))
print("Prompt Q6 prose:", format_steps("The sky was cloudy and the park was quiet."))
# Delimiters help separate the user's text from instructions.
# This helps the AI understand what text it should process
# and reduces confusion.

# --- Local Models with Ollama ---
# Ollama Q1
# Run separately in your terminal:
# ollama run qwen3:0.6b "Explain what a large language model is in two sentences."
print("Ollama Q1 OpenAI comparison:", ask(
    "Explain what a large language model is in two sentences."
).choices[0].message.content)

# Ollama Response:
# A large language model is an AI system trained on vast amounts
# of text data to understand language, generate text, and
# interact with humans. It can process context, learn from
# patterns, and produce coherent responses, making it versatile
# for tasks like writing, answering questions, or summarizing
# information.