
# Week 6 Project
import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


# Step 1: Load the API key
load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("API key loaded successfully.")
else:
    print("API key not found.")


# Step 2: Load the Groundwork Coffee documents
docs_dir = Path("assignments_06/resources/groundwork_docs")

assert docs_dir.exists(), f"Folder not found: {docs_dir}"

documents = SimpleDirectoryReader(
    input_dir=str(docs_dir)
).load_data()

print("Number of documents loaded:", len(documents))

for doc in documents:
    print("File:", doc.metadata.get("file_name"))

# Step 3: Build the vector index
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("Groundwork Coffee RAG system is ready!")


# Step 4: Ask five customer questions

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    print("\n------------------------------")
    print("Question:", question)

    response = query_engine.query(question)
    print("Answer:", response)

    print("\nRetrieved sources:")

    for node in response.source_nodes:
        print("File:", node.node.metadata.get("file_name"))
        print("Similarity score:", node.score)
        print("Text snippet:", node.node.get_content()[:200])
        print("------------------------------")


# Step 4 Reflections:
# 1. The system answered all five questions accurately
#    based on the information in the documents.

# 2. The answers were clear, helpful, and confident.

# 3. Some unrelated documents were retrieved.
#    For example, the weekend hours question retrieved
#    our_story.txt before faq.txt.

# 4. The system could improve by prioritizing documents
#    that directly relate to each question.


# Step 5: Test a question with missing information

question = "What is the Wi-Fi password at Groundwork Coffee?"

print("\n--- Step 5: Difficult Question ---")
print("Question:", question)

response = query_engine.query(question)

print("Answer:", response)

print("\nRetrieved sources:")

for node in response.source_nodes:
    print("File:", node.node.metadata.get("file_name"))
    print("Similarity score:", node.score)
    print("Text snippet:", node.node.get_content()[:200])
    print("------------------------------")


# Step 5 Reflections:
# 1. I expected the system to struggle because the
#    question asked for a specific Wi-Fi password.

# 2. The system suggested asking a barista instead
#    of providing the actual password.

# 3. The retrieved documents included the company story,
#    menu, and FAQ. The displayed snippets did not
#    provide the requested password.

# 4. The system could improve by checking whether
#    retrieved information supports the answer and
#    clearly stating when information is unavailable.


# Step 6: Final Reflections

# 1. LlamaIndex simplified the project by handling
#    document loading, embeddings, indexing, and retrieval.
#    This reduced the amount of code I needed to write.

# 2. Another business could use RAG to answer customer
#    questions about products, services, and policies.

# 3. RAG cannot completely prevent hallucinations.
#    If documents are missing or irrelevant, the model
#    may still generate an unsupported answer.
