
# Week 6 Warmup - RAG and LlamaIndex

from dotenv import load_dotenv
import os
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key.")


# --- RAG Concepts ---

# Concepts Q1

# Scenario A: RAG
# RAG is best because the legal team needs information from
# internal documents that are updated regularly.

# Scenario B: Fine-tuning
# Fine-tuning is best because the company wants a specific
# writing style and has many examples to train the model.

# Scenario C: Prompt engineering
# Prompt engineering is best because the analyst only needs
# information from one short report.


# Concepts Q2

# A confidently wrong AI answer is dangerous because people
# may trust it without checking whether it is correct.
# For example, incorrect medical advice could harm someone.
# When AI admits uncertainty, users know to verify the answer.


# Concepts Q3

# Correct order of the RAG pipeline:
# 1. Extract text from source documents.
# 2. Split text into smaller chunks.
# 3. Convert text chunks into embeddings.
# 4. Receive the user's query.
# 5. Embed the user's query.
# 6. Retrieve the most relevant chunks.
# 7. Inject retrieved chunks into the prompt.
# 8. Generate a response from the LLM.

print("RAG Concepts completed.")


# --- Keyword RAG ---

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# Keyword Q1

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])

# Keyword Q1 Reflection
# The function selected loyalty.txt instead of hours.txt.
# Three documents had the same overlap score of 1.
# Since the function sorts in reverse order, loyalty.txt won the tie.
# This shows that keyword matching can select the wrong document
# because it does not understand the meaning of the question.


# Keyword Q2

query = "Do you have anything without caffeine?"

result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])

# Q2 Reflection:
# No document was selected because none of the documents
# contained the same keywords as the question.
# Keyword RAG failed to find a match.
# Semantic RAG would work better because it understands
# meaning rather than relying on exact words.

# Keyword Q3

# Prediction:
# I predict loyalty.txt because rewards are related to
# a loyalty program. However, the function may not recognize
# that relationship without matching keywords.

query = "How do I sign up for rewards?"

result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])

# Q3 Actual Result:
# My prediction was incorrect. The function returned "None found."
# The loyalty document discusses earning points and free drinks,
# but it does not contain the words "rewards" or "sign up."
# This shows that keyword RAG cannot recognize related meanings.


# --- Semantic RAG Concepts ---

# Semantic Q1

# 1. A vector embedding is a list of numbers that represents
# the meaning of text. It helps computers compare how
# similar different pieces of text are.

# 2. The chunk with a score of 0.85 is more relevant.
# A higher cosine similarity score means the text is
# more closely related to the meaning of the question.

# 3. Semantic search understands relationships between words.
# It can find relevant information even when the exact
# words in the question are not present in the document.


# Semantic Q2

# Feature                  Keyword RAG              Semantic RAG
# ---------------------------------------------------------------------
# What is compared?        Exact word overlap       Meaning of text
# What is retrieved?       Full document            Relevant chunks
# Can it handle synonyms?  No                       Yes
# Storage format           Text dictionary          Vector index
# Relevance score          Matching word count      Cosine similarity

print("Semantic RAG Concepts completed.")


# --- LlamaIndex ---

# LlamaIndex Q1

# Set the embedding and response models
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Locate the Brightleaf PDF documents
pdf_dir = Path("assignments_06/resources/brightleaf_pdfs")

assert pdf_dir.exists(), f"Folder not found: {pdf_dir}"

# Load the documents
docs = SimpleDirectoryReader(str(pdf_dir)).load_data()

# Build the vector index
index = VectorStoreIndex.from_documents(docs)

# Retrieve the top 3 relevant chunks
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for question in questions:
    print("\nQuestion:", question)

    response = query_engine.query(question)
    print("Answer:", response)

    for node in response.source_nodes:
        print("Similarity score:", node.score)
        print("Text snippet:", node.node.get_content()[:150])
        print("-" * 30)


# LlamaIndex Q1 Reflections:
# 1. The retrieved information was mostly relevant.
#    The highest-scoring chunks matched each question.
#    However, some chunks came from unrelated documents.

# 2. The answers sounded confident and detailed.
#    The model did not express much uncertainty.

# 3. Some unexpected chunks were retrieved.
#    For the benefits question, partnership and mission
#    documents appeared. For the security question,
#    benefits and mission documents appeared.


# LlamaIndex Q2: Compare top_k=1 and top_k=5

question = "What employee benefits does BrightLeaf offer?"

for k in [1, 5]:
    print(f"\n--- Top K = {k} ---")

    engine = index.as_query_engine(similarity_top_k=k)
    response = engine.query(question)

    print("Answer:", response)

    for node in response.source_nodes:
        print("Similarity score:", node.score)
        print("Text snippet:", node.node.get_content()[:150])
        print("-" * 30)


# LlamaIndex Q2 Reflections:
# 1. Top K = 1 gave a detailed answer using one relevant
#    document. Top K = 5 gave a shorter summary.

# 2. More context is not always better because it can
#    include unrelated information.

# 3. Top K = 1 worked better for this question because
#    the employee benefits document had enough information.


# LlamaIndex Q3: Test a difficult question

question = "What is BrightLeaf's employee vacation policy?"

print("\n--- LlamaIndex Q3 ---")
print("Question:", question)

response = query_engine.query(question)

print("Answer:", response)

for node in response.source_nodes:
    print("Similarity score:", node.score)
    print("Text snippet:", node.node.get_content()[:200])
    print("-" * 30)


# LlamaIndex Q3 Reflections:
# 1. I expected the model to struggle because the documents
#    do not provide a specific employee vacation policy.

# 2. The model correctly stated that the information was
#    unavailable instead of making up an answer.

# 3. The retrieved chunks included employee benefits,
#    partnerships, and the company mission. Only the
#    benefits document was somewhat relevant.

# 4. I could improve the system by adding a vacation
#    policy document and filtering unrelated results.


# LlamaIndex Q4: Evaluate RAG responses

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

judge_llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

evaluation_questions = [
    "What employee benefits does BrightLeaf offer?",
    "What is BrightLeaf's employee vacation policy?",
]

for question in evaluation_questions:
    print("\n--- LlamaIndex Q4 Evaluation ---")
    print("Question:", question)

    response = query_engine.query(question)
    print("Answer:", response)

    faithfulness_result = faithfulness_evaluator.evaluate_response(
        query=question,
        response=response,
    )

    relevancy_result = relevancy_evaluator.evaluate_response(
        query=question,
        response=response,
    )

    print("Faithfulness:", faithfulness_result.passing)
    print("Relevancy:", relevancy_result.passing)
    print("Faithfulness score:", faithfulness_result.score)
    print("Relevancy score:", relevancy_result.score)


# LlamaIndex Q4 Reflections:
# 1. Faithfulness checks whether an answer is supported
#    by the retrieved documents. Relevancy checks whether
#    the answer addresses the question.

# 2. The employee benefits question scored 1.0 for both
#    metrics because the answer was supported and relevant.

# 3. The vacation policy question scored 0.0 for both
#    metrics because the documents lacked the information.
#    However, the model correctly admitted it did not know.

# 4. LLM-as-judge evaluation is useful, but it has
#    limitations. A low score does not always mean the
#    model behaved incorrectly, and scores may vary.

