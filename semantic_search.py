
import os
import re
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --------------------------------
# 1. Load embedding model
# --------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True}
)

# --------------------------------
# 2. Connect to ChromaDB
# --------------------------------

vectorstore = Chroma(
    collection_name="education_documents",
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# --------------------------------
# 3. Load 20 documents
# --------------------------------

folder = "dataset/documents"

documents = []
document_ids = []

for filename in sorted(os.listdir(folder)):
    if filename.endswith(".txt"):
        filepath = os.path.join(folder, filename)

        with open(filepath, "r", encoding="utf-8-sig") as file:
            content = file.read().strip()

        documents.append(content)
        document_ids.append(filename.replace(".txt", ""))

print("Total documents loaded:", len(documents))

# --------------------------------
# 4. Add missing documents
# --------------------------------

existing_data = vectorstore.get(include=["documents"])
existing_ids = set(existing_data["ids"])

new_documents = []
new_ids = []

for doc_id, content in zip(document_ids, documents):
    if doc_id not in existing_ids:
        new_documents.append(content)
        new_ids.append(doc_id)

if new_documents:
    vectorstore.add_texts(
        texts=new_documents,
        ids=new_ids,
        metadatas=[
            {"source": doc_id} for doc_id in new_ids
        ]
    )

print("New documents added:", len(new_documents))
print("Total documents in ChromaDB:", vectorstore._collection.count())

# --------------------------------
# 5. Create document ID mapping
# --------------------------------

stored_data = vectorstore.get(include=["documents"])

content_to_id = {}

for stored_id, stored_content in zip(
    stored_data["ids"],
    stored_data["documents"]
):
    content_to_id[stored_content.strip()] = stored_id

# Generate normalized document embeddings
document_vectors = model.encode(
    documents,
    normalize_embeddings=True
)

vector_map = dict(zip(document_ids, document_vectors))

# --------------------------------
# 6. Read 15 queries
# --------------------------------

queries_df = pd.read_csv("dataset/queries.csv")

search_results = []
similarity_results = []

stop_words = {
    "what", "is", "are", "the", "a", "an", "how",
    "can", "does", "do", "in", "of", "to", "and",
    "for", "with", "on", "by", "their", "they"
}

# --------------------------------
# 7. Process queries
# --------------------------------

for _, row in queries_df.iterrows():

    query_id = row["Query_ID"]
    query = str(row["Query"])

    print("\nProcessing:", query_id, "-", query)

    query_vector = model.encode(
        query,
        normalize_embeddings=True
    )

    # Semantic search
    semantic_docs = vectorstore.similarity_search(
        query,
        k=5
    )

    for rank, doc in enumerate(semantic_docs, start=1):

        doc_id = doc.metadata.get("source")

        if not doc_id:
            doc_id = content_to_id.get(
                doc.page_content.strip()
            )

        if doc_id not in vector_map:
            print("Skipping document with unknown ID")
            continue

        similarity = float(
            np.dot(query_vector, vector_map[doc_id])
        )

        search_results.append({
            "Query_ID": query_id,
            "Query": query,
            "Search_Type": "Semantic",
            "Rank": rank,
            "Document_ID": doc_id,
            "Similarity_Score": similarity,
            "Retrieved_Document": doc.page_content
        })

        similarity_results.append({
            "Query_ID": query_id,
            "Query": query,
            "Document_ID": doc_id,
            "Rank": rank,
            "Similarity_Score": similarity
        })

    # Keyword-based search
    query_words = [
        word for word in re.findall(r"\w+", query.lower())
        if word not in stop_words
    ]

    keyword_scores = []

    for doc_id, content in zip(document_ids, documents):

        text = content.lower()

        score = sum(
            text.count(word) for word in query_words
        )

        keyword_scores.append(
            (doc_id, content, score)
        )

    keyword_scores.sort(
        key=lambda item: (-item[2], item[0])
    )

    for rank, (doc_id, content, score) in enumerate(
        keyword_scores[:5], start=1
    ):

        similarity = float(
            np.dot(query_vector, vector_map[doc_id])
        )

        search_results.append({
            "Query_ID": query_id,
            "Query": query,
            "Search_Type": "Keyword",
            "Rank": rank,
            "Document_ID": doc_id,
            "Similarity_Score": similarity,
            "Keyword_Match_Count": score,
            "Retrieved_Document": content
        })

print("\nSaving results...")

# --------------------------------
# 8. Export CSV files
# --------------------------------

os.makedirs("dataset", exist_ok=True)

results_df = pd.DataFrame(search_results)
results_df.to_csv(
    "dataset/search_results.csv",
    index=False
)

scores_df = pd.DataFrame(similarity_results)
scores_df.to_csv(
    "dataset/similarity_scores.csv",
    index=False
)

# --------------------------------
# 9. Display summary
# --------------------------------

print("\nSEMANTIC DOCUMENT SEARCH COMPLETED!")
print("Total documents:", len(documents))
print("Total queries:", len(queries_df))
print(
    "Semantic results:",
    len(similarity_results)
)
print("Total search results:", len(results_df))
print("Results saved: dataset/search_results.csv")
print("Similarity scores saved: dataset/similarity_scores.csv")
