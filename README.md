# Semantic Document Search using LangChain

## Project Overview
This project implements a Semantic Document Search system in the domain of AI in Education. It searches educational documents using both semantic search and keyword search techniques.

The system processes 20 text documents and 15 user queries. It retrieves the top 5 relevant documents for each query and calculates cosine similarity scores.

## Objectives
- Build a semantic document search system.
- Store and retrieve document embeddings.
- Compare semantic search with keyword search.
- Calculate cosine similarity scores.
- Export search results into CSV files.

## Technologies Used
- Python 3.12
- LangChain
- ChromaDB
- Sentence Transformers
- Pandas
- NumPy
- Jupyter Notebook

## Dataset
- Total documents: 20
- Total queries: 15
- Domain: AI in Education

## Project Structure
```text
Semantic_Document_Search/
│
├── dataset/
│   ├── documents/
│   ├── queries.csv
│   ├── search_results.csv
│   └── similarity_scores.csv
│
├── chroma_db/
├── notebook/
│   └── Semantic_Document_Search.ipynb
│
├── semantic_search.py
├── requirements.txt
└── README.md
```

## Methodology
1. Collect and prepare educational text documents.
2. Generate document embeddings using Sentence Transformers.
3. Store embeddings in ChromaDB.
4. Perform semantic search using LangChain.
5. Perform keyword-based search for comparison.
6. Retrieve the top 5 results for each query.
7. Calculate cosine similarity scores.
8. Export the results into CSV files.

## How to Run
Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the search program:

```bash
python semantic_search.py
```

## Output Files
- `search_results.csv` – Contains semantic and keyword search results.
- `similarity_scores.csv` – Contains cosine similarity scores.

## Conclusion
This project demonstrates how semantic search can retrieve educational information based on meaning and context. Comparing semantic and keyword search helps understand the benefits of AI-powered information retrieval.
