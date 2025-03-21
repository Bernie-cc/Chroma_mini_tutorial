# Unlocking the Power of Vector Search with Chroma: A Mini-Tutorial for Movie Recommendations

**For the assignemnt 3, I choose MLOps Tool Option and Chroma is tool I want to write about.**

Machine learning systems have transformed how we interact with data, with vector embeddings playing a crucial role in tasks like recommendation systems, semantic search, and natural language processing. However, managing and querying these high-dimensional embeddings in production systems remains a challenge. Enter Chroma, an open-source vector database designed to simplify and supercharge your ML workflows.

In this blog post, we’ll introduce Chroma, explore the problems it solves, and walk through a mini-tutorial where we build a **content-based movie recommendation system using Chroma**. Let’s dive in!

## What Is Chroma?

Chroma is a lightweight, open-source vector database optimized for managing and querying embeddings. It allows machine learning practitioners to efficiently store, retrieve, and manipulate vector representations of data. Whether you're building a recommendation system, a semantic search engine, or an AI assistant, Chroma provides a robust foundation for working with embeddings in production.

Key features of Chroma include:

- **Ease of Use**: Simple Python APIs for quick integration into ML pipelines.
- **Scalability**: Handles large-scale embeddings efficiently.
- **Integration**: Works seamlessly with popular ML frameworks like Hugging Face and LangChain.

## The Problem Chroma Solves

In production ML systems, embeddings are commonly used to represent items like movies, documents, or user preferences in a high-dimensional space. However, without a proper database to manage and query these embeddings, you may face:

- **Storage Challenges**: Embeddings can quickly grow into gigabytes or terabytes.
- **Query Inefficiency**: Finding nearest neighbors in high-dimensional space can be slow without optimization.
- **Data Management Complexity**: Linking embeddings to their metadata (e.g., movie titles, genres) is often cumbersome.

Chroma addresses these issues by providing a fast and efficient vector database that supports metadata management and similarity queries.

## Mini-Tutorial: Building a Movie Recommendation System with Chroma

For this tutorial, let’s build a content-based recommendation system for a movie streaming platform. The goal is to recommend movies similar to a given movie based on their embeddings.

**Prerequisites**

Before we start, ensure you have the following installed:

- Python 3.7+
- Chroma (pip install chromadb)
- Sentence Transformers (pip install sentence-transformers)

**Step 1: Data Preparation**

We’ll use a small dataset of movies with their titles, descriptions, year and genres.

```python
movies = [
    {
        "id": "1",
        "title": "Inception",
        "description": "A mind-bending thriller about dreams.",
        "genre": "Sci-Fi",
        "year": 2010
    },
    {
        "id": "2",
        "title": "The Matrix",
        "description": "A hacker discovers the truth about reality.",
        "genre": "Sci-Fi",
        "year": 1999
    },
    {
        "id": "3",
        "title": "The Godfather",
        "description": "The saga of a crime family.",
        "genre": "Crime",
        "year": 1972
    },
    {
        "id": "4",
        "title": "Interstellar",
        "description": "A team of explorers travels through a wormhole.",
        "genre": "Sci-Fi",
        "year": 2014
    },
    {
        "id": "5",
        "title": "The Social Network",
        "description": "The story of Facebook's creation.",
        "genre": "Drama",
        "year": 2010
    }
]
``` 

**Step 2: Generate Embeddings**

We’ll use Sentence Transformers to generate embeddings for the movie descriptions.

```python
from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for movie descriptions
for movie in movies:
    movie["embedding"] = model.encode(movie["description"])
```


**Step 3: Initialize Chroma and Add Data**

Now, we’ll initialize a Chroma database and insert the movie embeddings.

```python
import chromadb
from chromadb.config import Settings

# Initialize Chroma
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))

# Create a collection for movies
collection = client.get_or_create_collection("movies")

# Add movies to the collection
for movie in movies:
    collection.add(
        ids=[movie["id"]],
        metadatas=[{"title": movie["title"], "genre": movie["genre"], "year": movie["year"]}],
        embeddings=[movie["embedding"]]
    )
```

**Step 4: Query the Database**

Finally, we’ll query Chroma to find movies similar to a given movie based on their embeddings.

```python
# Query similar movies to "Inception"
query = "A mind-bending thriller about dreams."
query_embedding = model.encode(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# Print the recommendations
for title in results["metadatas"][0]:
    print(f"Recommended Movie: {title['title']}")


# Output:
# Recommended Movie: Inception
# Recommended Movie: Interstellar
# Recommended Movie: The Matrix
```

**Step 5: Query the Database with filter**


```python
query = "A story of friendship and ambition."
query_embedding = model.encode(query)

# query with metadata filter
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    where={"genre": "Drama"}  
)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# Print the recommendations
for title in results["metadatas"][0]:
    print(f"Recommended Movie: {title['title']} ({title['year']})")

## Output
# Recommended Movie: The Social Network (2010)
```

## Strengths and Limitations of Chroma

**Strengths:**

- Ease of Use: Chroma’s simple Python APIs make it easy to integrate into existing workflows.
- Speed: Efficient nearest-neighbor search enables real-time recommendations.
- Metadata Support: Allows linking embeddings with meaningful metadata (e.g., movie titles).

**Limitations:**

- Scalability: While Chroma handles large datasets, it may require additional optimization for massive-scale deployments.
- Integration Overhead: Some manual work is needed to integrate Chroma with non-standard ML pipelines.
- Limited Features: Chroma focuses on vector search and metadata but lacks advanced analytics or visualization tools.

## Why Choose Chroma for Your ML Workflow?

Chroma is an excellent choice for teams looking to manage vector embeddings effectively in production. Whether you're working on recommendations, semantic search, or personalization, Chroma provides the performance and flexibility you need to succeed.

## Conclusion

In this blog post, we introduced Chroma, a powerful open-source vector database, and demonstrated how to use it to build a content-based movie recommendation system. With Chroma, managing embeddings and performing similarity searches becomes fast and efficient, making it an invaluable tool for production ML systems.

If you’re looking for a lightweight and intuitive solution to handle embeddings, give Chroma a try! You can find its documentation and source code on [GitHub](https://github.com/Bernie-cc/Chroma_mini_tutorial#).

## References

- [Chroma Documentation](https://www.trychroma.com/docs)
- [Chroma GitHub Repository](https://github.com/chroma-core/chroma)