from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Step 1: Sample movie dataset with metadata
movies = [
    {
        "id": "1",
        "title": "Inception",
        "description": "A mind-bending thriller about dreams and reality.",
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
        "description": "The saga of a crime family and their legacy.",
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
print("Successfully loaded the movie dataset.")

# Step 2: Load a pre-trained Sentence Transformer model to generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Successfully loaded the model.")

# Generate embeddings for each movie description
for movie in movies:
    movie["embedding"] = model.encode(movie["description"])

# Step 3: Initialize Chroma and create a collection
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection named "movies"
collection = client.get_or_create_collection("movies")
print("Successfully created the collection.")
# Add movie data to the collection
for movie in movies:
    collection.add(
        ids=[movie["id"]],
        metadatas=[{"title": movie["title"], "genre": movie["genre"], "year": movie["year"]}],
        embeddings=[movie["embedding"]]
    )


# Step 4: Query the collection
# Query similar movies to "Inception"
query = "A mind-bending thriller about dreams."
query_embedding = model.encode(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# Print the recommendations
print("For query: ", query)
for title in results["metadatas"][0]:
    print(f"Recommended Movie: {title['title']}")



# Step 5: Query the collection with metadata filtering

# Define a new query for a movie about "a story of friendship and ambition"
query = "A story of friendship and ambition."
query_embedding = model.encode(query)

# Query similar movies to the query embedding, filtering by genre "Drama"
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    where={"genre": "Drama"}  # Add metadata filter
)

# Print the recommended results
print("For query: ", query)
for metadata in results["metadatas"][0]:
    print(f"Recommended Movie: {metadata['title']} ({metadata['year']})")