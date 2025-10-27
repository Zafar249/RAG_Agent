from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

api_key = os.environ["PINECONE_API_KEY"]

# Initialize a pinecone client
client = Pinecone(api_key=api_key)  

# Create a dense index with seperate embeddings
index_name = "storage-py"   

# Load the model for generating sentence embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def create_vector_db():
    global client, index_name
    # If the index does not already exist
    if not client.has_index(index_name):
        client.create_index(
            name = index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

def insert_into_vector_db(chunks):
    # Target the index
    dense_index = client.Index(index_name)

    records = get_vector_embeddings(chunks)

    # Upsert the records into a namespace
    dense_index.upsert(namespace="example-namespace", vectors=records)

def get_vector_embeddings(chunks):
    global model
    records = []

    embeddings = model.encode(chunks)

    # Iterate over each chunk adding it to the vector database
    for i in range(len(embeddings)):
        records.append({"id":"vec"+str(i+1),"values":embeddings[i].tolist(), "metadata":{"chunk_text":chunks[i]}})

    return records

def query_vector_db(text):
    global model
    query = model.encode(text).tolist()

    # Target the index
    dense_index = client.Index(index_name)

    results = dense_index.query(
    namespace="example-namespace",
    vector = query,
    top_k=3,
    include_values=True,
    include_metadata=True
    )

    context = preprocess(results)
    return context

def preprocess(results):
    context = ""
    # Add the text of each relevant chunk to the context
    for hit in results["matches"]:
        context += hit["metadata"]["chunk_text"]

    return context

def delete_vector_db():
    global client, index_name
    # If the index exists, delete it.
    if client.has_index(index_name):
        client.delete_index(index_name)