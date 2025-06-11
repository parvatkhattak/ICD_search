from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os
from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

collections = ["Medical_Coder"]

for collection in collections:
    qdrant_client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Adjust size based on embedding model
    )

    print(collection, "collection was created successfully.")