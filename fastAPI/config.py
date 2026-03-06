import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = BASE_DIR / "chromadb" / "chroma_db"
embed_model = SentenceTransformer("BAAI/bge-m3")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("products")
