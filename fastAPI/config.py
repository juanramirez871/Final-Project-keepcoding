import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "chromadb" / "chroma_db"

embed_model = SentenceTransformer("BAAI/bge-m3")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("products")

venv_vits = BASE_DIR / "model_vits_finetuning" / ".venv" / "bin" / "python"
script_vits = BASE_DIR / "model_vits_finetuning" / "run_model_vits.py"
folder_vits = BASE_DIR / "model_vits_finetuning"

venv_llama = BASE_DIR / "model_llama_finetuning" / ".venv" / "bin" / "python"
script_llama = BASE_DIR / "model_llama_finetuning" / "run_model_llama.py"
folder_llama = BASE_DIR / "model_llama_finetuning"
