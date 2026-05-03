import os
from dotenv import load_dotenv

# Load env variables from .env if present
load_dotenv()

# Set env var to avoid huggingface tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_huggingface import HuggingFaceEmbeddings

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding model. Uses HF_TOKEN if available in environment."""
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Returns the LangChain embeddings object."""
        return self.embeddings
