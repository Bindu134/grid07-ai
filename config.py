# config.py
# Loads all environment variables and exposes them as typed constants.
# Add your keys to a .env file — never commit real keys.

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# --- Embeddings ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Router ---
# NOTE: The spec suggests 0.85 but all-MiniLM-L6-v2 scores range 0.2-0.55
# for cross-topic similarity. Tuned to 0.3 for realistic routing.
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
