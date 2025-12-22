import os

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("STACK_MODEL", "qwen3:4b-instruct")
EMBED_MODEL = os.environ.get("STACK_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
INDEX_DIR = os.environ.get("STACK_INDEX_DIR", ".chroma")
TOP_K = 5
