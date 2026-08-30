from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_HTML_DIR = DATA_DIR / "raw" / "html"
RAW_DOCUMENTS_DIR = DATA_DIR / "raw" / "documents"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
VECTORDB_DIR = DATA_DIR / "vectordb"
METADATA_DIR = DATA_DIR / "metadata"

INGESTION_MANIFEST_PATH = METADATA_DIR / "ingestion_manifest.json"
CHUNKING_MANIFEST_PATH = METADATA_DIR / "chunking_manifest.json"
EMBEDDING_MANIFEST_PATH = METADATA_DIR / "embedding_manifest.json"
VECTOR_STORE_MANIFEST_PATH = METADATA_DIR / "vector_store_manifest.json"
ALL_CHUNKS_PATH = CHUNKS_DIR / "all_chunks.json"
ALL_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "all_embeddings.json"
