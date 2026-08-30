"""Phase 4 — Vector Store: persist embeddings in ChromaDB for similarity search."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config.models import EMBEDDING_MODEL_NAME
from src.config.paths import (
    EMBEDDING_MANIFEST_PATH,
    METADATA_DIR,
    VECTOR_STORE_MANIFEST_PATH,
    VECTORDB_DIR,
)
from src.config.vectordb import COLLECTION_NAME
from src.ingestion.embedder import EmbeddedChunk, EmbeddingError, load_embeddings

UPSERT_BATCH_SIZE = 100


class VectorStoreError(Exception):
    """Raised when embeddings cannot be stored in ChromaDB."""


@dataclass
class VectorStoreManifest:
    store_timestamp: str
    embedding_timestamp: str | None
    ingestion_timestamp: str
    collection_name: str
    embedding_model: str
    embedding_dim: int
    status: str
    chunks_stored: int
    persist_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    fund_name: str
    source_url: str
    ingestion_timestamp: str
    distance: float


def ensure_output_dirs() -> None:
    for path in (VECTORDB_DIR, METADATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _get_chroma_client(persist_path: Path | None = None) -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise VectorStoreError(
            "chromadb is required. Install with: pip install chromadb"
        ) from exc

    return chromadb.PersistentClient(path=str(persist_path or VECTORDB_DIR))


def _replace_collection(
    client: Any,
    collection_name: str,
    collection_metadata: dict[str, str],
) -> Any:
    try:
        client.delete_collection(collection_name)
    except (ValueError, Exception):
        pass

    return client.create_collection(
        name=collection_name,
        metadata=collection_metadata,
    )


def _chunk_metadata(item: EmbeddedChunk) -> dict[str, str]:
    return {
        "fund_name": item.fund_name,
        "source_url": item.source_url,
        "ingestion_timestamp": item.ingestion_timestamp,
    }


def upsert_embeddings(
    embeddings: list[EmbeddedChunk],
    persist_path: Path | None = None,
    collection_name: str = COLLECTION_NAME,
) -> Any:
    if not embeddings:
        raise VectorStoreError("Cannot store an empty embedding list.")

    models = {item.embedding_model for item in embeddings}
    dimensions = {item.embedding_dim for item in embeddings}
    if len(models) != 1 or len(dimensions) != 1:
        raise VectorStoreError("All embeddings must share the same model and dimension.")

    embedding_model = next(iter(models))
    embedding_dim = next(iter(dimensions))
    ingestion_timestamp = embeddings[0].ingestion_timestamp

    client = _get_chroma_client(persist_path)
    collection = _replace_collection(
        client,
        collection_name,
        {
            "ingestion_timestamp": ingestion_timestamp,
            "embedding_model": embedding_model,
            "embedding_dim": str(embedding_dim),
        },
    )

    for start in range(0, len(embeddings), UPSERT_BATCH_SIZE):
        batch = embeddings[start : start + UPSERT_BATCH_SIZE]
        collection.add(
            ids=[item.chunk_id for item in batch],
            embeddings=[item.embedding for item in batch],
            documents=[item.text for item in batch],
            metadatas=[_chunk_metadata(item) for item in batch],
        )

    if collection.count() != len(embeddings):
        raise VectorStoreError(
            f"Expected {len(embeddings)} records in ChromaDB, found {collection.count()}."
        )

    return collection


def get_collection(
    persist_path: Path | None = None,
    collection_name: str = COLLECTION_NAME,
) -> Any:
    client = _get_chroma_client(persist_path)
    try:
        return client.get_collection(collection_name)
    except Exception as exc:
        raise VectorStoreError(
            f"Collection '{collection_name}' not found. Run Phase 4 vector store first."
        ) from exc


def get_collection_ingestion_timestamp(
    persist_path: Path | None = None,
    collection_name: str = COLLECTION_NAME,
) -> str:
    collection = get_collection(persist_path, collection_name)
    metadata = collection.metadata or {}
    timestamp = metadata.get("ingestion_timestamp")
    if not timestamp:
        raise VectorStoreError("Collection metadata missing ingestion_timestamp.")
    return str(timestamp)


def query_similar(
    query_embedding: list[float],
    top_k: int = 3,
    persist_path: Path | None = None,
    collection_name: str = COLLECTION_NAME,
) -> list[RetrievedChunk]:
    if top_k < 1:
        raise VectorStoreError("top_k must be at least 1.")

    collection = get_collection(persist_path, collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: list[RetrievedChunk] = []
    for chunk_id, text, metadata, distance in zip(
        ids, documents, metadatas, distances, strict=True
    ):
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text or "",
                fund_name=metadata.get("fund_name", ""),
                source_url=metadata.get("source_url", ""),
                ingestion_timestamp=metadata.get("ingestion_timestamp", ""),
                distance=float(distance),
            )
        )
    return retrieved


def save_manifest(manifest: VectorStoreManifest) -> Path:
    VECTOR_STORE_MANIFEST_PATH.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return VECTOR_STORE_MANIFEST_PATH


def run_vector_store(
    persist_path: Path | None = None,
    collection_name: str = COLLECTION_NAME,
) -> VectorStoreManifest:
    from src.ingestion.loader import utc_now_iso

    ensure_output_dirs()

    if not EMBEDDING_MANIFEST_PATH.exists():
        raise VectorStoreError("Embedding manifest not found. Run Phase 3 first.")

    embedding_manifest = json.loads(
        EMBEDDING_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if embedding_manifest.get("status") != "success":
        raise VectorStoreError(
            "Embedding phase did not succeed. Vector store is blocked (E-8)."
        )

    embeddings = load_embeddings()
    collection = upsert_embeddings(
        embeddings,
        persist_path=persist_path,
        collection_name=collection_name,
    )

    manifest = VectorStoreManifest(
        store_timestamp=utc_now_iso(),
        embedding_timestamp=embedding_manifest.get("embedding_timestamp"),
        ingestion_timestamp=get_collection_ingestion_timestamp(
            persist_path, collection_name
        ),
        collection_name=collection_name,
        embedding_model=embedding_manifest.get("embedding_model", EMBEDDING_MODEL_NAME),
        embedding_dim=int(embedding_manifest.get("embedding_dim", 0)),
        status="success",
        chunks_stored=collection.count(),
        persist_path=str(persist_path or VECTORDB_DIR),
    )
    save_manifest(manifest)
    return manifest


def main() -> None:
    manifest = run_vector_store()
    print(
        f"Vector store {manifest.status}: "
        f"{manifest.chunks_stored} chunks stored in "
        f"'{manifest.collection_name}'."
    )
    print(f"Ingestion timestamp: {manifest.ingestion_timestamp}")
    print(f"Manifest: {VECTOR_STORE_MANIFEST_PATH}")
    print(f"Persist path: {manifest.persist_path}")


if __name__ == "__main__":
    main()
