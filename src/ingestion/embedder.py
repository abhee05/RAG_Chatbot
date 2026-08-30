"""Phase 3 — Embedding: encode chunks with a shared sentence-transformers model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config.models import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME
from src.config.paths import (
    ALL_CHUNKS_PATH,
    ALL_EMBEDDINGS_PATH,
    CHUNKING_MANIFEST_PATH,
    EMBEDDING_MANIFEST_PATH,
    EMBEDDINGS_DIR,
    METADATA_DIR,
)
from src.ingestion.chunker import Chunk

DEFAULT_BATCH_SIZE = 32


class EmbeddingError(Exception):
    """Raised when chunks cannot be embedded."""


@dataclass
class EmbeddedChunk:
    chunk_id: str
    text: str
    fund_name: str
    source_url: str
    ingestion_timestamp: str
    embedding: list[float]
    embedding_model: str
    embedding_dim: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EmbeddingManifest:
    embedding_timestamp: str
    chunking_timestamp: str | None
    embedding_model: str
    embedding_dim: int
    status: str
    chunks_embedded: int
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Embedder:
    """Shared embedder for batch chunk encoding and single-query encoding (FR-7)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            return int(self.model.get_embedding_dimension())
        return int(self.model.get_sentence_embedding_dimension())

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return [vector.tolist() for vector in vectors]

    def encode_query(self, query: str) -> list[float]:
        if not query.strip():
            raise EmbeddingError("Query text cannot be empty.")
        return self.encode_texts([query.strip()])[0]


def ensure_output_dirs() -> None:
    for path in (EMBEDDINGS_DIR, METADATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_chunks(path: Path | None = None) -> list[Chunk]:
    chunks_path = path or ALL_CHUNKS_PATH
    if not chunks_path.exists():
        raise EmbeddingError(
            f"Chunks file not found at {chunks_path}. Run Phase 2 chunking first."
        )

    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    return [
        Chunk(
            chunk_id=item["chunk_id"],
            text=item["text"],
            fund_name=item["fund_name"],
            source_url=item["source_url"],
            ingestion_timestamp=item["ingestion_timestamp"],
        )
        for item in payload
    ]


def _assert_chunking_available() -> dict[str, Any]:
    if not CHUNKING_MANIFEST_PATH.exists():
        raise EmbeddingError(
            "Chunking manifest not found. Run Phase 2 chunking first."
        )

    manifest = json.loads(CHUNKING_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") == "failed":
        raise EmbeddingError(
            "Chunking failed. Downstream embedding is blocked (E-8)."
        )
    if not manifest.get("total_chunks"):
        raise EmbeddingError("No chunks available to embed.")
    return manifest


def embed_chunks(
    chunks: list[Chunk],
    embedder: Embedder | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[EmbeddedChunk]:
    if not chunks:
        raise EmbeddingError("Cannot embed an empty chunk list.")

    encoder = embedder or Embedder()
    texts = [chunk.text for chunk in chunks]
    vectors = encoder.encode_texts(texts, batch_size=batch_size)

    if len(vectors) != len(chunks):
        raise EmbeddingError("Embedding count does not match chunk count.")

    dimension = encoder.dimension
    if dimension != EMBEDDING_DIMENSION:
        raise EmbeddingError(
            f"Expected embedding dimension {EMBEDDING_DIMENSION}, got {dimension}."
        )

    embedded: list[EmbeddedChunk] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        if len(vector) != dimension:
            raise EmbeddingError(
                f"Vector for {chunk.chunk_id} has invalid dimension {len(vector)}."
            )
        embedded.append(
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                fund_name=chunk.fund_name,
                source_url=chunk.source_url,
                ingestion_timestamp=chunk.ingestion_timestamp,
                embedding=vector,
                embedding_model=encoder.model_name,
                embedding_dim=dimension,
            )
        )
    return embedded


def save_embeddings(embeddings: list[EmbeddedChunk]) -> Path:
    payload = [item.to_dict() for item in embeddings]
    ALL_EMBEDDINGS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return ALL_EMBEDDINGS_PATH


def save_manifest(manifest: EmbeddingManifest) -> Path:
    EMBEDDING_MANIFEST_PATH.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return EMBEDDING_MANIFEST_PATH


def run_embedding(
    embedder: Embedder | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbeddingManifest:
    from src.ingestion.loader import utc_now_iso

    ensure_output_dirs()
    chunking_manifest = _assert_chunking_available()
    chunks = load_chunks()
    embedded_chunks = embed_chunks(chunks, embedder=embedder, batch_size=batch_size)
    output_path = save_embeddings(embedded_chunks)

    manifest = EmbeddingManifest(
        embedding_timestamp=utc_now_iso(),
        chunking_timestamp=chunking_manifest.get("chunking_timestamp"),
        embedding_model=embedded_chunks[0].embedding_model,
        embedding_dim=embedded_chunks[0].embedding_dim,
        status="success",
        chunks_embedded=len(embedded_chunks),
        output_path=str(output_path),
    )
    save_manifest(manifest)
    return manifest


def load_embeddings(path: Path | None = None) -> list[EmbeddedChunk]:
    embeddings_path = path or ALL_EMBEDDINGS_PATH
    if not embeddings_path.exists():
        raise EmbeddingError(
            f"Embeddings file not found at {embeddings_path}. Run Phase 3 embedding first."
        )

    payload = json.loads(embeddings_path.read_text(encoding="utf-8"))
    return [
        EmbeddedChunk(
            chunk_id=item["chunk_id"],
            text=item["text"],
            fund_name=item["fund_name"],
            source_url=item["source_url"],
            ingestion_timestamp=item["ingestion_timestamp"],
            embedding=item["embedding"],
            embedding_model=item["embedding_model"],
            embedding_dim=item["embedding_dim"],
        )
        for item in payload
    ]


def _assert_embedding_available() -> dict[str, Any]:
    if not EMBEDDING_MANIFEST_PATH.exists():
        raise EmbeddingError(
            "Embedding manifest not found. Run Phase 3 embedding first."
        )

    manifest = json.loads(EMBEDDING_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "success":
        raise EmbeddingError(
            "Embedding phase did not succeed. Vector store is blocked (E-8)."
        )
    if not manifest.get("chunks_embedded"):
        raise EmbeddingError("No embeddings available to store.")
    return manifest


def main() -> None:
    manifest = run_embedding()
    print(
        f"Embedding {manifest.status}: "
        f"{manifest.chunks_embedded} chunks embedded "
        f"with {manifest.embedding_model} ({manifest.embedding_dim}-dim)."
    )
    print(f"Manifest: {EMBEDDING_MANIFEST_PATH}")
    print(f"Embeddings: {ALL_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
