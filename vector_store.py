"""
Vector database management for RAG.

Thin wrapper around ChromaDB + langchain's RecursiveCharacterTextSplitter.
Documents are split into chunks, embedded by Chroma's default embedder,
and stored in a single named collection. The RAG agent calls `search` to
retrieve relevant context before handing it to Claude.

If chromadb or langchain_text_splitters aren't installable (e.g. missing
system toolchain on Windows), the VectorStore silently becomes a no-op so
the rest of the app keeps working — Claude just gets an empty retrieval
context instead of chunked Jira docs.
"""
import logging
from typing import List, Dict, Any

from config import config

logger = logging.getLogger(__name__)

try:
    import chromadb
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _HAS_CHROMA = True
except Exception as _e:  # pragma: no cover - optional dependency
    chromadb = None  # type: ignore
    RecursiveCharacterTextSplitter = None  # type: ignore
    _HAS_CHROMA = False
    logger.warning(
        "chromadb not available (%s); RAG context retrieval disabled. "
        "Install with `pip install chromadb langchain-text-splitters` to enable.",
        _e,
    )


class VectorStore:
    def __init__(self, collection_name: str = "jira_rag"):
        """Initialize vector store with ChromaDB (or no-op if unavailable)."""
        self.collection_name = collection_name
        if not _HAS_CHROMA:
            self.client = None
            self.collection = None
            self.text_splitter = None
            return
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        logger.info(f"Initialized vector store with collection: {collection_name}")

    @property
    def enabled(self) -> bool:
        return _HAS_CHROMA and self.collection is not None

    def add_documents(
        self, documents: List[Dict[str, Any]], metadata: Dict[str, Any] = None
    ):
        """Add documents to vector store"""
        if not self.enabled:
            return
        try:
            ids: List[str] = []
            texts: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            metadata = metadata or {}

            for i, doc in enumerate(documents):
                base_id = doc.get("id", f"doc_{i}")
                content = doc.get("content", "") or ""
                if not content.strip():
                    continue
                chunks = self.text_splitter.split_text(content)
                for j, chunk in enumerate(chunks):
                    chunk_id = f"{base_id}_chunk_{j}"
                    md = {
                        "source": doc.get("source", "unknown"),
                        "type": doc.get("type", "unknown"),
                        "ticket_key": doc.get("ticket_key", ""),
                    }
                    md.update(metadata)
                    ids.append(chunk_id)
                    texts.append(chunk)
                    metadatas.append(md)

            if ids:
                self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
            logger.info(f"Added {len(ids)} chunks from {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search vector store"""
        if not self.enabled:
            return []
        try:
            top_k = top_k or config.TOP_K_RETRIEVAL
            results = self.collection.query(query_texts=[query], n_results=top_k)
            out = []
            docs = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            for i, doc in enumerate(docs):
                out.append({
                    "content": doc,
                    "distance": distances[i] if i < len(distances) else None,
                    "metadata": metas[i] if i < len(metas) else {},
                })
            return out
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def clear_collection(self, collection_name: str = None):
        """Clear all documents from collection"""
        if not self.enabled:
            return
        try:
            name = collection_name or self.collection_name
            self.client.delete_collection(name=name)
            self.collection = self.client.get_or_create_collection(name=name)
            logger.info(f"Cleared collection: {name}")
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
