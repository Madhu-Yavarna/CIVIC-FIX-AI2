import os
import re
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pymupdf  # PyMuPDF
import faiss

from app.config import FAISS_DIR, CATEGORY_KEYWORDS

logger = logging.getLogger("civicfix.rag")

INDEX_FILE = FAISS_DIR / "civic_faiss.index"
METADATA_FILE = FAISS_DIR / "civic_metadata.json"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768
EMBEDDING_BATCH_SIZE = 32

_STOPWORDS = {
    "the", "is", "in", "on", "of", "and", "to", "for", "a", "an", "with",
    "by", "at", "from", "that", "this", "are", "be", "as", "or", "it",
    "into", "near", "has", "have", "been", "was", "were", "will", "can",
    "must", "under", "within", "via", "any", "all", "per", "such"
}


def _stem_token(w: str) -> str:
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _local_embed_texts(texts: List[str]) -> np.ndarray:
    """
    Deterministic 768-dimensional normalized embedding fallback using
    word, stem, bigram, and domain-concept feature hashing.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    vectors = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)

    for row_idx, text in enumerate(texts):
        lower = (text or "").lower()
        body_lower = lower
        for cat in CATEGORY_KEYWORDS:
            prefix = f"{cat.lower()}:"
            if body_lower.startswith(prefix):
                body_lower = body_lower[len(prefix):].strip()
                break

        raw_tokens = [t for t in re.findall(r"[a-z0-9]{2,}", body_lower) if t not in _STOPWORDS]
        stems = [_stem_token(t) for t in raw_tokens]

        features: Dict[str, float] = {}
        for tok, st in zip(raw_tokens, stems):
            features[f"w:{tok}"] = features.get(f"w:{tok}", 0.0) + 1.0
            features[f"s:{st}"] = features.get(f"s:{st}", 0.0) + 1.2

        for i in range(len(stems) - 1):
            bg = f"b:{stems[i]}_{stems[i + 1]}"
            features[bg] = features.get(bg, 0.0) + 1.5

        # Boost civic category domain signals so queries and corresponding SOP sections align strongly
        for cat, kw_list in CATEGORY_KEYWORDS.items():
            cat_key = f"cat:{cat.lower()}"
            matched = 0
            for kw in kw_list:
                if kw in body_lower:
                    matched += 1
            if matched > 0:
                features[cat_key] = features.get(cat_key, 0.0) + (3.5 + min(matched, 4) * 1.5)

        if not features:
            features["empty"] = 1.0

        vec = vectors[row_idx]
        for feat, weight in features.items():
            scaled = float(np.log1p(weight)) if weight <= 3.0 else float(weight)
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h1 = int.from_bytes(digest[:4], "little")
            h2 = int.from_bytes(digest[4:], "little")
            idx1 = h1 % EMBEDDING_DIMENSION
            sign1 = 1.0 if (h1 & 1) == 0 else -1.0
            idx2 = h2 % EMBEDDING_DIMENSION
            sign2 = 1.0 if (h2 & 1) == 0 else -1.0
            vec[idx1] += sign1 * scaled
            vec[idx2] += sign2 * (scaled * 0.5)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def _embed_texts(texts: List[str], task_type: str) -> np.ndarray:
    """Generate normalized Gemini embeddings with automatic offline fallback."""
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    from app.llm_engine import get_llm_config

    api_key = (
        os.environ.get("CIVICFIX_EMBEDDING_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or get_llm_config().get("gemini_api_key")
    )
    if api_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            batches = []
            for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch = texts[start:start + EMBEDDING_BATCH_SIZE]
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=EMBEDDING_DIMENSION,
                    ),
                )
                if not response.embeddings or len(response.embeddings) != len(batch):
                    raise RuntimeError("Gemini returned an incomplete embedding response.")
                batches.extend(embedding.values for embedding in response.embeddings)

            vectors = np.asarray(batches, dtype=np.float32)
            if vectors.shape != (len(texts), EMBEDDING_DIMENSION):
                raise RuntimeError(
                    f"Gemini returned embedding shape {vectors.shape}; "
                    f"expected ({len(texts)}, {EMBEDDING_DIMENSION})."
                )
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise RuntimeError("Gemini returned a zero-length embedding.")
            return np.ascontiguousarray(vectors / norms)
        except Exception as exc:
            logger.warning(f"Gemini embedding call failed ({exc}); using local 768-d embedding fallback.")

    return _local_embed_texts(texts)


class RAGEngine:
    def __init__(self):
        self.index_path = str(INDEX_FILE)
        self.meta_path = str(METADATA_FILE)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []
        self.needs_rebuild = True
        self._load_or_initialize()

    def _load_or_initialize(self):
        """Loads existing FAISS index and metadata or initializes empty structures."""
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                index = faiss.read_index(self.index_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                if index.d != EMBEDDING_DIMENSION or index.ntotal != len(metadata):
                    raise ValueError("Saved FAISS index does not match the current embedding format.")
                self.index = index
                self.metadata = metadata
                self.needs_rebuild = (self.index.ntotal == 0)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors and {len(self.metadata)} metadata records.")
            except Exception as e:
                logger.error(f"Error loading existing index: {e}. Reinitializing.")
                self._initialize_empty()
        else:
            self._initialize_empty()

    def _initialize_empty(self):
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.metadata = []
        self._persist()

    def _persist(self):
        """Saves current index and metadata to disk."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def extract_and_chunk_pdf(
        self,
        pdf_path: str,
        doc_id: int,
        title: str,
        filename: str,
        category_tag: Optional[str] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Extracts text from PDF page-by-page using PyMuPDF and creates overlapping chunks.
        """
        doc = pymupdf.open(pdf_path)
        chunks: List[Dict[str, Any]] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_no = page_idx + 1
            text = page.get_text("text") or ""
            # Clean up whitespace
            cleaned_text = " ".join(text.split())
            if not cleaned_text:
                continue

            # Sliding window chunking
            start = 0
            while start < len(cleaned_text):
                end = min(start + chunk_size, len(cleaned_text))
                chunk_str = cleaned_text[start:end].strip()

                if len(chunk_str) >= 40:  # Skip trivial fragments
                    chunks.append({
                        "doc_id": doc_id,
                        "document_title": title,
                        "filename": filename,
                        "category_tag": category_tag,
                        "page_number": page_no,
                        "text": chunk_str
                    })

                if end == len(cleaned_text):
                    break
                start += (chunk_size - chunk_overlap)

        doc.close()
        return chunks

    def add_document_chunks(self, chunks: List[Dict[str, Any]]):
        """Generates embeddings for chunks and adds them to FAISS index."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]

        # Generate normalized embeddings for cosine similarity via inner product
        embeddings = _embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

        if self.index is None or self.index.d != EMBEDDING_DIMENSION:
            self._initialize_empty()

        self.index.add(embeddings)
        self.metadata.extend(chunks)
        self._persist()
        self.needs_rebuild = False
        logger.info(f"Added {len(chunks)} chunks to FAISS index. Total: {self.index.ntotal}")

    def rebuild_index_from_documents(self, documents: List[Dict[str, Any]]):
        """
        Rebuilds index completely from an active list of documents.
        Used after document deletion or modifications to ensure exact sync.
        """
        self._initialize_empty()
        all_chunks: List[Dict[str, Any]] = []

        for doc in documents:
            if os.path.exists(doc["file_path"]):
                chunks = self.extract_and_chunk_pdf(
                    pdf_path=doc["file_path"],
                    doc_id=doc["id"],
                    title=doc["title"],
                    filename=doc["filename"],
                    category_tag=doc.get("category_tag")
                )
                all_chunks.extend(chunks)

        if all_chunks:
            self.add_document_chunks(all_chunks)
        else:
            self._persist()
            self.needs_rebuild = False

    def query_guidelines(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.30
    ) -> List[Dict[str, Any]]:
        """
        Searches FAISS for top_k relevant guideline chunks.
        Returns empty list if below threshold or if no documents exist.
        """
        if self.index is None or self.index.ntotal == 0 or not self.metadata:
            return []

        query_vec = _embed_texts([query], task_type="RETRIEVAL_QUERY")

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            score = float(dist)
            # Check relevance threshold
            if score >= similarity_threshold:
                meta = self.metadata[idx]
                results.append({
                    "document_title": meta.get("document_title", "Government Guideline"),
                    "filename": meta.get("filename", "guideline.pdf"),
                    "page_number": meta.get("page_number", 1),
                    "score": round(score, 3),
                    "excerpt": meta.get("text", "")
                })

        return results


# Global singleton
rag_engine = RAGEngine()
