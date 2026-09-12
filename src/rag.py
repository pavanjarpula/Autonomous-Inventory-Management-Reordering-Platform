"""
FAISS-backed vector store for SellerSense.

Indexes item metadata, historical feedback, and sales summaries so the
chatbot can retrieve relevant context for natural language queries.
Uses sentence-transformers for embeddings (local, no API needed).

LangSmith: all operations are traceable via LangSmith tracing (trace_run()).
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logger import get_logger

logger = get_logger(__name__, extra_data={"module": "rag"})


# Default embedding model (small, fast, good enough for 25 SKUs)
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class InventoryVectorStore:
    """
    FAISS-backed vector store for inventory context retrieval.
    
    Indexes three types of documents:
    1. Item metadata (name, category, supplier, cost)
    2. Historical feedback (rejection reasons, adjustments)
    3. Sales summaries (recent trends, seasonality)
    
    Provides semantic search via natural language queries.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        if not HAS_FAISS:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers is required: pip install sentence-transformers")
        
        self.model_name = model_name
        self._model = None
        self._index = None
        self._documents = []
        self._metadatas = []
        self._dimension = None

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading embedding model", extra={"extra_data": {"model": self.model_name}})
            self._model = SentenceTransformer(self.model_name)
            try:
                self._dimension = self._model.get_embedding_dimension()
            except AttributeError:
                self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts into vectors."""
        return self.model.encode(texts, show_progress_bar=False)

    def build_index(
        self,
        items: pd.DataFrame,
        sales: pd.DataFrame,
        feedback_log: pd.DataFrame | None = None,
        festival_calendar: pd.DataFrame | None = None,
    ):
        """
        Build the FAISS index from inventory data.
        
        Args:
            items: items.csv DataFrame
            sales: daily_sales.csv DataFrame
            feedback_log: feedback_log.csv DataFrame (optional)
            festival_calendar: festival_calendar.csv DataFrame (optional)
        """
        documents = []
        metadatas = []

        # 1. Index item metadata
        for _, row in items.iterrows():
            doc = (
                f"{row['item_name']} ({row['item_id']}) in {row['category']} category. "
                f"Cost: ₹{row['unit_cost_inr']}, Price: ₹{row['unit_price_inr']}. "
                f"Supplier: {row['supplier_id']}."
            )
            documents.append(doc)
            metadatas.append({
                "type": "item_metadata",
                "item_id": row["item_id"],
                "item_name": row["item_name"],
                "category": row["category"],
            })

        # 2. Index recent sales summaries per item
        recent_sales = sales.groupby("item_id").agg(
            total_units=("units_sold", "sum"),
            avg_daily=("units_sold", "mean"),
            days_with_sales=("units_sold", "count"),
        ).reset_index()

        for _, row in recent_sales.iterrows():
            item_name = items[items["item_id"] == row["item_id"]]["item_name"].values
            item_name = item_name[0] if len(item_name) > 0 else row["item_id"]
            doc = (
                f"{item_name} sales summary: {row['total_units']} total units sold, "
                f"averaging {row['avg_daily']:.1f} units per day over {row['days_with_sales']} days."
            )
            documents.append(doc)
            metadatas.append({
                "type": "sales_summary",
                "item_id": row["item_id"],
                "total_units": int(row["total_units"]),
                "avg_daily": float(row["avg_daily"]),
            })

        # 3. Index feedback history if available
        if feedback_log is not None and len(feedback_log) > 0:
            for _, row in feedback_log.iterrows():
                item_name = items[items["item_id"] == row.get("item_id", "")]["item_name"].values
                item_name = item_name[0] if len(item_name) > 0 else row.get("item_id", "unknown")
                
                if row.get("decision") == "reject":
                    doc = (
                        f"User rejected order for {item_name}. "
                        f"Reason: {row.get('reason_code', 'not specified')}. "
                    )
                elif row.get("decision") == "approve":
                    doc = f"User approved order for {item_name}. "
                else:
                    doc = f"User action on {item_name}: {row.get('decision', 'unknown')}."
                
                documents.append(doc)
                metadatas.append({
                    "type": "feedback",
                    "item_id": row.get("item_id", ""),
                    "decision": row.get("decision", ""),
                    "reason_code": row.get("reason_code", ""),
                })

        # 4. Index festival context if available
        if festival_calendar is not None and len(festival_calendar) > 0:
            for _, row in festival_calendar.iterrows():
                doc = (
                    f"Festival: {row['festival_name']} on {row['date']}. "
                    f"Affects categories: {row['affected_categories']}. "
                    f"Uplift multiplier: {row['uplift_multiplier']}x."
                )
                documents.append(doc)
                metadatas.append({
                    "type": "festival",
                    "festival_name": row["festival_name"],
                    "affected_categories": row["affected_categories"],
                })

        # Build FAISS index
        if not documents:
            logger.warning("No documents to index")
            return

        logger.info("Building FAISS index", extra={"extra_data": {
            "n_documents": len(documents),
        }})

        embeddings = self._embed(documents)
        self._index = faiss.IndexFlatL2(self._dimension)
        self._index.add(embeddings.astype("float32"))
        self._documents = documents
        self._metadatas = metadatas

    def search(
        self,
        query: str,
        k: int = 5,
        filter_type: str | None = None,
    ) -> list[dict]:
        """
        Semantic search over indexed documents.
        
        Args:
            query: Natural language query
            k: Number of results to return
            filter_type: Optional filter by document type
                         ("item_metadata", "sales_summary", "feedback", "festival")
        
        Returns:
            List of dicts with keys: document, score, metadata
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        query_embedding = self._embed([query]).astype("float32")
        distances, indices = self._index.search(query_embedding, min(k, self._index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            meta = self._metadatas[idx]
            if filter_type and meta.get("type") != filter_type:
                continue
            results.append({
                "document": self._documents[idx],
                "score": float(1 / (1 + dist)),  # Convert distance to similarity
                "metadata": meta,
            })

        return results

    def retrieve_context(
        self,
        query: str,
        item_id: str | None = None,
        k: int = 3,
    ) -> str:
        """
        Retrieve relevant context for a chatbot query.
        
        Combines semantic search with optional item-specific filtering.
        Returns a formatted string ready to inject into a prompt.
        """
        results = self.search(query, k=k * 2)  # Get more, then filter

        if item_id:
            # Prioritize item-specific results
            item_results = [r for r in results if r["metadata"].get("item_id") == item_id]
            other_results = [r for r in results if r["metadata"].get("item_id") != item_id]
            results = item_results + other_results

        # Deduplicate and take top k
        seen = set()
        filtered = []
        for r in results:
            doc_key = r["document"][:100]  # Use first 100 chars as dedup key
            if doc_key not in seen:
                seen.add(doc_key)
                filtered.append(r)
            if len(filtered) >= k:
                break

        if not filtered:
            return "No relevant context found."

        context_parts = []
        for r in filtered:
            context_parts.append(f"- {r['document']}")
        
        return "\n".join(context_parts)


# Module-level singleton for easy use
_store: InventoryVectorStore | None = None


def get_vector_store() -> InventoryVectorStore:
    """Get or create the singleton vector store."""
    global _store
    if _store is None:
        _store = InventoryVectorStore()
    return _store


def build_store_from_data(data_dir: Path | None = None) -> InventoryVectorStore:
    """Build the vector store from CSV data files."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data"

    store = get_vector_store()
    
    items = pd.read_csv(data_dir / "items.csv")
    sales = pd.read_csv(data_dir / "daily_sales.csv", parse_dates=["date"])
    
    feedback_log = None
    feedback_path = data_dir / "feedback_log.csv"
    if feedback_path.exists():
        feedback_log = pd.read_csv(feedback_path)
    
    festival_calendar = None
    festival_path = data_dir / "festival_calendar.csv"
    if festival_path.exists():
        festival_calendar = pd.read_csv(festival_path, parse_dates=["date"])

    store.build_index(items, sales, feedback_log, festival_calendar)
    return store
