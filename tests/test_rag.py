"""
Tests for src/rag.py (Phase 3: FAISS + RAG).

Tests cover:
- Vector store initialization
- Index building from CSV data
- Semantic search functionality
- Context retrieval
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class TestVectorStore:
    """Tests for InventoryVectorStore."""

    def test_vector_store_imports(self):
        from rag import InventoryVectorStore
        assert InventoryVectorStore is not None

    def test_vector_store_creation(self):
        from rag import InventoryVectorStore
        store = InventoryVectorStore()
        assert store.model_name == "all-MiniLM-L6-v2"

    def test_vector_store_build_index(self):
        from rag import InventoryVectorStore
        store = InventoryVectorStore()
        
        data_dir = Path(__file__).resolve().parent.parent / "data"
        items = pd.read_csv(data_dir / "items.csv")
        sales = pd.read_csv(data_dir / "daily_sales.csv", parse_dates=["date"])
        
        store.build_index(items, sales)
        assert store._index is not None
        assert store._index.ntotal > 0

    def test_vector_store_search(self):
        from rag import InventoryVectorStore
        store = InventoryVectorStore()
        
        data_dir = Path(__file__).resolve().parent.parent / "data"
        items = pd.read_csv(data_dir / "items.csv")
        sales = pd.read_csv(data_dir / "daily_sales.csv", parse_dates=["date"])
        
        store.build_index(items, sales)
        results = store.search("toothpaste", k=3)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert "document" in results[0]
        assert "score" in results[0]
        assert "metadata" in results[0]

    def test_vector_store_search_with_filter(self):
        from rag import InventoryVectorStore
        store = InventoryVectorStore()
        
        data_dir = Path(__file__).resolve().parent.parent / "data"
        items = pd.read_csv(data_dir / "items.csv")
        sales = pd.read_csv(data_dir / "daily_sales.csv", parse_dates=["date"])
        
        store.build_index(items, sales)
        results = stock = store.search("sales", k=5, filter_type="sales_summary")
        
        assert isinstance(results, list)
        for r in results:
            assert r["metadata"]["type"] == "sales_summary"

    def test_vector_store_retrieve_context(self):
        from rag import InventoryVectorStore
        store = InventoryVectorStore()
        
        data_dir = Path(__file__).resolve().parent.parent / "data"
        items = pd.read_csv(data_dir / "items.csv")
        sales = pd.read_csv(data_dir / "daily_sales.csv", parse_dates=["date"])
        
        store.build_index(items, sales)
        context = store.retrieve_context("What is the stock level of biscuits?")
        
        assert isinstance(context, str)
        assert len(context) > 0

    def test_vector_store_retrieve_context_empty(self):
        from rag import InventoryVectorStore
        store = InventoryVectorStore()
        
        context = store.retrieve_context("test query")
        assert context == "No relevant context found."

    def test_singleton_get_vector_store(self):
        from rag import get_vector_store, InventoryVectorStore, _store
        # Reset singleton
        import rag
        rag._store = None
        
        store = get_vector_store()
        assert store is not None
        assert isinstance(store, InventoryVectorStore)

    def test_build_store_from_data(self):
        from rag import build_store_from_data
        store = build_store_from_data()
        assert store._index is not None
        assert store._index.ntotal > 0
