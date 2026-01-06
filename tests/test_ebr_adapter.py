"""
Tests for embedding-based retriever adapter.
"""

import pytest
import numpy as np
from c4engine.retriever import EmbeddingRetrieverAdapter, RetrievalResult


class TestEmbeddingRetrieverAdapter:
    """Test embedding retriever adapter."""
    
    def test_initialization(self):
        """Test retriever initialization."""
        retriever = EmbeddingRetrieverAdapter(
            model_name="all-MiniLM-L6-v2",
            index_type="flat",
        )
        assert retriever.model_name == "all-MiniLM-L6-v2"
        assert retriever.index_type == "flat"
        assert retriever.index is None
    
    def test_build_index(self):
        """Test building retrieval index."""
        retriever = EmbeddingRetrieverAdapter()
        documents = [
            "Crane safety is important",
            "Load capacity must be checked",
            "Wind speed affects operations",
        ]
        
        retriever.build_index(documents)
        
        assert retriever.index is not None
        assert len(retriever.documents) == 3
        assert retriever.doc_embeddings is not None
        assert retriever.doc_embeddings.shape[0] == 3
    
    def test_retrieve(self):
        """Test document retrieval."""
        retriever = EmbeddingRetrieverAdapter()
        documents = [
            "Crane safety is important",
            "Load capacity must be checked",
            "Wind speed affects operations",
        ]
        
        retriever.build_index(documents)
        results = retriever.retrieve("crane safety", k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(hasattr(r, 'doc_id') for r in results)
        assert all(hasattr(r, 'score') for r in results)
    
    def test_evaluate_retrieval(self):
        """Test retrieval evaluation."""
        retriever = EmbeddingRetrieverAdapter()
        documents = [
            "Crane safety procedures",
            "Load capacity guidelines",
            "Wind speed restrictions",
        ]
        
        retriever.build_index(documents)
        
        queries = ["safety procedures", "wind limits"]
        relevant_docs = [[0], [2]]
        
        metrics = retriever.evaluate_retrieval(queries, relevant_docs, k=2)
        
        assert "precision@k" in metrics
        assert "recall@k" in metrics
        assert "mrr" in metrics
        assert 0 <= metrics["precision@k"] <= 1
        assert 0 <= metrics["recall@k"] <= 1


class TestRetrievalResult:
    """Test RetrievalResult dataclass."""
    
    def test_retrieval_result_creation(self):
        """Test creating a retrieval result."""
        result = RetrievalResult(
            doc_id=0,
            doc_text="Test document",
            score=0.95,
            metadata={"source": "test"},
        )
        
        assert result.doc_id == 0
        assert result.doc_text == "Test document"
        assert result.score == 0.95
        assert result.metadata["source"] == "test"
