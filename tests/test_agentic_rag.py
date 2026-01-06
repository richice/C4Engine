"""
Tests for agentic RAG system.
"""

import pytest
from c4engine.rag import AgenticRAG, DocumentType, RAGContext, AgentAction
from c4engine.retriever import EmbeddingRetrieverAdapter


class TestAgenticRAG:
    """Test agentic RAG system."""
    
    def test_initialization(self):
        """Test RAG initialization."""
        retriever = EmbeddingRetrieverAdapter()
        documents = ["Test doc 1", "Test doc 2"]
        retriever.build_index(documents)
        
        rag = AgenticRAG(
            retriever=retriever,
            model="gpt-4",
            max_iterations=5,
        )
        
        assert rag.retriever is not None
        assert rag.model == "gpt-4"
        assert rag.max_iterations == 5
        assert len(rag.action_history) == 0
    
    def test_document_type_detection(self):
        """Test document type detection."""
        retriever = EmbeddingRetrieverAdapter()
        documents = [
            "This is a narrative document with sentences.",
            "Column1 | Column2 | Column3",
        ]
        retriever.build_index(documents)
        
        rag = AgenticRAG(retriever=retriever)
        
        # Test detection methods exist
        assert hasattr(rag, '_detect_document_type')
        assert hasattr(rag, '_is_tabular')
    
    def test_chunk_long_text(self):
        """Test text chunking."""
        retriever = EmbeddingRetrieverAdapter()
        retriever.build_index(["dummy"])
        
        rag = AgenticRAG(retriever=retriever, chunk_size=50)
        
        long_text = "word " * 100  # 500 characters
        chunks = rag._chunk_long_text(long_text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 60 for chunk in chunks)  # Some margin
    
    def test_query_structure(self):
        """Test that query returns proper structure."""
        retriever = EmbeddingRetrieverAdapter()
        documents = ["Crane safety document", "Load capacity info"]
        retriever.build_index(documents)
        
        rag = AgenticRAG(retriever=retriever)
        
        # Mock query without actual API call
        # In real test, would check:
        # response = rag.query("test query", enable_multi_step=False)
        # assert "answer" in response
        # assert "reasoning_steps" in response
        # assert "context_type" in response
        
        # For now, verify method exists
        assert hasattr(rag, 'query')
        assert callable(rag.query)


class TestDocumentType:
    """Test DocumentType enum."""
    
    def test_document_types(self):
        """Test document type enumeration."""
        assert DocumentType.NARRATIVE.value == "narrative"
        assert DocumentType.TABULAR.value == "tabular"
        assert DocumentType.MIXED.value == "mixed"


class TestAgentAction:
    """Test AgentAction dataclass."""
    
    def test_agent_action_creation(self):
        """Test creating an agent action."""
        action = AgentAction(
            action_type="retrieve",
            reasoning="Need to find relevant documents",
            parameters={"k": 5},
            result=["doc1", "doc2"],
        )
        
        assert action.action_type == "retrieve"
        assert action.reasoning == "Need to find relevant documents"
        assert action.parameters["k"] == 5
        assert len(action.result) == 2
