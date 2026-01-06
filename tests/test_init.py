"""Test configuration and initialization."""

import pytest


def test_package_import():
    """Test that package can be imported."""
    import c4engine
    assert c4engine.__version__ == "0.1.0"


def test_main_imports():
    """Test main component imports."""
    from c4engine import (
        EmbeddingRetrieverAdapter,
        SyntheticSupervisionGenerator,
        AgenticRAG,
        RAGEvaluator,
    )
    
    assert EmbeddingRetrieverAdapter is not None
    assert SyntheticSupervisionGenerator is not None
    assert AgenticRAG is not None
    assert RAGEvaluator is not None


def test_retriever_module():
    """Test retriever module imports."""
    from c4engine.retriever import (
        EmbeddingRetrieverAdapter,
        RetrievalResult,
        SyntheticSupervisionGenerator,
        SyntheticExample,
    )
    
    assert EmbeddingRetrieverAdapter is not None
    assert RetrievalResult is not None
    assert SyntheticSupervisionGenerator is not None
    assert SyntheticExample is not None


def test_rag_module():
    """Test RAG module imports."""
    from c4engine.rag import (
        AgenticRAG,
        DocumentType,
        RAGContext,
        AgentAction,
    )
    
    assert AgenticRAG is not None
    assert DocumentType is not None
    assert RAGContext is not None
    assert AgentAction is not None


def test_evaluation_module():
    """Test evaluation module imports."""
    from c4engine.evaluation import (
        RAGEvaluator,
        EvaluationResult,
    )
    
    assert RAGEvaluator is not None
    assert EvaluationResult is not None


def test_utils_module():
    """Test utils module imports."""
    from c4engine.utils import (
        load_config,
        setup_api_keys,
        validate_config,
        load_documents_from_file,
    )
    
    assert load_config is not None
    assert setup_api_keys is not None
    assert validate_config is not None
    assert load_documents_from_file is not None
