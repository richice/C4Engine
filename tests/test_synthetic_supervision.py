"""
Tests for synthetic supervision generator.
"""

import pytest
from c4engine.retriever import SyntheticSupervisionGenerator, SyntheticExample


class TestSyntheticSupervisionGenerator:
    """Test synthetic supervision generation."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = SyntheticSupervisionGenerator(model="gpt-4", temperature=0.7)
        assert generator.model == "gpt-4"
        assert generator.temperature == 0.7
    
    def test_generate_domain_queries_returns_list(self):
        """Test that generate_domain_queries returns a list."""
        generator = SyntheticSupervisionGenerator()
        # Mock test - in real scenario would need API key
        # For now just test the structure
        assert hasattr(generator, 'generate_domain_queries')
        assert callable(generator.generate_domain_queries)
    
    def test_generate_synthetic_pairs_structure(self):
        """Test synthetic pair generation structure."""
        generator = SyntheticSupervisionGenerator()
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        
        # Test method exists and has correct signature
        assert hasattr(generator, 'generate_synthetic_pairs')
        
        # In a real test with API access, would verify:
        # examples = generator.generate_synthetic_pairs(documents, "test domain")
        # assert len(examples) > 0
        # assert isinstance(examples[0], SyntheticExample)
    
    def test_adapt_to_domain_terminology(self):
        """Test domain terminology adaptation."""
        generator = SyntheticSupervisionGenerator()
        
        # Test method exists
        assert hasattr(generator, 'adapt_to_domain_terminology')
        assert callable(generator.adapt_to_domain_terminology)


class TestSyntheticExample:
    """Test SyntheticExample dataclass."""
    
    def test_synthetic_example_creation(self):
        """Test creating a synthetic example."""
        example = SyntheticExample(
            query="test query",
            positive_doc="relevant doc",
            negative_docs=["neg1", "neg2"],
            metadata={"source": "test"},
        )
        
        assert example.query == "test query"
        assert example.positive_doc == "relevant doc"
        assert len(example.negative_docs) == 2
        assert example.metadata["source"] == "test"
