"""
C4Engine: Adapted EBR and agentic RAG for crane safety analysis.

This package provides:
1. LLM-assisted synthetic supervision for embedding-based retriever adaptation
2. Agentic RAG for feature-aware analysis
3. Multifaceted evaluation suite
"""

__version__ = "0.1.0"

from .retriever.ebr_adapter import EmbeddingRetrieverAdapter
from .retriever.synthetic_supervision import SyntheticSupervisionGenerator
from .rag.agentic_rag import AgenticRAG
from .evaluation.evaluator import RAGEvaluator

__all__ = [
    "EmbeddingRetrieverAdapter",
    "SyntheticSupervisionGenerator",
    "AgenticRAG",
    "RAGEvaluator",
]
