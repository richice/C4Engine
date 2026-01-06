"""
Tests for RAG evaluator.
"""

import pytest
from c4engine.evaluation import RAGEvaluator, EvaluationResult


class TestRAGEvaluator:
    """Test RAG evaluation suite."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = RAGEvaluator(
            enable_human_eval=True,
            enable_llm_eval=True,
        )
        
        assert evaluator.enable_human_eval is True
        assert evaluator.enable_llm_eval is True
        assert len(evaluator.evaluation_results) == 0
    
    def test_evaluate_retrieval(self):
        """Test retrieval evaluation."""
        evaluator = RAGEvaluator()
        
        queries = ["query1", "query2"]
        retrieved_docs = [
            [{"doc_id": 0}, {"doc_id": 1}, {"doc_id": 2}],
            [{"doc_id": 1}, {"doc_id": 3}, {"doc_id": 4}],
        ]
        ground_truth = [[0, 1], [1, 3]]
        
        metrics = evaluator.evaluate_retrieval(
            queries=queries,
            retrieved_docs=retrieved_docs,
            ground_truth=ground_truth,
            k_values=[1, 3],
        )
        
        assert "precision@1" in metrics
        assert "recall@1" in metrics
        assert "precision@3" in metrics
        assert "mrr" in metrics
        assert "map" in metrics
        assert "ndcg" in metrics
        
        # Check all scores are between 0 and 1
        for key, value in metrics.items():
            assert 0 <= value <= 1, f"{key} score out of range: {value}"
    
    def test_evaluate_generation(self):
        """Test generation evaluation."""
        evaluator = RAGEvaluator(enable_llm_eval=False)
        
        questions = ["What is crane safety?", "What are load limits?"]
        answers = [
            "Crane safety involves following procedures",
            "Load limits depend on crane capacity",
        ]
        contexts = [
            "Crane safety procedures include inspections",
            "Crane load capacity specifications",
        ]
        
        metrics = evaluator.evaluate_generation(
            questions=questions,
            generated_answers=answers,
            contexts=contexts,
        )
        
        assert "faithfulness" in metrics
        assert "relevance" in metrics
        assert "completeness" in metrics
        
        # Check scores are between 0 and 1
        for key, value in metrics.items():
            assert 0 <= value <= 1, f"{key} score out of range: {value}"
    
    def test_calculate_ndcg(self):
        """Test NDCG calculation."""
        evaluator = RAGEvaluator()
        
        retrieved = [
            {"doc_id": 0},
            {"doc_id": 1},
            {"doc_id": 2},
        ]
        relevant_set = {0, 2}
        
        ndcg = evaluator._calculate_ndcg(retrieved, relevant_set)
        
        assert 0 <= ndcg <= 1
    
    def test_calculate_average_precision(self):
        """Test Average Precision calculation."""
        evaluator = RAGEvaluator()
        
        retrieved = [
            {"doc_id": 0},
            {"doc_id": 1},
            {"doc_id": 2},
        ]
        relevant_set = {0, 2}
        
        ap = evaluator._calculate_average_precision(retrieved, relevant_set)
        
        assert 0 <= ap <= 1
    
    def test_human_evaluation_interface(self):
        """Test human evaluation interface."""
        evaluator = RAGEvaluator(enable_human_eval=True)
        
        questions = ["Q1", "Q2"]
        answers = ["A1", "A2"]
        contexts = ["C1", "C2"]
        
        items = evaluator.human_evaluation_interface(
            questions=questions,
            answers=answers,
            contexts=contexts,
        )
        
        assert len(items) == 2
        assert all("question" in item for item in items)
        assert all("answer" in item for item in items)
        assert all("evaluation" in item for item in items)
    
    def test_get_summary(self):
        """Test getting evaluation summary."""
        evaluator = RAGEvaluator()
        
        # Add a mock result
        result = EvaluationResult(
            metric_name="test_metric",
            score=0.85,
            details={"precision": 0.8, "recall": 0.9},
        )
        evaluator.evaluation_results.append(result)
        
        summary = evaluator.get_summary()
        
        assert "num_evaluations" in summary
        assert summary["num_evaluations"] == 1
        assert "results" in summary
        assert len(summary["results"]) == 1


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""
    
    def test_evaluation_result_creation(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            metric_name="retrieval_effectiveness",
            score=0.85,
            details={"precision@5": 0.8, "recall@5": 0.9},
        )
        
        assert result.metric_name == "retrieval_effectiveness"
        assert result.score == 0.85
        assert result.details["precision@5"] == 0.8
        assert result.details["recall@5"] == 0.9
