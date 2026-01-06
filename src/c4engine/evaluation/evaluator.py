"""
Multifaceted evaluation suite combining human and automated assessment
of retrieval effectiveness and generative faithfulness.
"""

from typing import List, Dict, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class EvaluationResult:
    """Results from evaluation."""
    metric_name: str
    score: float
    details: Dict[str, Any]


class RAGEvaluator:
    """
    Comprehensive evaluation suite for RAG systems that assesses:
    1. Retrieval effectiveness
    2. Generative faithfulness
    3. Combined human and automated metrics
    """
    
    def __init__(
        self,
        enable_human_eval: bool = False,
        enable_llm_eval: bool = True,
    ):
        """
        Initialize the evaluator.
        
        Args:
            enable_human_eval: Whether to enable human evaluation interface
            enable_llm_eval: Whether to use LLM-based evaluation
        """
        self.enable_human_eval = enable_human_eval
        self.enable_llm_eval = enable_llm_eval
        self.evaluation_results = []
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        retrieved_docs: List[List[Dict]],
        ground_truth: List[List[int]],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval effectiveness with standard IR metrics.
        
        Args:
            queries: List of query strings
            retrieved_docs: List of retrieved document lists per query
            ground_truth: List of relevant document IDs per query
            k_values: Values of k for evaluation
            
        Returns:
            Dictionary of retrieval metrics
        """
        metrics = defaultdict(list)
        
        for query, retrieved, relevant in zip(queries, retrieved_docs, ground_truth):
            relevant_set = set(relevant)
            
            for k in k_values:
                retrieved_k = retrieved[:k]
                retrieved_ids = []
                for doc in retrieved_k:
                    doc_id = doc.get('doc_id') or doc.get('id')
                    if doc_id is not None:
                        retrieved_ids.append(doc_id)
                retrieved_set = set(retrieved_ids)
                
                # Precision@k
                hits = len(relevant_set & retrieved_set)
                precision = hits / k if k > 0 else 0
                metrics[f"precision@{k}"].append(precision)
                
                # Recall@k
                recall = hits / len(relevant_set) if relevant_set else 0
                metrics[f"recall@{k}"].append(recall)
                
                # F1@k
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0
                metrics[f"f1@{k}"].append(f1)
            
            # MRR (Mean Reciprocal Rank)
            mrr = 0.0
            for i, doc in enumerate(retrieved):
                doc_id = doc.get('doc_id') or doc.get('id')
                if doc_id is not None and doc_id in relevant_set:
                    mrr = 1.0 / (i + 1)
                    break
            metrics["mrr"].append(mrr)
            
            # MAP (Mean Average Precision)
            ap = self._calculate_average_precision(retrieved, relevant_set)
            metrics["map"].append(ap)
            
            # NDCG
            ndcg = self._calculate_ndcg(retrieved, relevant_set)
            metrics["ndcg"].append(ndcg)
        
        # Average all metrics
        results = {
            metric: np.mean(values) for metric, values in metrics.items()
        }
        
        result = EvaluationResult(
            metric_name="retrieval_effectiveness",
            score=results.get("ndcg", 0.0),
            details=results,
        )
        self.evaluation_results.append(result)
        
        return results
    
    def evaluate_generation(
        self,
        questions: List[str],
        generated_answers: List[str],
        reference_answers: Optional[List[str]] = None,
        contexts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate generative faithfulness and quality.
        
        Args:
            questions: List of questions
            generated_answers: List of generated answers
            reference_answers: Optional reference answers for comparison
            contexts: Optional contexts used for generation
            
        Returns:
            Dictionary of generation metrics
        """
        metrics = {}
        
        # 1. Faithfulness to context
        if contexts:
            faithfulness_scores = []
            for answer, context in zip(generated_answers, contexts):
                score = self._evaluate_faithfulness(answer, context)
                faithfulness_scores.append(score)
            metrics["faithfulness"] = np.mean(faithfulness_scores)
        
        # 2. Answer relevance to question
        relevance_scores = []
        for question, answer in zip(questions, generated_answers):
            score = self._evaluate_relevance(question, answer)
            relevance_scores.append(score)
        metrics["relevance"] = np.mean(relevance_scores)
        
        # 3. Answer completeness
        completeness_scores = []
        for answer in generated_answers:
            score = self._evaluate_completeness(answer)
            completeness_scores.append(score)
        metrics["completeness"] = np.mean(completeness_scores)
        
        # 4. Reference-based metrics (if available)
        if reference_answers:
            # ROUGE scores
            rouge_scores = self._calculate_rouge(generated_answers, reference_answers)
            metrics.update(rouge_scores)
            
            # BLEU score
            bleu_score = self._calculate_bleu(generated_answers, reference_answers)
            metrics["bleu"] = bleu_score
            
            # Semantic similarity
            if self.enable_llm_eval:
                sem_sim = self._calculate_semantic_similarity(
                    generated_answers, reference_answers
                )
                metrics["semantic_similarity"] = sem_sim
        
        # 5. LLM-based evaluation (if enabled)
        if self.enable_llm_eval:
            llm_scores = self._llm_based_evaluation(
                questions, generated_answers, contexts
            )
            metrics.update(llm_scores)
        
        result = EvaluationResult(
            metric_name="generation_quality",
            score=metrics.get("relevance", 0.0),
            details=metrics,
        )
        self.evaluation_results.append(result)
        
        return metrics
    
    def evaluate_end_to_end(
        self,
        queries: List[str],
        generated_answers: List[str],
        retrieved_docs: List[List[Dict]],
        ground_truth_docs: List[List[int]],
        reference_answers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the complete RAG pipeline end-to-end.
        
        Args:
            queries: Query strings
            generated_answers: Generated answers
            retrieved_docs: Retrieved documents per query
            ground_truth_docs: Ground truth relevant documents
            reference_answers: Optional reference answers
            
        Returns:
            Combined evaluation metrics
        """
        # Retrieval metrics
        retrieval_metrics = self.evaluate_retrieval(
            queries, retrieved_docs, ground_truth_docs
        )
        
        # Generation metrics
        contexts = [
            " ".join([doc.get("doc_text", str(doc))[:200] for doc in docs[:3]])
            for docs in retrieved_docs
        ]
        generation_metrics = self.evaluate_generation(
            queries, generated_answers, reference_answers, contexts
        )
        
        # Combined metrics
        combined = {
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
            "overall_score": self._calculate_overall_score(
                retrieval_metrics, generation_metrics
            ),
        }
        
        return combined
    
    def human_evaluation_interface(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[str],
        save_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create an interface for human evaluation.
        
        Args:
            questions: Questions to evaluate
            answers: Generated answers
            contexts: Contexts used
            save_path: Path to save evaluation results
            
        Returns:
            List of human evaluation records
        """
        if not self.enable_human_eval:
            print("Human evaluation not enabled")
            return []
        
        evaluation_items = []
        
        for i, (question, answer, context) in enumerate(zip(questions, answers, contexts)):
            item = {
                "id": i,
                "question": question,
                "answer": answer,
                "context": context[:500],
                "evaluation": {
                    "relevance": None,  # 1-5 scale
                    "accuracy": None,  # 1-5 scale
                    "completeness": None,  # 1-5 scale
                    "clarity": None,  # 1-5 scale
                    "comments": "",
                },
            }
            evaluation_items.append(item)
        
        # Save template for human evaluation
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(evaluation_items, f, indent=2)
            print(f"Human evaluation template saved to {save_path}")
        
        return evaluation_items
    
    def process_human_evaluations(
        self,
        evaluation_path: str,
    ) -> Dict[str, float]:
        """
        Process completed human evaluations.
        
        Args:
            evaluation_path: Path to completed evaluation JSON
            
        Returns:
            Aggregated human evaluation metrics
        """
        with open(evaluation_path, 'r') as f:
            evaluations = json.load(f)
        
        metrics = defaultdict(list)
        
        for item in evaluations:
            eval_data = item.get("evaluation", {})
            for key, value in eval_data.items():
                if isinstance(value, (int, float)) and value is not None:
                    metrics[f"human_{key}"].append(value)
        
        # Average scores
        results = {
            metric: np.mean(values) for metric, values in metrics.items()
        }
        
        result = EvaluationResult(
            metric_name="human_evaluation",
            score=results.get("human_relevance", 0.0),
            details=results,
        )
        self.evaluation_results.append(result)
        
        return results
    
    # Helper methods for metric calculation
    
    def _calculate_average_precision(
        self,
        retrieved: List[Dict],
        relevant_set: set,
    ) -> float:
        """Calculate Average Precision."""
        if not relevant_set:
            return 0.0
        
        hits = 0
        sum_precisions = 0.0
        
        for i, doc in enumerate(retrieved):
            doc_id = doc.get('doc_id') or doc.get('id')
            if doc_id is not None and doc_id in relevant_set:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i
        
        return sum_precisions / len(relevant_set) if relevant_set else 0.0
    
    def _calculate_ndcg(
        self,
        retrieved: List[Dict],
        relevant_set: set,
        k: Optional[int] = None,
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if k:
            retrieved = retrieved[:k]
        
        # DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved):
            doc_id = doc.get('doc_id') or doc.get('id')
            rel = 1.0 if (doc_id is not None and doc_id in relevant_set) else 0.0
            dcg += rel / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        # IDCG
        ideal_rels = sorted([1.0] * len(relevant_set), reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Evaluate if the answer is faithful to the context.
        Simple heuristic: check overlap and no hallucination.
        """
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Check overlap
        overlap = len(answer_words & context_words) / len(answer_words) if answer_words else 0
        
        return min(overlap * 1.2, 1.0)  # Boost overlap score
    
    def _evaluate_relevance(self, question: str, answer: str) -> float:
        """Evaluate answer relevance to question."""
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Simple relevance: word overlap and answer length
        overlap = len(question_words & answer_words) / len(question_words) if question_words else 0
        length_score = min(len(answer) / 100, 1.0)  # Prefer non-trivial answers
        
        return (overlap + length_score) / 2
    
    def _evaluate_completeness(self, answer: str) -> float:
        """Evaluate answer completeness."""
        # Heuristic: check for length, structure, and detail
        length_score = min(len(answer) / 200, 1.0)
        
        # Check for sentence structure
        sentences = answer.count('.') + answer.count('?') + answer.count('!')
        structure_score = min(sentences / 3, 1.0)
        
        return (length_score + structure_score) / 2
    
    def _calculate_rouge(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            scores = defaultdict(list)
            for pred, ref in zip(predictions, references):
                result = scorer.score(ref, pred)
                for key, value in result.items():
                    scores[key].append(value.fmeasure)
            
            return {key: np.mean(values) for key, values in scores.items()}
        
        except ImportError:
            print("rouge-score not installed, skipping ROUGE metrics")
            return {}
    
    def _calculate_bleu(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """Calculate BLEU score (simplified)."""
        # Simple unigram BLEU
        scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            overlap = len(pred_words & ref_words)
            score = overlap / len(pred_words) if pred_words else 0
            scores.append(score)
        
        return np.mean(scores)
    
    def _calculate_semantic_similarity(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """Calculate semantic similarity using embeddings."""
        try:
            from sentence_transformers import SentenceTransformer, util
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            pred_embeddings = model.encode(predictions, convert_to_tensor=True)
            ref_embeddings = model.encode(references, convert_to_tensor=True)
            
            similarities = util.cos_sim(pred_embeddings, ref_embeddings)
            return float(similarities.diagonal().mean())
        
        except ImportError:
            print("sentence-transformers not installed, skipping semantic similarity")
            return 0.0
    
    def _llm_based_evaluation(
        self,
        questions: List[str],
        answers: List[str],
        contexts: Optional[List[str]],
    ) -> Dict[str, float]:
        """
        Use LLM to evaluate answer quality.
        
        Note: This is a placeholder implementation. For production use,
        implement actual LLM-based evaluation using a model to score
        answers for quality, coherence, accuracy, etc.
        """
        # TODO: Implement actual LLM-based evaluation
        # For now, return empty dict to indicate not implemented
        return {}
    
    def _calculate_overall_score(
        self,
        retrieval_metrics: Dict[str, float],
        generation_metrics: Dict[str, float],
    ) -> float:
        """Calculate overall RAG system score."""
        # Weighted combination
        retrieval_score = retrieval_metrics.get("ndcg", 0.0)
        generation_score = generation_metrics.get("relevance", 0.0)
        
        # 40% retrieval, 60% generation
        overall = 0.4 * retrieval_score + 0.6 * generation_score
        
        return overall
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        summary = {
            "num_evaluations": len(self.evaluation_results),
            "results": [
                {
                    "metric": result.metric_name,
                    "score": result.score,
                    "details": result.details,
                }
                for result in self.evaluation_results
            ],
        }
        return summary
