"""
Agentic RAG system for feature-aware analysis across long narratives and tabular data.
"""

from typing import List, Dict, Optional, Any, Union
import openai
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class DocumentType(Enum):
    """Types of documents the agentic RAG can process."""
    NARRATIVE = "narrative"
    TABULAR = "tabular"
    MIXED = "mixed"


@dataclass
class RAGContext:
    """Context for RAG operations."""
    retrieved_docs: List[Dict[str, Any]]
    query: str
    document_type: DocumentType
    metadata: Dict[str, Any]


@dataclass
class AgentAction:
    """Represents an action taken by the agent."""
    action_type: str
    reasoning: str
    parameters: Dict[str, Any]
    result: Optional[Any] = None


class AgenticRAG:
    """
    Agentic RAG system that performs feature-aware analysis across
    long narratives and tabular data with multi-step reasoning.
    """
    
    def __init__(
        self,
        retriever,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        max_iterations: int = 5,
        chunk_size: int = 2000,
    ):
        """
        Initialize the agentic RAG system.
        
        Args:
            retriever: Embedding retriever instance
            model: LLM model for generation
            api_key: OpenAI API key
            max_iterations: Maximum reasoning iterations
            chunk_size: Size of chunks for long documents
        """
        self.retriever = retriever
        self.model = model
        self.max_iterations = max_iterations
        self.chunk_size = chunk_size
        
        if api_key:
            openai.api_key = api_key
        
        self.action_history: List[AgentAction] = []
    
    def query(
        self,
        question: str,
        document_type: Optional[DocumentType] = None,
        top_k: int = 5,
        enable_multi_step: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a query using agentic RAG.
        
        Args:
            question: User question
            document_type: Type of documents (auto-detect if None)
            top_k: Number of documents to retrieve
            enable_multi_step: Enable multi-step reasoning
            
        Returns:
            Dictionary with answer and metadata
        """
        self.action_history = []
        
        # Step 1: Retrieve relevant documents
        retrieval_action = AgentAction(
            action_type="retrieve",
            reasoning=f"Retrieving top-{top_k} relevant documents for query",
            parameters={"query": question, "k": top_k},
        )
        
        retrieved = self.retriever.retrieve(question, k=top_k)
        retrieval_action.result = retrieved
        self.action_history.append(retrieval_action)
        
        # Step 2: Analyze document types
        if document_type is None:
            document_type = self._detect_document_type(retrieved)
        
        # Step 3: Process based on document type
        if document_type == DocumentType.TABULAR:
            return self._process_tabular(question, retrieved, enable_multi_step)
        elif document_type == DocumentType.NARRATIVE:
            return self._process_narrative(question, retrieved, enable_multi_step)
        else:  # MIXED
            return self._process_mixed(question, retrieved, enable_multi_step)
    
    def _detect_document_type(self, documents: List) -> DocumentType:
        """Auto-detect document type from retrieved documents."""
        # Simple heuristic: check for table-like patterns
        tabular_indicators = 0
        narrative_indicators = 0
        
        for doc in documents[:3]:  # Check first 3 docs
            text = doc.doc_text.lower()
            
            # Tabular indicators
            if any(indicator in text for indicator in ["|", "\t", "column", "row", "table"]):
                tabular_indicators += 1
            
            # Narrative indicators
            if any(indicator in text for indicator in ["the ", "and ", "paragraph", "section"]):
                narrative_indicators += 1
        
        if tabular_indicators > narrative_indicators:
            return DocumentType.TABULAR
        elif narrative_indicators > tabular_indicators:
            return DocumentType.NARRATIVE
        else:
            return DocumentType.MIXED
    
    def _process_narrative(
        self,
        question: str,
        retrieved: List,
        enable_multi_step: bool,
    ) -> Dict[str, Any]:
        """
        Process narrative documents with long-form text analysis.
        
        Args:
            question: User question
            retrieved: Retrieved documents
            enable_multi_step: Whether to use multi-step reasoning
            
        Returns:
            Answer dictionary
        """
        # Chunk long documents
        chunked_docs = []
        for doc in retrieved:
            chunks = self._chunk_long_text(doc.doc_text)
            chunked_docs.extend([(chunk, doc.doc_id, doc.score) for chunk in chunks])
        
        # Sort by relevance score
        chunked_docs.sort(key=lambda x: x[2], reverse=True)
        
        if enable_multi_step:
            return self._multi_step_reasoning(question, chunked_docs, "narrative")
        else:
            return self._single_step_generation(question, chunked_docs[:5])
    
    def _process_tabular(
        self,
        question: str,
        retrieved: List,
        enable_multi_step: bool,
    ) -> Dict[str, Any]:
        """
        Process tabular data with structured analysis.
        
        Args:
            question: User question
            retrieved: Retrieved documents
            enable_multi_step: Whether to use multi-step reasoning
            
        Returns:
            Answer dictionary
        """
        # Extract tabular structure
        tables = []
        for doc in retrieved:
            table_data = self._extract_table_data(doc.doc_text)
            if table_data:
                tables.append((table_data, doc.doc_id, doc.score))
        
        if enable_multi_step:
            return self._multi_step_reasoning(question, tables, "tabular")
        else:
            return self._single_step_generation(question, tables[:3])
    
    def _process_mixed(
        self,
        question: str,
        retrieved: List,
        enable_multi_step: bool,
    ) -> Dict[str, Any]:
        """
        Process mixed narrative and tabular documents.
        
        Args:
            question: User question
            retrieved: Retrieved documents
            enable_multi_step: Whether to use multi-step reasoning
            
        Returns:
            Answer dictionary
        """
        # Separate narrative and tabular
        narrative_docs = []
        tabular_docs = []
        
        for doc in retrieved:
            if self._is_tabular(doc.doc_text):
                tabular_docs.append(doc)
            else:
                narrative_docs.append(doc)
        
        # Process both types
        context_parts = []
        
        if narrative_docs:
            for doc in narrative_docs[:3]:
                context_parts.append(f"[Narrative] {doc.doc_text[:500]}")
        
        if tabular_docs:
            for doc in tabular_docs[:2]:
                context_parts.append(f"[Tabular] {doc.doc_text[:500]}")
        
        if enable_multi_step:
            return self._multi_step_reasoning(question, context_parts, "mixed")
        else:
            return self._single_step_generation(question, context_parts)
    
    def _chunk_long_text(self, text: str) -> List[str]:
        """Chunk long text into manageable pieces."""
        chunks = []
        words = text.split()
        
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            
            if current_length >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks if chunks else [text]
    
    def _extract_table_data(self, text: str) -> Optional[str]:
        """Extract and format tabular data from text."""
        # Simple extraction - in practice, would use more sophisticated parsing
        if "|" in text or "\t" in text:
            return text
        return None
    
    def _is_tabular(self, text: str) -> bool:
        """Check if text contains tabular data."""
        indicators = ["|", "\t", "column", "row"]
        return any(ind in text.lower() for ind in indicators)
    
    def _multi_step_reasoning(
        self,
        question: str,
        context: List,
        context_type: str,
    ) -> Dict[str, Any]:
        """
        Perform multi-step reasoning to answer complex questions.
        
        Args:
            question: User question
            context: Context information
            context_type: Type of context
            
        Returns:
            Answer with reasoning steps
        """
        reasoning_steps = []
        current_context = context
        
        for iteration in range(self.max_iterations):
            # Plan next step
            plan_action = AgentAction(
                action_type="plan",
                reasoning=f"Planning step {iteration + 1}",
                parameters={"iteration": iteration, "question": question},
            )
            
            next_step = self._plan_next_step(
                question, current_context, reasoning_steps
            )
            plan_action.result = next_step
            self.action_history.append(plan_action)
            reasoning_steps.append(next_step)
            
            # Check if we have enough information
            if self._should_stop_reasoning(reasoning_steps):
                break
        
        # Generate final answer
        answer = self._generate_final_answer(question, context, reasoning_steps)
        
        return {
            "answer": answer,
            "reasoning_steps": reasoning_steps,
            "context_type": context_type,
            "iterations": len(reasoning_steps),
            "action_history": [
                {"type": a.action_type, "reasoning": a.reasoning}
                for a in self.action_history
            ],
        }
    
    def _plan_next_step(
        self,
        question: str,
        context: List,
        previous_steps: List[str],
    ) -> str:
        """Plan the next reasoning step."""
        context_str = "\n".join([str(c)[:200] for c in context[:3]])
        steps_str = "\n".join(previous_steps) if previous_steps else "None yet"
        
        prompt = f"""Question: {question}

Available context (truncated):
{context_str}

Previous reasoning steps:
{steps_str}

What is the next logical step to answer this question? Be specific and focused.
Return only the next step, not the full answer."""

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are analyzing information step by step."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200,
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error in planning: {e}"
    
    def _should_stop_reasoning(self, steps: List[str]) -> bool:
        """Determine if reasoning should stop."""
        if len(steps) >= self.max_iterations:
            return True
        
        # Check if recent steps indicate completion
        if len(steps) >= 2:
            last_step = steps[-1].lower()
            if any(keyword in last_step for keyword in ["sufficient", "enough", "complete", "answer"]):
                return True
        
        return False
    
    def _generate_final_answer(
        self,
        question: str,
        context: List,
        reasoning_steps: List[str],
    ) -> str:
        """Generate the final answer using all context and reasoning."""
        context_str = "\n\n".join([str(c)[:500] for c in context[:5]])
        steps_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(reasoning_steps))
        
        prompt = f"""Question: {question}

Reasoning steps:
{steps_str}

Relevant context:
{context_str}

Based on the reasoning steps and context, provide a comprehensive, accurate answer to the question.
Be specific, cite relevant details from the context, and ensure factual accuracy."""

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant providing accurate answers based on given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def _single_step_generation(
        self,
        question: str,
        context: List,
    ) -> Dict[str, Any]:
        """Generate answer in a single step."""
        context_str = "\n\n".join([str(c)[:500] for c in context[:5]])
        
        prompt = f"""Question: {question}

Context:
{context_str}

Provide a clear, accurate answer based on the context above."""

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "reasoning_steps": ["Single-step generation"],
                "context_type": "auto",
                "iterations": 1,
                "action_history": [
                    {"type": a.action_type, "reasoning": a.reasoning}
                    for a in self.action_history
                ],
            }
        except Exception as e:
            return {
                "answer": f"Error: {e}",
                "reasoning_steps": [],
                "context_type": "error",
                "iterations": 0,
                "action_history": [],
            }
