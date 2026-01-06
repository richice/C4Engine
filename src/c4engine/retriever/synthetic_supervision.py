"""
Synthetic supervision generator using LLM for domain terminology adaptation.
"""

from typing import List, Dict, Optional, Any
import openai
from dataclasses import dataclass
import json


@dataclass
class SyntheticExample:
    """Represents a synthetically generated training example."""
    query: str
    positive_doc: str
    negative_docs: List[str]
    metadata: Dict[str, Any]


class SyntheticSupervisionGenerator:
    """
    Generates synthetic supervision data using LLM to adapt retrievers
    to domain-specific terminology without manual labels.
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
    ):
        """
        Initialize the synthetic supervision generator.
        
        Args:
            model: LLM model to use for generation
            api_key: OpenAI API key (optional, can use env var)
            temperature: Temperature for generation (higher = more diverse)
        """
        self.model = model
        self.temperature = temperature
        if api_key:
            openai.api_key = api_key
    
    def generate_domain_queries(
        self,
        domain_context: str,
        num_queries: int = 10,
        domain_terms: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate domain-specific queries using LLM.
        
        Args:
            domain_context: Description of the domain (e.g., crane safety)
            num_queries: Number of queries to generate
            domain_terms: Optional list of domain-specific terms to incorporate
            
        Returns:
            List of generated queries
        """
        terms_section = ""
        if domain_terms:
            terms_section = f"\nKey domain terms to use: {', '.join(domain_terms)}"
        
        prompt = f"""You are an expert in {domain_context}.
Generate {num_queries} diverse and realistic queries that someone might ask when searching for information in this domain.
{terms_section}

The queries should:
1. Use domain-specific terminology naturally
2. Vary in complexity and specificity
3. Cover different aspects of the domain
4. Be practical questions that need answers

Return the queries as a JSON array of strings."""

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a domain expert helping generate training data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
            )
            
            content = response.choices[0].message.content.strip()
            # Parse JSON from response
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            queries = json.loads(content)
            return queries[:num_queries]
        
        except Exception as e:
            print(f"Error generating queries: {e}")
            return []
    
    def generate_synthetic_pairs(
        self,
        documents: List[str],
        domain_context: str,
        num_examples: int = 20,
        hard_negatives: bool = True,
    ) -> List[SyntheticExample]:
        """
        Generate synthetic query-document pairs for training.
        
        Args:
            documents: List of documents in the domain
            domain_context: Description of the domain
            num_examples: Number of synthetic examples to generate
            hard_negatives: Whether to generate hard negative examples
            
        Returns:
            List of synthetic examples with queries and relevance labels
        """
        examples = []
        
        for i, doc in enumerate(documents[:num_examples]):
            # Generate query for this document
            query = self._generate_query_for_document(doc, domain_context)
            
            # Select negative examples
            negatives = []
            for j, neg_doc in enumerate(documents):
                if j != i and len(negatives) < 3:
                    negatives.append(neg_doc)
            
            example = SyntheticExample(
                query=query,
                positive_doc=doc,
                negative_docs=negatives,
                metadata={
                    "doc_index": i,
                    "domain": domain_context,
                    "hard_negatives": hard_negatives,
                }
            )
            examples.append(example)
        
        return examples
    
    def _generate_query_for_document(
        self,
        document: str,
        domain_context: str,
    ) -> str:
        """
        Generate a query that would retrieve the given document.
        
        Args:
            document: The document to generate a query for
            domain_context: Description of the domain
            
        Returns:
            Generated query string
        """
        # Truncate document if too long
        doc_preview = document[:500] if len(document) > 500 else document
        
        prompt = f"""Given this document from the {domain_context} domain:

"{doc_preview}"

Generate a natural, realistic query that someone would ask to find this information.
The query should:
1. Use domain terminology appropriately
2. Be specific enough to match this document
3. Sound like a real user question
4. Be concise (1-2 sentences max)

Return only the query text, no explanation."""

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are generating training data for information retrieval."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating query: {e}")
            return ""
    
    def adapt_to_domain_terminology(
        self,
        source_terms: List[str],
        target_domain: str,
        num_variations: int = 5,
    ) -> Dict[str, List[str]]:
        """
        Generate domain-specific variations of terms for adaptation.
        
        Args:
            source_terms: General terms to adapt
            target_domain: Target domain context
            num_variations: Number of variations per term
            
        Returns:
            Dictionary mapping source terms to domain variations
        """
        adaptations = {}
        
        for term in source_terms:
            prompt = f"""In the context of {target_domain}, provide {num_variations} domain-specific variations or related terms for: "{term}"

These should be:
1. Commonly used in {target_domain}
2. Semantically related to the original term
3. Technically accurate
4. Actually used in practice

Return as a JSON array of strings."""

            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": f"You are a {target_domain} expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                )
                
                content = response.choices[0].message.content.strip()
                if content.startswith("```json"):
                    content = content.split("```json")[1].split("```")[0].strip()
                elif content.startswith("```"):
                    content = content.split("```")[1].split("```")[0].strip()
                
                variations = json.loads(content)
                adaptations[term] = variations[:num_variations]
            
            except Exception as e:
                print(f"Error adapting term '{term}': {e}")
                adaptations[term] = []
        
        return adaptations
