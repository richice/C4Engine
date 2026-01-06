"""
Embedding-based retriever adapter using synthetic supervision.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Represents a retrieval result."""
    doc_id: int
    doc_text: str
    score: float
    metadata: Dict


class EmbeddingRetrieverAdapter:
    """
    Adapts embedding-based retrievers to domain terminology using
    synthetic supervision without manual labels.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_type: str = "flat",
        device: str = "cpu",
    ):
        """
        Initialize the embedding retriever adapter.
        
        Args:
            model_name: Name of the sentence transformer model
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.index_type = index_type
        self.device = device
        
        # Load sentence transformer model
        self.encoder = SentenceTransformer(model_name)
        if device == "cuda":
            self.encoder = self.encoder.cuda()
        
        # Initialize variables
        self.index = None
        self.documents = []
        self.doc_embeddings = None
        self.doc_metadata = []
    
    def build_index(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 32,
    ):
        """
        Build the retrieval index from documents.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            batch_size: Batch size for encoding
        """
        self.documents = documents
        self.doc_metadata = metadata if metadata else [{} for _ in documents]
        
        # Encode documents
        print(f"Encoding {len(documents)} documents...")
        self.doc_embeddings = self.encoder.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        
        # Build FAISS index
        dimension = self.doc_embeddings.shape[1]
        
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, min(100, len(documents) // 10))
            self.index.train(self.doc_embeddings)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Add embeddings to index
        self.index.add(self.doc_embeddings)
        print(f"Index built with {self.index.ntotal} vectors")
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = True,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            return_scores: Whether to include similarity scores
            
        Returns:
            List of retrieval results
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):  # Valid index
                result = RetrievalResult(
                    doc_id=int(idx),
                    doc_text=self.documents[idx],
                    score=float(1.0 / (1.0 + dist)),  # Convert distance to similarity
                    metadata=self.doc_metadata[idx],
                )
                results.append(result)
        
        return results
    
    def adapt_with_synthetic_data(
        self,
        synthetic_examples: List,
        learning_rate: float = 0.001,
        epochs: int = 3,
    ):
        """
        Adapt the retriever using synthetic supervision data.
        
        This method fine-tunes the embedding model on synthetic examples
        to better capture domain-specific terminology.
        
        Args:
            synthetic_examples: List of SyntheticExample objects
            learning_rate: Learning rate for adaptation
            epochs: Number of training epochs
        """
        from sentence_transformers import losses, InputExample
        from torch.utils.data import DataLoader
        
        # Convert to InputExample format
        train_examples = []
        for example in synthetic_examples:
            # Positive pair
            train_examples.append(
                InputExample(texts=[example.query, example.positive_doc], label=1.0)
            )
            
            # Negative pairs
            for neg_doc in example.negative_docs[:2]:  # Use top 2 negatives
                train_examples.append(
                    InputExample(texts=[example.query, neg_doc], label=0.0)
                )
        
        # Create dataloader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        
        # Define loss
        train_loss = losses.CosineSimilarityLoss(self.encoder)
        
        # Train
        print(f"Adapting model with {len(train_examples)} examples for {epochs} epochs...")
        self.encoder.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            show_progress_bar=True,
        )
        
        print("Adaptation complete. Rebuilding index...")
        # Rebuild index with adapted model
        if self.documents:
            self.build_index(self.documents, self.doc_metadata)
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        relevant_docs: List[List[int]],
        k: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            queries: List of query strings
            relevant_docs: List of lists of relevant document indices per query
            k: Number of documents to retrieve
            
        Returns:
            Dictionary of evaluation metrics
        """
        total_recall = 0.0
        total_precision = 0.0
        total_mrr = 0.0
        
        for query, relevant in zip(queries, relevant_docs):
            results = self.retrieve(query, k=k)
            retrieved_ids = [r.doc_id for r in results]
            
            # Calculate metrics
            relevant_set = set(relevant)
            retrieved_set = set(retrieved_ids)
            
            hits = len(relevant_set & retrieved_set)
            precision = hits / len(retrieved_ids) if retrieved_ids else 0
            recall = hits / len(relevant_set) if relevant_set else 0
            
            # MRR
            mrr = 0.0
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_set:
                    mrr = 1.0 / (i + 1)
                    break
            
            total_precision += precision
            total_recall += recall
            total_mrr += mrr
        
        n_queries = len(queries)
        return {
            "precision@k": total_precision / n_queries,
            "recall@k": total_recall / n_queries,
            "mrr": total_mrr / n_queries,
        }
    
    def save_index(self, path: str):
        """Save the FAISS index to disk."""
        if self.index is not None:
            faiss.write_index(self.index, path)
    
    def load_index(self, path: str):
        """Load a FAISS index from disk."""
        self.index = faiss.read_index(path)
