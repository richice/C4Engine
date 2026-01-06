# C4Engine Documentation

**C4Engine: Adapted EBR and agentic RAG for crane safety analysis**

## Overview

C4Engine is a comprehensive system that combines three key innovations for domain-specific information retrieval and generation:

1. **LLM-assisted synthetic supervision** to adapt embedding-based retrievers (EBRs) to domain terminology without manual labels
2. **Agentic retrieval-augmented generation (RAG)** for feature-aware analysis across long narratives and tabular data
3. **A multifaceted evaluation suite** combining human and automated assessment of retrieval effectiveness and generative faithfulness

## Installation

```bash
pip install -e .
```

Or install requirements directly:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from c4engine import (
    EmbeddingRetrieverAdapter,
    SyntheticSupervisionGenerator,
    AgenticRAG,
    RAGEvaluator,
)

# 1. Generate synthetic supervision data
syn_gen = SyntheticSupervisionGenerator(model="gpt-4")
synthetic_examples = syn_gen.generate_synthetic_pairs(
    documents=your_documents,
    domain_context="crane safety",
)

# 2. Build and adapt retriever
retriever = EmbeddingRetrieverAdapter()
retriever.build_index(your_documents)
retriever.adapt_with_synthetic_data(synthetic_examples)

# 3. Run agentic RAG
rag = AgenticRAG(retriever=retriever)
response = rag.query("What are the safety requirements?")

# 4. Evaluate
evaluator = RAGEvaluator()
metrics = evaluator.evaluate_end_to_end(
    queries=test_queries,
    generated_answers=answers,
    retrieved_docs=retrieved,
    ground_truth_docs=ground_truth,
)
```

## Architecture

### 1. LLM-Assisted Synthetic Supervision

The `SyntheticSupervisionGenerator` uses large language models to create training data for adapting retrievers to domain-specific terminology without requiring manual annotation.

**Key Features:**
- Generates domain-specific queries
- Creates query-document pairs with relevance labels
- Adapts terminology to target domains
- Supports hard negative mining

**Example:**
```python
generator = SyntheticSupervisionGenerator(model="gpt-4")

# Generate queries for a domain
queries = generator.generate_domain_queries(
    domain_context="crane safety",
    num_queries=10,
    domain_terms=["load capacity", "rigging", "inspection"]
)

# Generate training pairs
examples = generator.generate_synthetic_pairs(
    documents=documents,
    domain_context="crane safety",
    num_examples=20,
    hard_negatives=True,
)

# Adapt terminology
adaptations = generator.adapt_to_domain_terminology(
    source_terms=["check", "limit", "procedure"],
    target_domain="crane safety operations",
)
```

### 2. Embedding-Based Retriever Adapter

The `EmbeddingRetrieverAdapter` provides a flexible retrieval system that can be adapted to domain terminology using synthetic supervision.

**Key Features:**
- Multiple FAISS index types (flat, IVF, HNSW)
- Fine-tuning on synthetic data
- Efficient vector search
- Evaluation metrics (Precision@k, Recall@k, MRR, MAP, NDCG)

**Example:**
```python
retriever = EmbeddingRetrieverAdapter(
    model_name="all-MiniLM-L6-v2",
    index_type="flat",
)

# Build index
retriever.build_index(documents, metadata=metadata)

# Adapt to domain
retriever.adapt_with_synthetic_data(
    synthetic_examples=examples,
    learning_rate=0.001,
    epochs=3,
)

# Retrieve
results = retriever.retrieve(query, k=5)
for result in results:
    print(f"Score: {result.score}, Doc: {result.doc_text}")

# Evaluate
metrics = retriever.evaluate_retrieval(
    queries=test_queries,
    relevant_docs=ground_truth,
    k=5,
)
```

### 3. Agentic RAG

The `AgenticRAG` system performs multi-step reasoning and handles both narrative and tabular data with feature-aware analysis.

**Key Features:**
- Automatic document type detection (narrative/tabular/mixed)
- Multi-step reasoning for complex queries
- Long document handling with intelligent chunking
- Tabular data extraction and processing
- Action history tracking

**Example:**
```python
rag = AgenticRAG(
    retriever=retriever,
    model="gpt-4",
    max_iterations=5,
    chunk_size=2000,
)

# Query with multi-step reasoning
response = rag.query(
    question="What safety procedures must be followed?",
    top_k=5,
    enable_multi_step=True,
)

print(response['answer'])
print(f"Reasoning steps: {response['reasoning_steps']}")
print(f"Document type: {response['context_type']}")
print(f"Iterations: {response['iterations']}")
```

### 4. Multifaceted Evaluation Suite

The `RAGEvaluator` provides comprehensive evaluation combining automated metrics and human assessment.

**Key Features:**
- Retrieval effectiveness metrics (Precision, Recall, MRR, MAP, NDCG)
- Generation quality metrics (Faithfulness, Relevance, Completeness)
- Reference-based metrics (ROUGE, BLEU, semantic similarity)
- LLM-based evaluation
- Human evaluation interface
- End-to-end evaluation

**Example:**
```python
evaluator = RAGEvaluator(
    enable_human_eval=True,
    enable_llm_eval=True,
)

# Evaluate retrieval
retrieval_metrics = evaluator.evaluate_retrieval(
    queries=queries,
    retrieved_docs=retrieved,
    ground_truth=ground_truth,
    k_values=[1, 3, 5, 10],
)

# Evaluate generation
generation_metrics = evaluator.evaluate_generation(
    questions=questions,
    generated_answers=answers,
    reference_answers=references,
    contexts=contexts,
)

# End-to-end evaluation
results = evaluator.evaluate_end_to_end(
    queries=queries,
    generated_answers=answers,
    retrieved_docs=retrieved,
    ground_truth_docs=ground_truth,
    reference_answers=references,
)

# Human evaluation
evaluator.human_evaluation_interface(
    questions=questions,
    answers=answers,
    contexts=contexts,
    save_path="human_eval.json",
)

# Process human evaluations
human_metrics = evaluator.process_human_evaluations("human_eval_completed.json")

# Get summary
summary = evaluator.get_summary()
```

## Configuration

Create a `.env` file with your API keys and settings:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
MODEL=gpt-4
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=2000
MAX_ITERATIONS=5
TEMPERATURE=0.7
```

Or use programmatically:

```python
from c4engine.utils import load_config, setup_api_keys

config = load_config()
setup_api_keys(openai_key=config["openai_api_key"])
```

## Utilities

### Data Processing

```python
from c4engine.utils import (
    load_documents_from_file,
    parse_tabular_data,
    chunk_narrative,
    extract_features_from_text,
)

# Load documents
docs = load_documents_from_file("data.txt", file_type="txt")

# Parse tables
table = parse_tabular_data(tabular_text)

# Chunk long narratives
chunks = chunk_narrative(long_text, chunk_size=500, overlap=50)

# Extract features
features = extract_features_from_text(text)
```

## Examples

See the `examples/` directory for complete working examples:

- `basic_usage.py` - Complete workflow demonstration
- `tabular_data_example.py` - Working with tabular data

Run examples:

```bash
python examples/basic_usage.py
python examples/tabular_data_example.py
```

## Use Cases

### Crane Safety Analysis

C4Engine was designed for crane safety analysis but can be adapted to any domain:

- Safety regulation compliance checking
- Incident analysis and prevention
- Operational procedure queries
- Training material generation

### Domain Adaptation

Adapt to new domains without manual labeling:

```python
# Generate domain-specific training data
synthetic = generator.generate_synthetic_pairs(
    documents=domain_docs,
    domain_context="your domain",
)

# Adapt retriever
retriever.adapt_with_synthetic_data(synthetic)
```

## API Reference

### Core Classes

- `SyntheticSupervisionGenerator`: Generate synthetic training data
- `EmbeddingRetrieverAdapter`: Adaptive embedding-based retrieval
- `AgenticRAG`: Multi-step reasoning RAG system
- `RAGEvaluator`: Comprehensive evaluation suite

### Data Classes

- `SyntheticExample`: Synthetic training example
- `RetrievalResult`: Retrieval result with metadata
- `RAGContext`: Context for RAG operations
- `AgentAction`: Agent action record
- `EvaluationResult`: Evaluation result with details

## Best Practices

1. **Synthetic Supervision**
   - Use diverse documents for synthetic pair generation
   - Include domain-specific terms in query generation
   - Generate sufficient examples (20+ for small domains)

2. **Retrieval Adaptation**
   - Start with base model (all-MiniLM-L6-v2)
   - Fine-tune for 2-3 epochs to avoid overfitting
   - Rebuild index after adaptation

3. **Agentic RAG**
   - Enable multi-step reasoning for complex queries
   - Adjust max_iterations based on query complexity
   - Use appropriate chunk_size for your documents

4. **Evaluation**
   - Combine automated and human evaluation
   - Use reference answers when available
   - Track metrics over time for improvements

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure `.env` file exists with valid keys
   - Check environment variables are loaded

2. **Memory Issues**
   - Use smaller batch sizes for encoding
   - Consider IVF or HNSW index for large datasets

3. **Slow Retrieval**
   - Use HNSW index for faster approximate search
   - Reduce embedding dimension if possible

4. **Poor Retrieval Quality**
   - Generate more synthetic examples
   - Increase fine-tuning epochs
   - Use domain-specific base model if available

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use C4Engine in your research, please cite:

```bibtex
@software{c4engine,
  title={C4Engine: Adapted EBR and Agentic RAG for Domain-Specific Analysis},
  author={C4Engine Team},
  year={2024},
  url={https://github.com/richice/C4Engine}
}
```

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/richice/C4Engine/issues
- Documentation: See this README and code docstrings

## Acknowledgments

Built with:
- OpenAI API for LLM capabilities
- Sentence Transformers for embeddings
- FAISS for efficient vector search
- LangChain for RAG components
