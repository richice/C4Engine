# C4Engine

Adapted EBR and agentic RAG for crane safety analysis

## Overview

C4Engine provides three key innovations for domain-specific information retrieval and generation:

1. **LLM-assisted synthetic supervision** - Adapt embedding-based retrievers (EBRs) to domain terminology without manual labels
2. **Agentic retrieval-augmented generation (RAG)** - Feature-aware analysis across long narratives and tabular data with multi-step reasoning
3. **Multifaceted evaluation suite** - Combined human and automated assessment of retrieval effectiveness and generative faithfulness

## Features

### 1. Synthetic Supervision for Domain Adaptation
- Generate training data using LLMs without manual annotation
- Adapt retrievers to domain-specific terminology
- Create query-document pairs with relevance labels
- Support for hard negative mining

### 2. Agentic RAG System
- Multi-step reasoning for complex queries
- Handles both narrative and tabular data
- Automatic document type detection
- Long document processing with intelligent chunking
- Action history tracking

### 3. Comprehensive Evaluation
- Retrieval metrics: Precision@k, Recall@k, MRR, MAP, NDCG
- Generation metrics: Faithfulness, Relevance, Completeness
- Reference-based metrics: ROUGE, BLEU, semantic similarity
- Human evaluation interface
- End-to-end pipeline evaluation

## Installation

```bash
# Clone the repository
git clone https://github.com/richice/C4Engine.git
cd C4Engine

# Install dependencies
pip install -e .
```

## Quick Start

```python
from c4engine import (
    EmbeddingRetrieverAdapter,
    SyntheticSupervisionGenerator,
    AgenticRAG,
    RAGEvaluator,
)

# 1. Generate synthetic supervision
generator = SyntheticSupervisionGenerator(model="gpt-4")
synthetic_examples = generator.generate_synthetic_pairs(
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
    queries=queries,
    generated_answers=answers,
    retrieved_docs=retrieved,
    ground_truth_docs=ground_truth,
)
```

## Configuration

Create a `.env` file:

```env
OPENAI_API_KEY=your-openai-key
MODEL=gpt-4
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Examples

See the `examples/` directory:

```bash
python examples/basic_usage.py
python examples/tabular_data_example.py
```

## Documentation

Comprehensive documentation is available in `docs/README.md`

## Use Cases

- **Crane Safety Analysis** - Compliance checking, incident analysis, operational procedures
- **Domain Adaptation** - Adapt to any domain without manual labeling
- **Long Document Analysis** - Process technical documents, regulations, manuals
- **Tabular Data QA** - Query structured data with natural language

## Architecture

```
C4Engine
├── retriever/          # Embedding-based retrieval with adaptation
│   ├── ebr_adapter.py
│   └── synthetic_supervision.py
├── rag/               # Agentic RAG system
│   └── agentic_rag.py
├── evaluation/        # Evaluation suite
│   └── evaluator.py
└── utils/            # Utilities and helpers
    ├── config.py
    └── data_processing.py
```

## License

MIT License

## Citation

```bibtex
@software{c4engine,
  title={C4Engine: Adapted EBR and Agentic RAG for Crane Safety Analysis},
  year={2024},
  url={https://github.com/richice/C4Engine}
}
```
