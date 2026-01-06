"""
Example: Basic usage of C4Engine for crane safety analysis.
"""

from c4engine import (
    EmbeddingRetrieverAdapter,
    SyntheticSupervisionGenerator,
    AgenticRAG,
    RAGEvaluator,
)
from c4engine.utils import load_config, setup_api_keys


def main():
    """Demonstrate basic C4Engine usage."""
    
    # Load configuration
    config = load_config()
    setup_api_keys(openai_key=config.get("openai_api_key"))
    
    # Sample crane safety documents
    documents = [
        "Crane operators must conduct pre-operation inspections including checking wire ropes, hooks, and safety devices before each use.",
        "Load capacity charts must be visible to operators and strictly followed. Never exceed the rated capacity of the crane.",
        "Wind speed restrictions: Crane operations should cease when wind speeds exceed 20 mph for mobile cranes.",
        "Proper rigging techniques are essential. Use appropriate slings and ensure load balance before lifting.",
        "Ground conditions must be assessed before crane setup. Outriggers must be fully extended on stable ground.",
    ]
    
    print("=" * 80)
    print("C4Engine Demo: Crane Safety Analysis")
    print("=" * 80)
    
    # Step 1: Generate synthetic supervision data
    print("\n[1] Generating synthetic supervision data...")
    syn_gen = SyntheticSupervisionGenerator(
        model=config.get("model", "gpt-4"),
        temperature=config.get("temperature", 0.7),
    )
    
    # Generate domain-specific queries
    queries = syn_gen.generate_domain_queries(
        domain_context="crane safety and operations",
        num_queries=5,
        domain_terms=["load capacity", "rigging", "wind speed", "inspection"],
    )
    
    print(f"Generated {len(queries)} domain-specific queries:")
    for i, query in enumerate(queries, 1):
        print(f"  {i}. {query}")
    
    # Generate synthetic training pairs
    print("\n[2] Generating synthetic query-document pairs...")
    synthetic_examples = syn_gen.generate_synthetic_pairs(
        documents=documents,
        domain_context="crane safety",
        num_examples=5,
    )
    
    print(f"Generated {len(synthetic_examples)} synthetic training examples")
    
    # Step 2: Build and adapt retriever
    print("\n[3] Building embedding-based retriever...")
    retriever = EmbeddingRetrieverAdapter(
        model_name=config.get("embedding_model", "all-MiniLM-L6-v2"),
        index_type="flat",
    )
    
    retriever.build_index(documents)
    print(f"Index built with {len(documents)} documents")
    
    # Adapt with synthetic data
    print("\n[4] Adapting retriever with synthetic supervision...")
    retriever.adapt_with_synthetic_data(
        synthetic_examples=synthetic_examples,
        learning_rate=0.001,
        epochs=2,
    )
    print("Adaptation complete")
    
    # Step 3: Test retrieval
    print("\n[5] Testing retrieval...")
    test_query = "What are the wind speed limits for crane operations?"
    results = retriever.retrieve(test_query, k=3)
    
    print(f"\nQuery: {test_query}")
    print("\nTop 3 retrieved documents:")
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. Score: {result.score:.3f}")
        print(f"     {result.doc_text[:100]}...")
    
    # Step 4: Agentic RAG
    print("\n[6] Running agentic RAG...")
    rag = AgenticRAG(
        retriever=retriever,
        model=config.get("model", "gpt-4"),
        max_iterations=3,
    )
    
    test_questions = [
        "What safety checks must be performed before operating a crane?",
        "What are the load capacity requirements?",
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        response = rag.query(question, top_k=3, enable_multi_step=True)
        
        print(f"Answer: {response['answer'][:200]}...")
        print(f"Reasoning steps: {len(response['reasoning_steps'])}")
        print(f"Document type: {response['context_type']}")
    
    # Step 5: Evaluation
    print("\n[7] Running evaluation suite...")
    evaluator = RAGEvaluator(enable_llm_eval=False)
    
    # Mock evaluation data
    retrieved_docs = [
        [{"doc_id": 0, "doc_text": documents[0]}, {"doc_id": 2, "doc_text": documents[2]}]
        for _ in test_questions
    ]
    ground_truth = [[0, 2], [1, 3]]
    
    retrieval_metrics = evaluator.evaluate_retrieval(
        queries=test_questions,
        retrieved_docs=retrieved_docs,
        ground_truth=ground_truth,
        k_values=[1, 3, 5],
    )
    
    print("\nRetrieval Metrics:")
    for metric, value in retrieval_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Generate mock answers for generation evaluation
    generated_answers = [response['answer'] for _ in test_questions]
    
    generation_metrics = evaluator.evaluate_generation(
        questions=test_questions,
        generated_answers=["Sample answer about crane safety checks.", 
                          "Sample answer about load capacity."],
        contexts=[" ".join([doc["doc_text"] for doc in docs]) for docs in retrieved_docs],
    )
    
    print("\nGeneration Metrics:")
    for metric, value in generation_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
