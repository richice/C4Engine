"""
Command-line interface for C4Engine.
"""

import argparse
import json
from pathlib import Path
from c4engine import (
    EmbeddingRetrieverAdapter,
    SyntheticSupervisionGenerator,
    AgenticRAG,
    RAGEvaluator,
)
from c4engine.utils import load_config, setup_api_keys, load_documents_from_file


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="C4Engine: Adapted EBR and agentic RAG"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Build retrieval index")
    index_parser.add_argument("--input", required=True, help="Input file with documents")
    index_parser.add_argument("--type", default="txt", help="File type (txt, csv, json)")
    index_parser.add_argument("--output", required=True, help="Output index path")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("--index", required=True, help="Path to index")
    query_parser.add_argument("--query", required=True, help="Query string")
    query_parser.add_argument("--k", type=int, default=5, help="Number of results")
    query_parser.add_argument("--multi-step", action="store_true", help="Enable multi-step reasoning")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate RAG system")
    eval_parser.add_argument("--queries", required=True, help="JSON file with queries")
    eval_parser.add_argument("--ground-truth", required=True, help="JSON file with ground truth")
    eval_parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    setup_api_keys(openai_key=config.get("openai_api_key"))
    
    if args.command == "index":
        build_index(args, config)
    elif args.command == "query":
        run_query(args, config)
    elif args.command == "evaluate":
        run_evaluation(args, config)
    else:
        parser.print_help()


def build_index(args, config):
    """Build retrieval index."""
    print(f"Loading documents from {args.input}...")
    documents = load_documents_from_file(args.input, args.type)
    print(f"Loaded {len(documents)} documents")
    
    print("Building index...")
    retriever = EmbeddingRetrieverAdapter(
        model_name=config.get("embedding_model", "all-MiniLM-L6-v2")
    )
    retriever.build_index(documents)
    
    print(f"Saving index to {args.output}...")
    retriever.save_index(args.output)
    print("Done!")


def run_query(args, config):
    """Run a query."""
    print(f"Loading index from {args.index}...")
    retriever = EmbeddingRetrieverAdapter()
    retriever.load_index(args.index)
    
    print("Creating RAG system...")
    rag = AgenticRAG(
        retriever=retriever,
        model=config.get("model", "gpt-4"),
    )
    
    print(f"\nQuery: {args.query}")
    print("Generating response...\n")
    
    response = rag.query(
        args.query,
        top_k=args.k,
        enable_multi_step=args.multi_step,
    )
    
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(response["answer"])
    print()
    
    if args.multi_step:
        print("REASONING STEPS")
        print("=" * 80)
        for i, step in enumerate(response["reasoning_steps"], 1):
            print(f"{i}. {step}")
        print()
    
    print(f"Document type: {response['context_type']}")
    print(f"Iterations: {response['iterations']}")


def run_evaluation(args, config):
    """Run evaluation."""
    print("Loading evaluation data...")
    with open(args.queries, 'r') as f:
        eval_data = json.load(f)
    
    with open(args.ground_truth, 'r') as f:
        ground_truth = json.load(f)
    
    print("Running evaluation...")
    evaluator = RAGEvaluator()
    
    # Evaluate (simplified - would need full pipeline)
    metrics = evaluator.evaluate_retrieval(
        queries=eval_data["queries"],
        retrieved_docs=eval_data.get("retrieved", []),
        ground_truth=ground_truth["relevant_docs"],
    )
    
    print("\nEvaluation Results:")
    print("=" * 80)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
