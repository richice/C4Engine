"""
Example: Working with tabular crane safety data.
"""

import pandas as pd
from c4engine import EmbeddingRetrieverAdapter, AgenticRAG
from c4engine.utils import merge_tabular_and_narrative, load_config


def main():
    """Demonstrate handling of tabular crane safety data."""
    
    config = load_config()
    
    # Create sample tabular data
    crane_incidents = pd.DataFrame({
        "Date": ["2023-01-15", "2023-02-20", "2023-03-10"],
        "Crane_Type": ["Mobile", "Tower", "Mobile"],
        "Incident_Type": ["Load drop", "Wind damage", "Tip over"],
        "Wind_Speed_mph": [25, 35, 15],
        "Load_kg": [5000, 3000, 8000],
        "Cause": ["Exceeded capacity", "High winds", "Unstable ground"],
    })
    
    # Convert tabular data to documents
    documents = []
    for _, row in crane_incidents.iterrows():
        doc = f"Incident on {row['Date']}: {row['Crane_Type']} crane experienced {row['Incident_Type']}. "
        doc += f"Wind speed was {row['Wind_Speed_mph']} mph with load of {row['Load_kg']} kg. "
        doc += f"Cause: {row['Cause']}"
        documents.append(doc)
    
    # Add narrative safety guidelines
    documents.extend([
        "Wind speed limits: Operations must stop when winds exceed 20 mph for mobile cranes, 30 mph for tower cranes.",
        "Load capacity must never be exceeded. Check manufacturer specifications and load charts.",
        "Ground stability is critical. Conduct soil analysis and use proper outrigger support.",
    ])
    
    print("=" * 80)
    print("C4Engine: Tabular Data Analysis")
    print("=" * 80)
    
    # Build retriever
    print("\n[1] Building retriever with mixed narrative and tabular data...")
    retriever = EmbeddingRetrieverAdapter()
    retriever.build_index(documents)
    print(f"Indexed {len(documents)} documents (3 from table, 3 narratives)")
    
    # Create agentic RAG
    print("\n[2] Creating agentic RAG system...")
    rag = AgenticRAG(retriever=retriever, model=config.get("model", "gpt-4"))
    
    # Query with tabular awareness
    queries = [
        "What are the common causes of crane incidents?",
        "What wind speed limits should be followed?",
        "What was the incident with the highest load?",
    ]
    
    print("\n[3] Running queries with feature-aware analysis...")
    for query in queries:
        print(f"\nQuery: {query}")
        response = rag.query(query, top_k=3, enable_multi_step=True)
        print(f"Answer: {response['answer'][:300]}...")
        print(f"Document type detected: {response['context_type']}")
    
    # Demonstrate table merging
    print("\n[4] Merging tabular and narrative data...")
    narrative = "Historical incident analysis shows patterns in crane failures."
    merged = merge_tabular_and_narrative(crane_incidents, narrative)
    print(f"Merged document length: {len(merged)} characters")
    print(f"Preview:\n{merged[:200]}...")
    
    print("\n" + "=" * 80)
    print("Tabular data demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
