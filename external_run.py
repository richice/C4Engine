import re
import numpy as np

from langchain_core.output_parsers import StrOutputParser

from pipeline_X import check_X_relevance


# Short natural-language prototype per modality; the router scores the query against these.
_MODALITY_PROTOTYPES = {
    "row": "individual crane accident case records: specific incidents, cases, and events",
    "colstat": "per-column statistical summaries: value counts, frequencies, and distributions across all cases",
}


def _cosine_matrix(query_vec, doc_matrix):
    """Row-wise cosine similarity between one query vector and a (n, d) doc matrix."""
    q = np.asarray(query_vec, dtype=float)
    M = np.asarray(doc_matrix, dtype=float)
    qn = np.linalg.norm(q) or 1.0
    Mn = np.linalg.norm(M, axis=1)
    Mn[Mn == 0] = 1.0
    return (M @ q) / (Mn * qn)


def _topk_from_modality(query_vec, chunk_texts, embed, k):
    """Embed a modality's chunks and return [(score, text), ...] for its top-k by cosine."""
    if not chunk_texts:
        return []
    doc_vecs = embed.embed_documents(chunk_texts)
    sims = _cosine_matrix(query_vec, doc_vecs)
    order = np.argsort(-sims)[:k]
    return [(float(sims[i]), chunk_texts[i]) for i in order]


def run_mmrag_style_arm(qid, X_df, query, embed_base, llm, llmaz4, top_k=6, row_only=False):
    """mmRAG-protocol reimplementation on the X corpus (see module docstring).

    embed_base : BAAI/bge-large-en-v1.5 (mmRAG's default retriever), = load_models_lite()'s embed_base.
    llm        : llama3.1 (llmlm) — same generator as naive-rag.
    llmaz4     : Azure gpt-4.1, for check_X_relevance only.

    Returns (answer, reference, content, relevance).
    """
    # Imported here (not at module top) to avoid a circular import: ablation_run imports this
    # module, and NAIVE_RAG_PROMPT/serialize_* live in ablation_run.
    from ablation_run import NAIVE_RAG_PROMPT, serialize_X_rows, serialize_X_columns

    query_vec = embed_base.embed_query(query)

    # --- Build the heterogeneous chunk pool (modalities) ---------------------------------
    row_chunks = [c[0] for c in serialize_X_rows(X_df)]
    modalities = {"row": row_chunks}

    used_modalities = ["row"]
    if not row_only:
        col_chunks_meta, err = serialize_X_columns(qid)
        if err:
            # q2/q6/q9/q10/q14: no column-stats JSON. Degrade to row-only rather than crash;
            # the caller/notebook records this. `mmrag-style` == `mmrag-style-rowonly` for these.
            print(f"[q{qid}/mmrag-style] modality B unavailable ({err}); using row modality only.")
        else:
            modalities["colstat"] = [c[0] for c in col_chunks_meta]
            used_modalities.append("colstat")

    # --- Semantic weighted router: route the query across modalities ---------------------
    # Score each modality's prototype so we can report where the query was routed (mmRAG's
    # router concept), then do weighted retrieval: top-k from each modality, keep global top-k.
    proto_texts = [_MODALITY_PROTOTYPES[m] for m in used_modalities]
    proto_sims = _cosine_matrix(query_vec, embed_base.embed_documents(proto_texts))
    routing = {m: round(float(s), 4) for m, s in zip(used_modalities, proto_sims)}

    pooled = []  # (score, text, modality)
    for m in used_modalities:
        for score, text in _topk_from_modality(query_vec, modalities[m], embed_base, top_k):
            pooled.append((score, text, m))
    pooled.sort(key=lambda t: -t[0])
    selected = pooled[:top_k]

    provenance = {m: sum(1 for _, _, mm in selected if mm == m) for m in used_modalities}
    print(f"[q{qid}/mmrag-style{'-rowonly' if row_only else ''}] "
          f"routing={routing}  selected top-{top_k} provenance={provenance}")

    context = "\n\n".join(text for _, text, _ in selected)

    # --- Generation (identical to naive-rag) ---------------------------------------------
    answer_ori = (NAIVE_RAG_PROMPT | llm | StrOutputParser()).invoke(
        {"query": query, "context": context}
    )
    answer = re.sub(r"<think>.*?</think>", "", answer_ori, flags=re.DOTALL).strip()
    relevance = check_X_relevance(answer, llmaz4, query)

    return answer, context, answer, relevance
