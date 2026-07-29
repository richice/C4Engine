"""
================================================================================
Table Lens  --  Sample Python Script
================================================================================
A didactic, illustration-only implementation of Algorithm 1: Table Lens.

    Input : S in R^{n x m}  (a spreadsheet)
            q               (a user query)
    Output: O_json          (structured column-level evidence)
            O_text          (concatenated natural-language overview)

Each function below is annotated with the algorithm line it corresponds to,
so a reader can map pseudocode -> Python -> prompt template in one glance.

NOTE
----
This file is NOT meant to run as-is. File paths, API keys, deployment names
and domain-specific column names are placeholders (`<...>`). Replace them
before executing in a real environment.
================================================================================
"""

# ============================================================
# 0. Imports & placeholder configuration
# ============================================================
import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai import AzureChatOpenAI


# ---- Placeholders (replace with real values) --------------------------------
EXCEL_PATH        = "<PATH/TO/SPREADSHEET.xlsx>"
OUTPUT_JSON_PATH  = "<PATH/TO/table_lens_output.json>"

os.environ["AZURE_OPENAI_API_KEY"]  = "<YOUR_AZURE_KEY>"
os.environ["AZURE_OPENAI_ENDPOINT"] = "<YOUR_AZURE_ENDPOINT>"

# LLM used for vertical column selection (Algorithm line 3)
llm = AzureChatOpenAI(
    azure_deployment="<DEPLOYMENT_NAME>",
    api_version="<API_VERSION>",
)


# ============================================================
# 1. GetColumnList(S)                                 [line 1]
#    Produce column-level metadata: name, intro, type,
#    missing count, distinct count, top values, numeric stats.
# ============================================================
def is_empty_value(value) -> bool:
    """Values considered null/empty for counting missing entries."""
    if value is None:
        return True
    if isinstance(value, float) and (np.isnan(value) or value == 0.0):
        return True
    if isinstance(value, int) and value == 0:
        return True
    if isinstance(value, str) and value.strip().lower() in ("", "-", "nan"):
        return True
    return False


def generate_intro(column_name: str, col_type: str) -> str:
    """
    Rule-based short description for a column.
    Used later inside TemplateToSentence (line 19) so the sentence
    doesn't read as a bare list of numbers.
    """
    name = str(column_name).lower().strip()

    # ---- Illustrative rules; the real system typically has many more ----
    if name == "<COLUMN_A>":
        return "Description of column A ..."
    if name == "<COLUMN_B>":
        return "Description of column B ..."
    if re.match(r"\d+ <PATTERN_NAME>", name):
        return "Description shared by all columns matching this pattern."

    return f"Generic description for column '{column_name}'."


def generate_overview(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    C_all <- GetColumnList(S)                        [line 1]

    Returns a dict keyed by column name with:
        id, name, intro, type, num_missing, num_unmissing,
        top_values, metadata (numeric stats when applicable).
    """
    overview: Dict[str, Dict[str, Any]] = {}
    for idx, col in enumerate(df.columns, start=1):
        series = df[col]
        col_type      = str(series.dtype)
        num_missing   = int(series.isna().sum())
        num_unmissing = int(series.notna().sum())
        top_values    = series.value_counts(dropna=True).head(10).to_dict()

        metadata: Dict[str, Any] = {}
        if col_type in ("int64", "float64"):
            metadata = {
                "min":  float(series.min())  if num_unmissing else None,
                "max":  float(series.max())  if num_unmissing else None,
                "mean": float(series.mean()) if num_unmissing else None,
                "std":  float(series.std())  if num_unmissing else None,
            }

        overview[col] = {
            "id":            idx,
            "name":          col,
            "intro":         generate_intro(col, col_type),
            "type":          col_type,
            "num_missing":   num_missing,
            "num_unmissing": num_unmissing,
            "top_values":    top_values,
            "metadata":      metadata,
        }
    return overview


def prune_overview(overview: Dict, exclude_ids: List[int]) -> Dict:
    """
    Housekeeping for GetColumnList: drop columns that are
    (a) all-null / all-zero numeric, or (b) explicitly excluded.
    """
    pruned = {}
    for col, info in overview.items():
        stats = info.get("metadata", {})
        if stats and all(
            v is None or v == 0 or (isinstance(v, float) and np.isnan(v))
            for v in stats.values()
        ):
            continue
        if info["id"] in exclude_ids:
            continue
        pruned[col] = info
    return pruned


# ============================================================
# 2. RuleBasedRowFilter(S, q)                         [line 4]
#    Horizontal filtering. Each rule has the form
#        "<rule_type>:<keywords>"
#    rule_type: 1 = KEEP any, 2 = KEEP all,
#               3 = REMOVE any, 4 = REMOVE all
#    Keywords wrapped in "..." require an exact word match.
# ============================================================
def _parse_keywords(keywords_str: str) -> List[Tuple[str, bool]]:
    """Return list of (keyword, exact_match) tuples."""
    out: List[Tuple[str, bool]] = []
    for raw in keywords_str.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        exact = raw.startswith('"') and raw.endswith('"')
        kw    = raw.strip('"').lower()
        out.append((kw, exact))
    return out


def _row_matches(row_str: str,
                 keywords: List[Tuple[str, bool]],
                 mode: str) -> bool:
    """`mode` is either 'any' or 'all'."""
    text = row_str.lower()
    hits = []
    for kw, exact in keywords:
        if exact:
            hits.append(bool(re.search(rf"\b{re.escape(kw)}\b", text)))
        else:
            hits.append(kw in text)
    return any(hits) if mode == "any" else all(hits)


def apply_keyword_filters(df: pd.DataFrame,
                          rules_dict: Dict[int, str]) -> pd.DataFrame:
    """
    R_rel <- RuleBasedRowFilter(S, q)                [line 4]

    Apply the ordered filtering rules and return the surviving rows.
    """
    current = df.copy()
    for rule_idx in sorted(rules_dict.keys()):
        rule_type, kw_str = rules_dict[rule_idx].split(":", 1)
        keywords = _parse_keywords(kw_str)
        row_texts = current.astype(str).agg(" ".join, axis=1)

        if rule_type == "1":                          # KEEP any
            mask =  row_texts.apply(lambda t: _row_matches(t, keywords, "any"))
        elif rule_type == "2":                        # KEEP all
            mask =  row_texts.apply(lambda t: _row_matches(t, keywords, "all"))
        elif rule_type == "3":                        # REMOVE any
            mask = ~row_texts.apply(lambda t: _row_matches(t, keywords, "any"))
        elif rule_type == "4":                        # REMOVE all
            mask = ~row_texts.apply(lambda t: _row_matches(t, keywords, "all"))
        else:
            continue

        current = current[mask]
    return current


# ============================================================
# 3. BuildSchemaPrompt + LLM_SelectColumns          [lines 2-3]
#    Vertical selection: ask the LLM which columns are
#    relevant to answering the user query.
# ============================================================
class NumberedListParser(BaseOutputParser[List[str]]):
    """Parse a numbered list ('1. foo\\n2. bar') into ['foo', 'bar']."""
    def parse(self, text: str) -> List[str]:
        out = []
        for line in text.strip().split("\n"):
            m = re.match(r"^\d+\.\s*(.*)", line)
            if m:
                out.append(m.group(1).strip())
        return out


# ---- Prompt template: P_schema <- BuildSchemaPrompt(C_all, q) ----
COLUMN_PROMPT = PromptTemplate(
    input_variables=["query", "background_info", "column_names"],
    template="""
You are a highly experienced <DOMAIN> expert with 20 years of expertise.
Your task is to think step by step and identify the columns that will best
help answer the user's question.

---
**User's Question:**
"{query}"

**Background Information (column intros):**
{background_info}

**Available Columns:**
{column_names}
---

**Instructions:**
- Include only column names that appear verbatim in the list above.
- Output as a numbered list, one column per line, no extra text.
- Return at most the top 6 most relevant columns.
- Reason step by step before writing the final numbered list.

**Examples:**

Question: "<EXAMPLE_QUESTION_1>"
Thought 1: This is a time-related question -> look at date/time columns.
Observation 1: [<COLUMN_X>, <COLUMN_Y>] are time-related.
Result:
1. <COLUMN_X>
2. <COLUMN_Y>

Question: "<EXAMPLE_QUESTION_2>"
Thought 1: This concerns event categorization -> look at type columns.
Observation 1: [<COLUMN_A>, <COLUMN_B>, <COLUMN_C>] are type-related.
Result:
1. <COLUMN_A>
2. <COLUMN_B>
3. <COLUMN_C>
"""
)


def llm_select_columns(query: str, overview: Dict) -> List[str]:
    """
    P_schema <- BuildSchemaPrompt(C_all, q)          [line 2]
    C_rel    <- LLM_SelectColumns(P_schema)          [line 3]
    """
    column_names    = [info["name"] for info in overview.values()]
    background_info = "\n".join(
        f"- {info['name']}: {info['intro']}" for info in overview.values()
    )
    chain = COLUMN_PROMPT | llm | NumberedListParser()
    return chain.invoke({
        "query":           query,
        "background_info": background_info,
        "column_names":    column_names,
    })


# ============================================================
# 4. Per-column analysis                           [lines 5-19]
#    For each c_j in C_rel: infer type, count missing/distinct,
#    compute stats, then build o_j and its sentence s_j.
# ============================================================
def infer_type(values: pd.Series) -> str:
    """tau_j <- InferType(V_j) in {numeric, categorical, datetime, free-text}."""
    if pd.api.types.is_numeric_dtype(values):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(values):
        return "datetime"
    nunique = values.nunique(dropna=True)
    # Small cardinality -> categorical; otherwise treat as free-text
    if nunique <= max(20, int(0.05 * max(len(values), 1))):
        return "categorical"
    return "free-text"


def count_missing(values: pd.Series) -> int:
    """m_j <- CountMissing(V_j)                       [line 8]"""
    return int(values.isna().sum() + values.apply(is_empty_value).sum())


def count_distinct(values: pd.Series) -> int:
    """u_j <- CountDistinct(V_j)                      [line 9]"""
    return int(values.dropna().nunique())


def top_k_categories(values: pd.Series, k: int = 5) -> Dict[str, int]:
    """stats_j <- TopKCategories(V_j, K)             [line 13]"""
    counts = values.dropna().value_counts()
    return {str(key): int(v) for key, v in counts.head(k).items()}


def basic_summary(values: pd.Series) -> Dict[str, Any]:
    """stats_j <- BasicSummary(V_j)                  [line 15]"""
    non_null = values.dropna().astype(str)
    return {
        "sample":     non_null.head(3).tolist(),
        "avg_length": float(non_null.str.len().mean()) if len(non_null) else 0.0,
    }


# ---- Fixed templates for TemplateToSentence ----   [line 19]
NUMERIC_TMPL = (
    "Column {id} [{name}] ({intro}) is numeric: "
    "{missing} missing / {distinct} distinct values; "
    "min={min}, max={max}, mean={mean:.2f}, std={std:.2f}."
)

CATEGORICAL_TMPL = (
    "Column {id} [{name}] ({intro}) is categorical: "
    "{missing} missing / {distinct} distinct. "
    "Top categories: {topk}."
)

DATETIME_TMPL = (
    "Column {id} [{name}] ({intro}) is datetime: "
    "{missing} missing / {distinct} distinct. Range spans {sample}."
)

FREETEXT_TMPL = (
    "Column {id} [{name}] ({intro}) is free-text: "
    "{missing} missing / {distinct} distinct. Example values: {sample}."
)


def template_to_sentence(o_j: Dict[str, Any]) -> str:
    """
    s_j <- TemplateToSentence(o_j)                   [line 19]
    Fixed-template verbalization -- keeps the natural-language
    overview O_text consistent and easy to consume downstream.
    """
    t = o_j["type"]
    common = dict(
        id=o_j["id"], name=o_j["name"], intro=o_j["intro"],
        missing=o_j["missing"], distinct=o_j["distinct"],
    )
    stats = o_j["stats"]

    if t == "numeric":
        return NUMERIC_TMPL.format(**common, **stats)

    if t == "categorical":
        topk_str = ", ".join(f"{k}({v})" for k, v in stats.items())
        return CATEGORICAL_TMPL.format(**common, topk=topk_str)

    if t == "datetime":
        return DATETIME_TMPL.format(**common, sample=stats.get("sample", []))

    return FREETEXT_TMPL.format(**common, sample=stats.get("sample", []))


# ============================================================
# 5. Table Lens driver                             [Algorithm 1]
# ============================================================
def table_lens(df: pd.DataFrame,
               query: str,
               rules: Optional[Dict[int, str]] = None,
               exclude_ids: Optional[List[int]] = None,
               top_k: int = 5) -> Tuple[List[Dict], str]:
    """
    End-to-end Algorithm 1: Table Lens.

    Parameters
    ----------
    df           : the spreadsheet S
    query        : the user query q
    rules        : keyword filtering rules for RuleBasedRowFilter (line 4)
    exclude_ids  : column ids to skip after GetColumnList (line 1 cleanup)
    top_k        : K for TopKCategories (line 13)

    Returns
    -------
    O_json : list of per-column evidence dicts               (line 22)
    O_text : concatenated natural-language column overview   (line 22)
    """
    O_json:       List[Dict[str, Any]] = []                       # init
    O_text_parts: List[str]            = []                       # init

    # ---- line 1: C_all <- GetColumnList(S) --------------------
    C_all = generate_overview(df)
    C_all = prune_overview(C_all, exclude_ids or [])

    # ---- lines 2-3: schema prompt + LLM column selection ------
    C_rel = llm_select_columns(query, C_all)

    # ---- line 4: R_rel <- RuleBasedRowFilter(S, q) ------------
    R_rel = apply_keyword_filters(df, rules or {})

    # ---- lines 5-19: per-column analysis loop -----------------
    for c_j in C_rel:                                             # line 5
        if c_j not in df.columns:
            continue

        V_j   = R_rel[c_j] if c_j in R_rel.columns else df[c_j]   # line 6
        tau_j = infer_type(V_j)                                   # line 7
        m_j   = count_missing(V_j)                                # line 8
        u_j   = count_distinct(V_j)                               # line 9

        if tau_j == "numeric":                                    # lines 10-11
            v = V_j.dropna()
            stats_j = {
                "min":  float(v.min())  if len(v) else 0.0,
                "max":  float(v.max())  if len(v) else 0.0,
                "mean": float(v.mean()) if len(v) else 0.0,
                "std":  float(v.std())  if len(v) else 0.0,
            }
        elif tau_j == "categorical":                              # lines 12-13
            stats_j = top_k_categories(V_j, k=top_k)
        else:                                                     # lines 14-15
            stats_j = basic_summary(V_j)

        info_c = C_all.get(c_j, {})
        o_j = {                                                   # lines 16-17
            "id":       info_c.get("id"),
            "name":     c_j,
            "intro":    info_c.get("intro", ""),
            "type":     tau_j,
            "missing":  m_j,
            "distinct": u_j,
            "stats":    stats_j,
        }
        O_json.append(o_j)                                        # line 17
        O_text_parts.append(template_to_sentence(o_j))            # lines 18-19

    # ---- lines 20-22: Concat + return -------------------------
    O_text = "\n".join(O_text_parts)
    return O_json, O_text


# ============================================================
# 6. Example usage (illustrative -- placeholders throughout)
# ============================================================
if __name__ == "__main__":
    df_S = pd.read_excel(EXCEL_PATH)

    user_query = "<USER_QUERY_HERE>"

    # Rules: keys give order, values are "<rule_type>:<;-separated keywords>"
    # Wrap a keyword in double quotes for exact word matching.
    example_rules = {
        1: '1:<KEYWORD_1>;"<EXACT_KEYWORD>";<KEYWORD_2>',   # KEEP any
        2: '3:<UNWANTED_KEYWORD_1>;<UNWANTED_KEYWORD_2>',   # REMOVE any
    }

    O_json, O_text = table_lens(
        df=df_S,
        query=user_query,
        rules=example_rules,
        exclude_ids=[1, 9, 10, 11, 12],   # columns to drop after GetColumnList
        top_k=5,
    )

    # Persist structured evidence
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(O_json, f, indent=2, ensure_ascii=False, default=str)

    # Print natural-language overview
    print("=== Table Lens: column-level evidence overview ===")
    print(O_text)
