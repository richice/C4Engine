
import os
import re
import time
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

try:
    import tiktoken
except Exception:  # tiktoken optional; DataExtractor falls back to char/4
    tiktoken = None

from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.callbacks import get_openai_callback

# --- Paths / config -----------------------------------------------------------
BASE_PATH = 
EMBEDDING_MODEL_PATH = 

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- Model initialization ---------------------------

def _load_env():
    # Kept for any other keys in .env; Azure values above take precedence.
    load_dotenv(dotenv_path=BASE_PATH / ".env", verbose=False)
    os.environ["AZURE_OPENAI_ENDPOINT"] =  
    os.environ["AZURE_OPENAI_API_KEY"] = 


_load_env()

llmaz =
llmaz4 =
# Encoder on CPU: the answer-relevance cosine similarity is tiny, and keeping it off
# the GPU avoids VRAM contention with the Ollama generator during a run.
embed = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH, model_kwargs={"device": "cpu"})


class ModelType(Enum):
    GPT = "gpt"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    QIANFAN = "qianfan"


# --- BaseEvaluator (notebook cell 10, GPT path only) --------------------------
class BaseEvaluator:
    """Base class: model init + a GPT invocation wrapper with token tracking."""

    def __init__(self, model_type: ModelType = ModelType.GPT, model_name: str = "gpt-4.1"):
        self.model_type = model_type
        self.model_name = model_name
        self.llm_client = self._initialize_model()
        logger.info(f"Initialized {type(self).__name__} with {model_type.value} model: {model_name}")

    def _initialize_model(self):
        model_map = {
            "gpt-4.1": llmaz4,
        }
        return model_map.get(self.model_name, llmaz)

    def _call_model(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        with get_openai_callback() as cb:
            response = self.llm_client.invoke(prompt)
            duration = time.time() - start_time
            return {
                "content": response.content if hasattr(response, "content") else str(response),
                "duration": duration,
                "tokens": cb.total_tokens if cb else "N/A",
                "cost": cb.total_cost if cb else 0.0,
            }


# --- FaithfulnessEvaluator (notebook cell 14) ---------------------------------
class FaithfulnessEvaluator(BaseEvaluator):
    """Faithfulness = |supported statements| / |statements|."""

    def extract_statements(self, question: str, answer: str) -> Tuple[List[str], Dict]:
        prompt = f"""Given a question and answer, create one or more statements from each sentence in the given answer.

question: {question}
answer: {answer}

Note: The answer is based on both the question and a comprehensive case spreadsheet containing accident data. Extract factual claims that can be verified."""
        try:
            result = self._call_model(prompt)
            statements = self._parse_statements(result["content"])
            return statements, {"statement_count": len(statements)}
        except Exception as e:
            logger.error(f"Error in statement extraction: {e}")
            statements = [s.strip() for s in answer.split(".") if s.strip() and len(s.strip()) > 10]
            return statements, {"error": str(e)}

    def _parse_statements(self, response: str) -> List[str]:
        statements = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            line = re.sub(r"^statement:?\s*", "", line, flags=re.IGNORECASE)
            line = line.strip().strip("*").strip()  # drop markdown bold/heading markers
            if not line or len(line) <= 10:
                continue
            low = line.lower()
            # Skip meta/boilerplate lines that are not factual claims.
            if low.startswith(("note:", "question:", "answer:", "certainly", "here are",
                                "sure,", "of course", "summary of factual claims",
                                "based on the analysis")):
                continue
            statements.append(line)
        return statements

    def verify_statements(self, statements: List[str], context: str) -> Tuple[List[Dict], Dict]:
        if not statements:
            return [], {}
        statements_text = "\n".join([f"statement: {stmt}" for stmt in statements])
        # NOTE: Faithfulness uses the ORIGINAL RAGAS prompt (a "Faithfulness2" variant
        # was tried but re-sampling the statement extraction made scores unstable and
        # occasionally inverted the arm ordering, so v1 faithfulness values are kept).
        # Only Context-Relevance got a v2 prompt (see extract_relevant_sentences).
        prompt = f"""Consider the given context and following statements, then determine whether they are supported by the information present in the context. Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.

Context: {context}

{statements_text}"""
        try:
            result = self._call_model(prompt)
            verdicts = self._parse_verdicts(result["content"], statements)
            return verdicts, {"supported_count": sum(1 for v in verdicts if v["verdict"])}
        except Exception as e:
            logger.error(f"Error in statement verification: {e}")
            verdicts = [{"statement": s, "verdict": True, "explanation": "Error"} for s in statements]
            return verdicts, {"error": str(e)}

    def _parse_verdicts(self, response: str, statements: List[str]) -> List[Dict]:
        """Parse Yes/No verdicts from the verification response.

        The judge (gpt-4.1) returns one `Verdict: Yes/No` line per statement, in order
        (usually preceded by `Explanation: ...`). The robust parse is therefore to
        pull the ordered sequence of `verdict: (yes|no)` markers and align them to the
        statements. We fall back to a looser per-statement scan only if the count of
        explicit verdict markers does not match the statement count.
        """
        # Primary: ordered "Verdict: Yes/No" markers.
        markers = re.findall(r"verdict\s*[:\-]?\s*(yes|no)\b", response, flags=re.IGNORECASE)

        if len(markers) == len(statements) and markers:
            return [
                {"statement": stmt, "verdict": markers[i].lower() == "yes", "explanation": ""}
                for i, stmt in enumerate(statements)
            ]

        # If we found some (but a mismatched number of) explicit verdicts, still use
        # them positionally as far as they go; default missing ones to supported=False.
        if markers:
            verdicts = []
            for i, stmt in enumerate(statements):
                v = markers[i].lower() == "yes" if i < len(markers) else False
                verdicts.append({"statement": stmt, "verdict": v, "explanation": ""})
            return verdicts

        # Fallback: no "Verdict:" markers at all — scan near each statement's text.
        response_lower = response.lower()
        verdicts = []
        for stmt in statements:
            verdict = False
            snippet = stmt[:30].lower()
            if snippet in response_lower:
                pos = response_lower.find(snippet)
                after = response_lower[pos:pos + 200]
                if re.search(r"\byes\b", after):
                    verdict = True
                elif re.search(r"\bno\b", after):
                    verdict = False
            verdicts.append({"statement": stmt, "verdict": verdict, "explanation": ""})
        return verdicts

    def calculate_faithfulness_score(self, verdicts: List[Dict]) -> float:
        if not verdicts:
            return 0.0
        V = sum(1 for v in verdicts if v["verdict"])
        return V / len(verdicts)

    def evaluate_single_item(self, question: str, answer: str, reference: str, item_id: str = None) -> Dict[str, Any]:
        statements, _ = self.extract_statements(question, answer)
        verdicts, _ = self.verify_statements(statements, reference)
        score = self.calculate_faithfulness_score(verdicts)
        return {
            "item_id": item_id,
            "faithfulness_score": score,
            "supported_count": sum(1 for v in verdicts if v["verdict"]),
            "total_statements": len(statements),
        }



# --- AnswerRelevanceEvaluator (notebook cell 17) ------------------------------
class AnswerRelevanceEvaluator(BaseEvaluator):
    """Answer Relevance = mean cosine similarity between the original question and
    n questions regenerated from the answer."""

    def __init__(self, model_type: ModelType = ModelType.GPT, model_name: str = "gpt-4.1", n_questions: int = 3):
        super().__init__(model_type, model_name)
        self.n_questions = n_questions
        self.embedding_model = embed

    def generate_questions(self, answer: str) -> Tuple[List[str], Dict]:
        generated_questions = []
        for _ in range(self.n_questions):
            prompt = f"""Generate a generalized question for the given answer.
answer: {answer}"""
            try:
                result = self._call_model(prompt)
                q = self._parse_generated_question(result["content"])
                if q:
                    generated_questions.append(q)
            except Exception as e:
                logger.error(f"Error generating question: {e}")
                continue
        return generated_questions, {"questions_generated": len(generated_questions)}

    def _parse_generated_question(self, response: str) -> str:
        response = response.strip()
        for prefix in ["question:", "generated question:", "q:", "the question is:"]:
            if response.lower().startswith(prefix):
                response = response[len(prefix):].strip()
        if response and not response.endswith("?"):
            response += "?"
        return response

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        return np.array(self.embedding_model.embed_documents(texts))

    def calculate_similarity(self, e1: np.ndarray, e2: np.ndarray) -> float:
        if e1.ndim == 1:
            e1 = e1.reshape(1, -1)
        if e2.ndim == 1:
            e2 = e2.reshape(1, -1)
        return float(cosine_similarity(e1, e2)[0][0])

    def calculate_answer_relevance_score(self, original_question: str, generated_questions: List[str]) -> Tuple[float, List[float]]:
        if not generated_questions:
            return 0.0, []
        original_embedding = self.compute_embeddings([original_question])[0]
        generated_embeddings = self.compute_embeddings(generated_questions)
        similarities = [self.calculate_similarity(original_embedding, g) for g in generated_embeddings]
        return sum(similarities) / len(similarities), similarities

    def evaluate_single_item(self, question: str, answer: str, reference: str, item_id: str = None) -> Dict[str, Any]:
        generated_questions, _ = self.generate_questions(answer)
        ar_score, similarities = self.calculate_answer_relevance_score(question, generated_questions)
        return {
            "item_id": item_id,
            "answer_relevance": {
                "score": ar_score,
                "generated_questions": generated_questions,
                "similarities": similarities,
                "n_questions": len(generated_questions),
            },
        }


# --- ContextRelevanceEvaluator (deterministic denominator) --------------------
class ContextRelevanceEvaluator(BaseEvaluator):
    @staticmethod
    def _answer_claims(answer: str) -> List[str]:
        # deterministic segmentation of the answer into non-trivial claim lines/sentences
        import re as _re
        parts = _re.split(r"(?<=[.!?])\s+|\n", answer)
        out = []
        for p in parts:
            p = _re.sub(r"^[\s*+\-\d.\)]+", "", p).strip()
            if len(p) > 15 and not p.lower().startswith(("based on", "here", "in conclusion", "note")):
                out.append(p)
        return out

    def supported_claim_indices(self, claims: List[str], context: str) -> set:
        """LLM returns indices of answer claims that are directly supported by the context."""
        if not claims:
            return set()
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        prompt = f"""You are given a context (statistical evidence from a spreadsheet) and a numbered list of claims taken from an answer. Output ONLY the numbers of the claims that are DIRECTLY supported by the context — i.e. the specific data, count, category, or comparison in the claim can be found in or derived from the context. Output the numbers as a comma-separated list (e.g. "1, 3, 4"), or the single word NONE if no claim is supported.

Context:
{context}

Claims:
{numbered}"""
        try:
            content = self._call_model(prompt)["content"]
        except Exception as e:
            logger.error(f"Error selecting supported claims: {e}")
            return set()
        if "none" in content.lower() and not re.search(r"\d", content):
            return set()
        nums = {int(x) for x in re.findall(r"\d+", content)}
        return {n for n in nums if 1 <= n <= len(claims)}  # numerator ⊆ denominator

    def evaluate_single_item(self, question: str, answer: str, reference: str, item_id: str = None) -> Dict[str, Any]:
        claims = self._answer_claims(answer)
        total = len(claims)
        supported = self.supported_claim_indices(claims, reference)
        score = (len(supported) / total) if total else 0.0
        return {
            "item_id": item_id,
            "context_relevance": {  # key kept for compatibility; metric = Context Utilization
                "score": score,
                "extracted_count": len(supported),
                "total_sentences": total,
            },
        }


# --- ContextRelevanceRatioEvaluator (ORIGINAL RAGAS, for Table A) -------------
class ContextRelevanceRatioEvaluator(BaseEvaluator):

    @staticmethod
    def _context_lines(context: str) -> List[str]:
        # objective segmentation of the context into non-trivial lines (the denominator)
        import re as _re
        out = []
        for ln in (context or "").split("\n"):
            ln = _re.sub(r"^[\s*+\-]+", "", ln).strip()
            # keep substantive lines only: has a digit or >=3 words (drops blank/rule lines)
            if len(ln) >= 8 and (_re.search(r"\d", ln) or len(ln.split()) >= 3):
                out.append(ln)
        return out

    def relevant_line_indices(self, question: str, lines: List[str]) -> set:
        """LLM returns the indices of context lines relevant to answering the question."""
        if not lines:
            return set()
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(lines))
        prompt = f"""You are given a question and a numbered list of context lines (statistical evidence / records). Output ONLY the numbers of the lines that are RELEVANT to answering the question — i.e. a line that provides information usable to answer it. Output the numbers as a comma-separated list (e.g. "1, 3, 4"), or the single word NONE if no line is relevant.

Question: {question}

Context lines:
{numbered}"""
        try:
            content = self._call_model(prompt)["content"]
        except Exception as e:
            logger.error(f"Error selecting relevant context lines: {e}")
            return set()
        if "none" in content.lower() and not re.search(r"\d", content):
            return set()
        nums = {int(x) for x in re.findall(r"\d+", content)}
        return {n for n in nums if 1 <= n <= len(lines)}  # numerator ⊆ denominator

    def evaluate_single_item(self, question: str, answer: str, reference: str, item_id: str = None) -> Dict[str, Any]:
        lines = self._context_lines(reference)
        total = len(lines)
        relevant = self.relevant_line_indices(question, lines)
        score = (len(relevant) / total) if total else 0.0
        return {
            "item_id": item_id,
            "context_relevance": {  # ORIGINAL RAGAS ratio (relevant/total)
                "score": score,
                "extracted_count": len(relevant),
                "total_sentences": total,
            },
        }


# --- DataExtractor: text preprocessing (notebook cell 5, trimmed) -------------
class DataExtractor:
    """Cleans table-border/whitespace noise out of the JSON text fields before
    scoring. Faithful subset of the notebook's DataExtractor."""

    def __init__(self):
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base") if tiktoken else None
        except Exception:
            self.tokenizer = None

    def _clean_table_formatting(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r"[│┌┐└┘├┤┬┴┼─═║╔╗╚╝╠╣╦╩╬]", "", text)
        text = re.sub(r"[|]{2,}", "", text)
        text = re.sub(r"^[|\s]*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[\s│|]+", "", text, flags=re.MULTILINE)
        text = re.sub(r"[\s│|]+$", "", text, flags=re.MULTILINE)
        return text

    def _optimize_spacing(self, text: str) -> str:
        if not text:
            return text
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def preprocess(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        return self._optimize_spacing(self._clean_table_formatting(text))


# --- Public convenience API ---------------------------------------------------
def score_record(question: str, answer: str, reference: str, item_id: str = None,
                 model_name: str = "gpt-4.1", faith_context: str = None) -> Dict[str, float]:
    
    extractor = DataExtractor()
    q = extractor.preprocess(question)
    a = extractor.preprocess(answer)
    r = extractor.preprocess(reference)
    fr = extractor.preprocess(faith_context) if faith_context is not None else r

    faith = FaithfulnessEvaluator(model_name=model_name).evaluate_single_item(q, a, fr, item_id)
    ans = AnswerRelevanceEvaluator(model_name=model_name).evaluate_single_item(q, a, r, item_id)
    ctx = ContextRelevanceEvaluator(model_name=model_name).evaluate_single_item(q, a, r, item_id)

    return {
        "faithfulness": round(float(faith["faithfulness_score"]), 4),
        "answer_relevance": round(float(ans["answer_relevance"]["score"]), 4),
        "context_relevance": round(float(ctx["context_relevance"]["score"]), 4),
    }


def score_record_owncontext(question: str, answer: str, reference: str, item_id: str = None,
                            model_name: str = "gpt-4.1") -> Dict[str, float]:
    """TABLE A — standard/textbook RAGAS, every arm judged on its OWN context.

    Unlike `score_record` (which can pass a shared `faith_context` and uses the redefined
    Context Utilization), this entry point makes NO metric changes and uses NO shared
    context:
      - Faithfulness  = FaithfulnessEvaluator against the arm's own `reference`.
      - AnswerRel     = AnswerRelevanceEvaluator (context-free; identical to score_record).
      - ContextRel    = ContextRelevanceRatioEvaluator = original RAGAS |relevant|/|total|.

    This is the arm-scored-on-its-own-context view requested for Table A; it is reported
    ALONGSIDE (not replacing) the shared-context / Context-Utilization tables so the
    reviewer can see the textbook result and the reason we corrected it (naive-RAG parrots
    its own retrieved rows -> inflated own-context Faithfulness; the long two-stage context
    -> deflated original Context Relevance).
    """
    extractor = DataExtractor()
    q = extractor.preprocess(question)
    a = extractor.preprocess(answer)
    r = extractor.preprocess(reference)

    faith = FaithfulnessEvaluator(model_name=model_name).evaluate_single_item(q, a, r, item_id)
    ans = AnswerRelevanceEvaluator(model_name=model_name).evaluate_single_item(q, a, r, item_id)
    ctx = ContextRelevanceRatioEvaluator(model_name=model_name).evaluate_single_item(q, a, r, item_id)

    return {
        "faithfulness": round(float(faith["faithfulness_score"]), 4),
        "answer_relevance": round(float(ans["answer_relevance"]["score"]), 4),
        "context_relevance": round(float(ctx["context_relevance"]["score"]), 4),
    }


def score_json(json_path: str, model_name: str = "gpt-4.1") -> Dict[str, float]:
    """Load an output_*.json (fields question/answer/reference) and score it."""
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = data[0]
    return score_record(
        data.get("question", ""),
        data.get("answer", ""),
        data.get("reference", ""),
        item_id=Path(json_path).stem,
        model_name=model_name,
    )


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(p, "->", score_json(p))
