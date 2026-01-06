"""
Data processing utilities for handling narratives and tabular data.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from bs4 import BeautifulSoup


def load_documents_from_file(file_path: str, file_type: str = "txt") -> List[str]:
    """
    Load documents from a file.
    
    Args:
        file_path: Path to file
        file_type: Type of file (txt, csv, json, html)
        
    Returns:
        List of document strings
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file type is unsupported
    """
    documents = []
    
    try:
        if file_type == "txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by double newline for separate documents
                documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
        
        elif file_type == "csv":
            df = pd.read_csv(file_path)
            # Convert each row to a document
            for _, row in df.iterrows():
                doc = " | ".join([f"{col}: {val}" for col, val in row.items()])
                documents.append(doc)
        
        elif file_type == "json":
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    documents = [str(item) for item in data]
                else:
                    documents = [str(data)]
        
        elif file_type == "html":
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                # Extract text from paragraphs
                paragraphs = soup.find_all('p')
                documents = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading documents from {file_path}: {e}")
    
    return documents


def parse_tabular_data(data: str) -> Optional[pd.DataFrame]:
    """
    Parse tabular data from string format.
    
    Args:
        data: String containing tabular data
        
    Returns:
        DataFrame if parseable, None otherwise
    """
    try:
        # Try pipe-separated
        if '|' in data:
            lines = data.strip().split('\n')
            rows = [line.split('|') for line in lines]
            if len(rows) > 1:
                df = pd.DataFrame(rows[1:], columns=rows[0])
                return df
        
        # Try tab-separated
        if '\t' in data:
            from io import StringIO
            df = pd.read_csv(StringIO(data), sep='\t')
            return df
        
        return None
    
    except Exception as e:
        print(f"Error parsing tabular data: {e}")
        return None


def chunk_narrative(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Chunk long narrative text with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk (in characters)
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If overlap >= chunk_size
    """
    if overlap >= chunk_size:
        raise ValueError(f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > 0:
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


def extract_features_from_text(text: str) -> Dict[str, Any]:
    """
    Extract features from text for analysis.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of extracted features
    """
    words = text.split()
    word_count = len(words)
    
    features = {
        "length": len(text),
        "word_count": word_count,
        "sentence_count": text.count('.') + text.count('!') + text.count('?'),
        "has_numbers": any(char.isdigit() for char in text),
        "has_tables": '|' in text or '\t' in text,
        "avg_word_length": sum(len(word) for word in words) / word_count if word_count > 0 else 0,
    }
    
    return features


def format_for_display(text: str, max_length: int = 200) -> str:
    """
    Format text for display with truncation.
    
    Args:
        text: Text to format
        max_length: Maximum length
        
    Returns:
        Formatted text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."


def merge_tabular_and_narrative(
    table_data: pd.DataFrame,
    narrative: str,
) -> str:
    """
    Merge tabular data with narrative context.
    
    Args:
        table_data: DataFrame with tabular data
        narrative: Narrative text
        
    Returns:
        Merged representation
    """
    table_str = table_data.to_string(index=False)
    merged = f"{narrative}\n\n[Table Data]\n{table_str}"
    
    return merged
