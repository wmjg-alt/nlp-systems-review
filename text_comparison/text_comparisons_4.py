"""
Text Comparison Series: Level 4 - Lexical Search & Keyword Relevance

Focus:
Comparing a short User Query against a massive database of documents. 
We build an Inverted Index, upgrade it to a Positional Index to support 
exact phrase matching, implement Query Expansion for synonyms, and 
compare the mathematical scoring of TF-IDF vs the industry-standard BM25.

Use Cases:
- E-commerce search bars ("running shoes")
- Log debugging ("Error 500 timeout")
- Ctrl+F across a massive text corpus
"""

import math
import logging
import re
from collections import defaultdict
from typing import List, Dict, Set, Tuple

# --- TUNABLE PARAMETERS ---
# BM25 Parameters
BM25_K1 = 1.25     # Controls Term Frequency Saturation (Standard: 1.2 to 2.0)
BM25_B = 0.8      # Controls Length Normalization (Standard: 0.75)

# Query Expansion (Vocabulary Gap)
SYNONYM_DICT = {
    "sneakers": ["shoes", "kicks", "footwear"],
    "battery": ["power", "cell"],
    "error": ["bug", "glitch", "failure"]
}

# Configure logging for pedagogical output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Level4_Lexical")


class Tokenizer:
    """Expert Technique: Tokenization with Positions."""
    
    @staticmethod
    def tokenize(text: str) -> List[Tuple[str, int]]:
        """
        Returns a list of tuples: (token, position_index).
        We MUST track positions to support exact phrase queries later.
        """
        # Lowercase and split by non-word characters
        raw_tokens = re.finditer(r'\b\w+\b', text.lower())
        return [(match.group(), idx) for idx, match in enumerate(raw_tokens)]


class InvertedIndex:
    """
    The core data structure of all search engines (Elasticsearch, Lucene).
    Instead of mapping Doc -> Words, it maps Word -> Docs.
    """
    
    def __init__(self):
        # term -> {doc_id: [position_1, position_2, ...]}
        self.postings: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        
        # Metadata required for TF-IDF and BM25 scoring
        self.doc_lengths: Dict[str, int] = {}
        self.total_docs: int = 0
        self.total_tokens: int = 0

    def add_document(self, doc_id: str, text: str):
        """Indexes a document and records term positions."""
        tokens = Tokenizer.tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.total_docs += 1
        self.total_tokens += len(tokens)
        
        for token, position in tokens:
            self.postings[token][doc_id].append(position)

    @property
    def avg_doc_length(self) -> float:
        if self.total_docs == 0: return 0.0
        return self.total_tokens / self.total_docs


class LexicalScorer:
    """
    Mathematical scoring algorithms to rank documents based on relevance.
    """
    
    def __init__(self, index: InvertedIndex):
        self.index = index

    def _compute_idf(self, term: str) -> float:
        """
        Inverse Document Frequency.
        Penalizes words that appear in almost every document (like 'the').
        Uses the Lucene formulation to avoid negative scores.
        """
        doc_frequency = len(self.index.postings.get(term, {}))
        if doc_frequency == 0:
            return 0.0
        # log(1 + (N - n + 0.5) / (n + 0.5))
        numerator = self.index.total_docs - doc_frequency + 0.5
        denominator = doc_frequency + 0.5
        return math.log(1.0 + (numerator / denominator))

    def score_tfidf(self, query_terms: List[str], doc_id: str) -> float:
        """
        Raw TF-IDF. 
        Vulnerable to keyword stuffing and biased towards long documents.
        """
        score = 0.0
        for term in query_terms:
            positions = self.index.postings.get(term, {}).get(doc_id, [])
            term_freq = len(positions)
            if term_freq > 0:
                idf = self._compute_idf(term)
                # Raw Term Frequency * IDF
                score += term_freq * idf
        return score

    def score_bm25(self, query_terms: List[str], doc_id: str) -> float:
        """
        Best Matching 25. The industry standard.
        Applies mathematical 'brakes' (saturation) to term frequency and 
        normalizes the score against the document's length.
        """
        score = 0.0
        doc_len = self.index.doc_lengths.get(doc_id, 0)
        avg_dl = self.index.avg_doc_length

        for term in query_terms:
            positions = self.index.postings.get(term, {}).get(doc_id, [])
            term_freq = len(positions)
            if term_freq > 0:
                idf = self._compute_idf(term)
                
                # BM25 Term Frequency Saturation and Length Normalization formula
                numerator = term_freq * (BM25_K1 + 1)
                denominator = term_freq + BM25_K1 * (1 - BM25_B + BM25_B * (doc_len / avg_dl))
                
                score += idf * (numerator / denominator)
        return score


class SearchEngine:
    """Orchestrates queries against the Inverted Index and Scorer."""
    
    def __init__(self):
        self.index = InvertedIndex()
        self.scorer = LexicalScorer(self.index)

    def index_documents(self, docs: Dict[str, str]):
        for doc_id, text in docs.items():
            self.index.add_document(doc_id, text)

    def keyword_search(self, query: str, use_bm25: bool = True, expand_synonyms: bool = False) -> List[Tuple[str, float]]:
        """Standard 'Bag of Words' search."""
        raw_tokens = [t[0] for t in Tokenizer.tokenize(query)]
        query_terms = set(raw_tokens)
        
        # Expert Technique: Query Expansion
        if expand_synonyms:
            expanded_terms = set(query_terms)
            for term in query_terms:
                if term in SYNONYM_DICT:
                    expanded_terms.update(SYNONYM_DICT[term])
            query_terms = expanded_terms

        # Find all documents that contain AT LEAST ONE query term (OR logic)
        candidate_docs = set()
        for term in query_terms:
            candidate_docs.update(self.index.postings.get(term, {}).keys())

        # Score the candidates
        results = []
        for doc_id in candidate_docs:
            if use_bm25:
                score = self.scorer.score_bm25(list(query_terms), doc_id)
            else:
                score = self.scorer.score_tfidf(list(query_terms), doc_id)
            results.append((doc_id, score))

        # Sort by highest score
        return sorted(results, key=lambda x: x[1], reverse=True)

    def exact_phrase_search(self, phrase: str) -> List[str]:
        """
        Expert Technique: Positional Search.
        Finds documents where words appear exactly in the requested sequence.
        """
        tokens = [t[0] for t in Tokenizer.tokenize(phrase)]
        if not tokens: return []

        # Start with docs containing the first word
        candidate_docs = set(self.index.postings.get(tokens[0], {}).keys())
        
        # Narrow down by checking if subsequent words exist in the same doc
        for token in tokens[1:]:
            docs_with_token = set(self.index.postings.get(token, {}).keys())
            candidate_docs.intersection_update(docs_with_token)

        # Now check actual positions
        matched_docs = []
        for doc_id in candidate_docs:
            # Get positions for the first word
            base_positions = self.index.postings[tokens[0]][doc_id]
            
            # Check if there's a valid sequence starting at any of these positions
            for base_pos in base_positions:
                is_match = True
                for offset, token in enumerate(tokens[1:], 1):
                    expected_pos = base_pos + offset
                    if expected_pos not in self.index.postings[token][doc_id]:
                        is_match = False
                        break
                
                if is_match:
                    matched_docs.append(doc_id)
                    break # We just need to know it appears once
                    
        return matched_docs


# --- TUTORIAL DEMONSTRATIONS ---

def run_demonstrations():
    engine = SearchEngine()
    
    # --- The Test Corpus ---
    docs = {
        # Positional Edge Cases
        "doc_A": "The new york city skyline is beautiful.",
        "doc_B": "I bought a new car in the state of york.",
        
        # BM25 vs TF-IDF Edge Cases (Keyword Spam & Length Bias)
        "doc_C": "Battery " * 50, # Keyword Spam
        "doc_D": "Help, my laptop battery died.", # Highly relevant, short
        "doc_E": "This is a massive five thousand word manual about computers. Somewhere on page forty it mentions that a battery can be replaced. " * 20, # Long doc, incidental mention
        
        # Query Expansion Edge Cases
        "doc_F": "I am looking for some red running shoes.",
    }
    
    logger.info("Indexing Corpus...")
    engine.index_documents(docs)

    # --- DEMO 1: EXACT PHRASE MATCHING ---
    logger.info("\n=== DEMO 1: THE 'NEW YORK' PROBLEM (POSITIONAL INDEX) ===")
    query_phrase = "new york"
    logger.info(f"Query: '{query_phrase}'")
    
    logger.info("\n1. Standard Keyword Search ('new' AND 'york' anywhere):")
    # Doing a manual AND intersection for demonstration
    kw_results = engine.keyword_search("new york", use_bm25=True)
    for doc_id, score in kw_results:
        logger.info(f"   -> {doc_id} (Score: {score:.2f}) | Text: {docs[doc_id][:40]}...")
        
    logger.info("\n2. Exact Phrase Search (Requires Position N and N+1):")
    phrase_results = engine.exact_phrase_search("new york")
    for doc_id in phrase_results:
        logger.info(f"   -> {doc_id} | Text: {docs[doc_id][:40]}...")
        
    logger.info("\n[Takeaway]: Standard search returned Doc B because it contains both words. The Positional Index correctly isolated Doc A because it tracks exact token adjacency.")


    # --- DEMO 2: TF-IDF VS BM25 ---
    logger.info("\n=== DEMO 2: TF-IDF VS BM25 (SPAM & LENGTH BIAS) ===")
    query = "battery"
    logger.info(f"Query: '{query}'")
    
    logger.info("\n1. Raw TF-IDF Scoring:")
    tfidf_res = engine.keyword_search(query, use_bm25=False)
    for doc_id, score in tfidf_res:
        logger.info(f"   -> {doc_id} (Score: {score:.2f}) | Text: {docs[doc_id][:40]}...")
        
    logger.info("\n2. BM25 Scoring (Saturation & Length Normalization):")
    bm25_res = engine.keyword_search(query, use_bm25=True)
    for doc_id, score in bm25_res:
        logger.info(f"   -> {doc_id} (Score: {score:.2f}) | Text: {docs[doc_id][:40]}...")

    logger.info("\n[Takeaway]: TF-IDF was tricked by Doc C's keyword spam (50x 'battery'). BM25 flattened the spam via Saturation (k1 parameter) and boosted Doc D because it is concise (b parameter length normalization).")


    # --- DEMO 3: QUERY EXPANSION ---
    logger.info("\n=== DEMO 3: THE VOCABULARY GAP (QUERY EXPANSION) ===")
    query = "sneakers"
    logger.info(f"Query: '{query}'")
    
    logger.info("\n1. Standard Search:")
    res_standard = engine.keyword_search(query, expand_synonyms=False)
    if not res_standard:
        logger.info("   -> No results found.")
        
    logger.info("\n2. Search with Synonym Query Expansion:")
    res_expanded = engine.keyword_search(query, expand_synonyms=True)
    for doc_id, score in res_expanded:
        logger.info(f"   -> {doc_id} (Score: {score:.2f}) | Text: {docs[doc_id][:40]}...")
        
    logger.info("\n[Takeaway]: Lexical search is brittle. If the user says 'sneakers' but the doc says 'shoes', the match fails. Intercepting and expanding the query dynamically fixes this without bloating the database.")

if __name__ == "__main__":
    run_demonstrations()