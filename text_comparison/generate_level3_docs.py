"""
Generator Script for Level 3: Syntactic Similarity

This script creates a 'docs/' directory and generates controlled test 
documents based on a base text. By strictly controlling the differences 
between the documents, we can perfectly demonstrate the strengths and 
weaknesses of Jaccard Similarity, Containment, MinHash, and LSH.
"""

import os
from pathlib import Path

# Fallback text in case the user doesn't have a lorem_ipsum.txt ready
BASE_LOREM = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis 
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore 
eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt 
in culpa qui officia deserunt mollit anim id est laborum.
"""

UNRELATED_TEXT = """
The mitochondria is the powerhouse of the cell. Photosynthesis is the process 
by which green plants and some other organisms use sunlight to synthesize foods 
from carbon dioxide and water. Cellular respiration is a set of metabolic reactions 
and processes that take place in the cells of organisms to convert biochemical 
energy from nutrients into adenosine triphosphate.
"""

def setup_docs():
    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)

    # 1. Base Document
    doc_a = BASE_LOREM.strip()
    (out_dir / "doc_A_original.txt").write_text(doc_a)

    # 2. Minor Edits (High Jaccard Similarity)
    # We change a few words. This simulates a slightly edited article.
    doc_b = doc_a.replace("consectetur adipiscing elit", "completely different words")
    doc_b = doc_b.replace("laborum", "laborum and some extra text")
    (out_dir / "doc_B_minor_edits.txt").write_text(doc_b)

    # 3. Subset / Plagiarism (Low Jaccard, High Containment)
    # Just the first sentence.
    doc_c = doc_a.split(".")[0] + "."
    (out_dir / "doc_C_subset.txt").write_text(doc_c)

    # 4. Aggregate / Digest (Low Jaccard, High Containment)
    # The original document buried inside a much larger, unrelated document.
    doc_d = UNRELATED_TEXT.strip() + "\n\n" + doc_a + "\n\n" + UNRELATED_TEXT.strip()
    (out_dir / "doc_D_aggregate.txt").write_text(doc_d)

    # 5. Completely Unrelated (Zero Jaccard, Zero Containment)
    (out_dir / "doc_E_unrelated.txt").write_text(UNRELATED_TEXT.strip())

    print(f"✅ Generated 5 test documents in '{out_dir.absolute()}'")

if __name__ == "__main__":
    setup_docs()