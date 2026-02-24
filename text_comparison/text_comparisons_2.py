"""
Text Comparison Series: Level 2 - Fuzzy Matching & Metric Search

Focus: 
Moving beyond binary exact matches to quantify *how different* two strings are.
We explore distance metrics (Levenshtein, Damerau), similarity metrics (Jaro-Winkler),
linguistic noise reduction (Stemming), and how to scale fuzzy search to massive 
dictionaries using SymSpell and BK-Trees.

Use Cases:
- Search Autocorrect ("iphne" -> "iphone")
- Entity Resolution (Merging "Jon Smith" and "John Smith")
- OCR Error Correction (Handling scanned "1" vs "l")
- DNA Sequence Alignment
"""

import logging
from typing import Set, List, Tuple

# --- TUNABLE PARAMETERS ---
# Metric Adjustments
PREFIX_WEIGHT = 0.1          # Jaro-Winkler prefix scaling factor (Standard is 0.1)
DAMERAU_PENALTY = 1          # Cost of a transposition (swapping adjacent characters)

# Scaling Structure Configurations
SYMSPELL_MAX_DISTANCE = 2    # Max deletions to pre-calculate (Memory intensive if > 3)
BK_TREE_MAX_RADIUS = 2       # Max edit distance allowed when traversing the BK-Tree

# Configure logging for pedagogical output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Level2_Fuzzy")


class LinguisticProcessor:
    """
    Expert Technique: Stemming.
    Before we check edit distance, we must remove linguistic noise.
    'Astronomer' and 'Astronomy' have an edit distance of 2, but mean the same root.
    Note: Production systems use Porter/Snowball stemmers (e.g., via NLTK/spaCy).
    """

    @staticmethod
    def simple_stem(word: str) -> str:
        """A heuristic suffix-stripper for educational purposes."""
        word = word.lower()
        suffixes = ["ing", "ed", "ly", "es", "s", "ment", "er", "y"]
        
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word


class DistanceMetrics:
    """
    Core algorithms for quantifying string differences.
    """

    @staticmethod
    def levenshtein(s1: str, s2: str) -> int:
        """
        Calculates the minimum number of edits (Insertions, Deletions, Substitutions).
        Memory Optimized: Only keeps the previous row in memory O(min(N, M)).
        """
        if len(s1) < len(s2):
            return DistanceMetrics.levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    @staticmethod
    def damerau_levenshtein(s1: str, s2: str) -> int:
        """
        Expert Metric: Includes Transpositions (swapping adjacent characters).
        Crucial for human typing errors (e.g., 'hte' instead of 'the').
        Uses a full 2D matrix to allow looking back two steps.
        """
        len1, len2 = len(s1), len(s2)
        d = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1): d[i][0] = i
        for j in range(len2 + 1): d[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                
                d[i][j] = min(
                    d[i - 1][j] + 1,      # Deletion
                    d[i][j - 1] + 1,      # Insertion
                    d[i - 1][j - 1] + cost # Substitution
                )
                
                # Transposition Check: Swapping adjacent characters
                if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                    d[i][j] = min(d[i][j], d[i - 2][j - 2] + DAMERAU_PENALTY)
                    
        return d[len1][len2]

    @staticmethod
    def jaro_winkler(s1: str, s2: str) -> float:
        """
        Similarity Metric: Returns 0.0 (Different) to 1.0 (Identical).
        Biases heavily towards strings that share a common prefix.
        Excellent for matching human names or database entities.
        """
        if s1 == s2:
            return 1.0

        # Prefix length matching (standard Jaro-Winkler caps at 4 characters)
        prefix = 0
        for i in range(min(len(s1), len(s2), 4)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break

        # Convert Levenshtein to a baseline 0-1 similarity ratio for this demo
        dist = DistanceMetrics.levenshtein(s1, s2)
        sim = 1.0 - (dist / max(len(s1), len(s2)))
        
        # Apply the Winkler prefix boost
        return sim + (prefix * PREFIX_WEIGHT * (1.0 - sim))


class SymSpellIndex:
    """
    Scale Solution 1: Symmetric Delete Spelling Correction.
    Trades Memory for Speed. Pre-calculates all character deletions up to a max distance.
    Result: O(1) fuzzy lookups.
    """

    def __init__(self, max_distance: int):
        self.max_distance = max_distance
        self.dictionary = {}  # Word -> Popularity/Frequency
        self.deletions = {}   # DeletionVariant -> Set of Original Words

    def _generate_deletions(self, word: str, distance: int) -> Set[str]:
        """Recursively generates all strings resulting from 1 to N deletions."""
        deletes = {word}
        if distance <= 0:
            return deletes

        for i in range(len(word)):
            variant = word[:i] + word[i+1:]
            if len(variant) > 0:
                # Recursively find deeper deletions
                deletes.update(self._generate_deletions(variant, distance - 1))
        return deletes

    def load_word(self, word: str, count: int = 1):
        """Indexes a word and all its deletion variants."""
        word = word.lower()
        self.dictionary[word] = self.dictionary.get(word, 0) + count
        
        variants = self._generate_deletions(word, self.max_distance)
        for variant in variants:
            if variant not in self.deletions:
                self.deletions[variant] = set()
            self.deletions[variant].add(word)

    def lookup(self, typo: str) -> List[str]:
        """Finds dictionary candidates by overlapping their deletion variants."""
        typo = typo.lower()
        candidates = set()
        
        typo_variants = self._generate_deletions(typo, self.max_distance)
        
        for variant in typo_variants:
            if variant in self.deletions:
                candidates.update(self.deletions[variant])
        
        # Rank the results by how common the word is in the dictionary
        return sorted(list(candidates), key=lambda x: self.dictionary.get(x, 0), reverse=True)


class BKTreeNode:
    def __init__(self, word: str):
        self.word = word
        self.children = {}  # Keys are distances (int), Values are BKTreeNodes


class BKTree:
    """
    Scale Solution 2: Burkhard-Keller Trees.
    Trades Speed for Memory. Stores words in a metric tree using the Triangle Inequality.
    Result: O(log N) fuzzy lookups without storing millions of deletion variants.
    """

    def __init__(self, metric_function):
        self.root = None
        self.metric = metric_function

    def add(self, word: str):
        if self.root is None:
            self.root = BKTreeNode(word)
            return

        curr = self.root
        while True:
            dist = self.metric(curr.word, word)
            if dist == 0:
                return  # Word already exists
            
            if dist in curr.children:
                curr = curr.children[dist]  # Traverse deeper
            else:
                curr.children[dist] = BKTreeNode(word)  # Add new branch
                break

    def search(self, query: str, max_radius: int) -> List[Tuple[int, str]]:
        if self.root is None:
            return []

        results = []
        candidates = [self.root]

        while candidates:
            curr = candidates.pop(0)
            dist = self.metric(curr.word, query)
            
            if dist <= max_radius:
                results.append((dist, curr.word))

            # Triangle Inequality Pruning:
            # Only evaluate children whose distance to the parent falls within the radius
            lower_bound = dist - max_radius
            upper_bound = dist + max_radius
            
            for edge_dist, child_node in curr.children.items():
                if lower_bound <= edge_dist <= upper_bound:
                    candidates.append(child_node)

        # Sort by closest match (lowest distance)
        return sorted(results, key=lambda x: x[0])


# --- DEMONSTRATIONS & TUTORIALS ---

def demo_metrics_and_linguistics():
    logger.info("=== DEMO 1: METRICS & LINGUISTIC NOISE ===")
    
    # 1. Levenshtein vs Damerau (The Transposition Problem)
    w1, w2 = "galaxy", "galxay"  # 'x' and 'a' swapped
    
    lev_dist = DistanceMetrics.levenshtein(w1, w2)
    dam_dist = DistanceMetrics.damerau_levenshtein(w1, w2)
    
    logger.info(f"Comparing '{w1}' and typo '{w2}':")
    logger.info(f"   Standard Levenshtein: {lev_dist} edits (delete 'a', insert 'a')")
    logger.info(f"   Damerau-Levenshtein:  {dam_dist} edit  (transposition recognized)")
    
    # 2. Jaro-Winkler (Prefix Bias)
    name1, name2, name3 = "Jonathan", "Johnathan", "Jonathon"
    logger.info(f"\nComparing '{name1}' vs '{name2}' vs '{name3}':")
    logger.info(f"   J-W '{name1}' & '{name2}': {DistanceMetrics.jaro_winkler(name1, name2):.3f}")
    logger.info(f"   J-W '{name1}' & '{name3}': {DistanceMetrics.jaro_winkler(name1, name3):.3f}")
    logger.info("   [Takeaway]: Jaro-Winkler loves shared prefixes. Breaking the prefix early ('John' vs 'Jon') hurts the score more than a suffix typo.")

    # 3. Stemming
    root, variant = "astronomy", "astronomer"
    raw_dist = DistanceMetrics.damerau_levenshtein(root, variant)
    stem_dist = DistanceMetrics.damerau_levenshtein(
        LinguisticProcessor.simple_stem(root), 
        LinguisticProcessor.simple_stem(variant)
    )
    logger.info(f"\nStemming '{root}' vs '{variant}':")
    logger.info(f"   Raw Edit Distance:     {raw_dist}")
    logger.info(f"   Stemmed Edit Distance: {stem_dist}")
    logger.info("   [Takeaway]: Strip morphology before comparing text to capture actual semantic drift.\n")


def demo_scale_structures():
    logger.info("=== DEMO 2: SEARCHING AT SCALE ===")
    
    vocab = ["satellite", "planet", "planetarium", "star", "starlight", "supernova", "nebula"]
    typo = "planett"

    # 1. SymSpell
    logger.info(f"-> Building SymSpell Index (Max Deletions: {SYMSPELL_MAX_DISTANCE})...")
    symspell = SymSpellIndex(max_distance=SYMSPELL_MAX_DISTANCE)
    for word in vocab:
        symspell.load_word(word)
        
    logger.info(f"   SymSpell lookup for '{typo}': {symspell.lookup(typo)}")
    logger.info("   [Method]: Generated deletions of 'planett' and found exact overlaps in the dictionary (O(1)).")

    # 2. BK-Tree
    logger.info(f"\n-> Building BK-Tree (Metric: Damerau-Levenshtein)...")
    bktree = BKTree(metric_function=DistanceMetrics.damerau_levenshtein)
    for word in vocab:
        bktree.add(word)
        
    logger.info(f"   BK-Tree lookup for '{typo}' (Max Radius: {BK_TREE_MAX_RADIUS}):")
    results = bktree.search(typo, max_radius=BK_TREE_MAX_RADIUS)
    for dist, match in results:
        logger.info(f"      - {match} (Distance: {dist})")
    logger.info("   [Method]: Mathematically pruned the search tree, ignoring branches too far from the query (O(log N)).\n")


if __name__ == "__main__":
    demo_metrics_and_linguistics()
    demo_scale_structures()