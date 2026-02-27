"""
=============================================================================
  INTERVIEW PREP · Q10 · THE GENERATOR & LAZY EVALUATION
  Run:  python q10_generators.py
=============================================================================

  The Question You'll Be Asked:
    "How would you process a file larger than available RAM?"

  The Wrong Answer: Load it all into memory.
  The Right Answer: Use a generator — process one item at a time.
  The Great Answer: Chain generators into a lazy pipeline. O(1) memory,
                    composable, clean, and works on infinite data streams.

  This file teaches you:
    1. What a generator IS and how yield works
    2. HOW to measure the difference (timing + memory)
    3. WHY generators beat list comprehensions in specific scenarios
    4. WHEN to use each — because list comps aren't always wrong
    5. The advanced chained pipeline pattern
=============================================================================
"""

import time
import os
import gc
import tracemalloc
import tempfile
import functools
from typing import Iterator

# =============================================================================
#  TUNABLE PARAMETERS
# =============================================================================

FILE_LINES      = 200_000   # Lines in our fake log file
LINE_TEMPLATE   = "2024-01-01 ERROR service:{i} message: request timed out\n"
TIMING_REPEAT   = 3         # How many times to repeat each timing test

# =============================================================================
#  DISPLAY HELPERS
# =============================================================================

DIVIDER = "\n" + "=" * 72
SECTION = "\n" + "-" * 60

def header(title):
    print(f"{DIVIDER}\n  {title}{DIVIDER}")

def section(title):
    print(f"{SECTION}\n  {title}{SECTION}")

def result(label, value):
    print(f"    → {label}: {value}")

def note(msg):
    print(f"    ℹ {msg}")

def trap(msg):
    print(f"    ⚠  TRAP: {msg}")

def fix(msg):
    print(f"    ✓  FIX:  {msg}")

def show_timing(label, seconds):
    print(f"    ⏱  {label}: {seconds:.4f}s")

def show_mem(label, peak_kb):
    if peak_kb > 1024:
        print(f"    🧠 {label}: {peak_kb/1024:.1f} MB")
    else:
        print(f"    🧠 {label}: {peak_kb:.0f} KB")

def separator():
    print()

# =============================================================================
#  ✨ THE TIMING DECORATOR
#
#  A decorator wraps a function to add behaviour before/after it runs —
#  without changing the function's code. Here we use one to measure how
#  long any function takes and how much memory it peaks at.
#
#  Why a decorator and not just manual time.perf_counter() calls?
#    - DRY: write the measurement logic once, apply it anywhere
#    - Clean: the measured function stays readable
#    - Reusable: works on any function you hand it
# =============================================================================

def timed(label=None, repeat=1):
    """
    Decorator factory. Wraps a function to print its wall-clock time
    and peak memory usage (via tracemalloc).

    Usage:
        @timed("readlines approach", repeat=3)
        def my_fn():
            ...

    repeat > 1 averages multiple runs to reduce noise.
    """
    def decorator(fn):
        @functools.wraps(fn)     # preserves fn.__name__, __doc__, etc.
        def wrapper(*args, **kwargs):
            name = label or fn.__name__

            elapsed_times = []
            result_val = None

            for i in range(repeat):
                gc.collect()                    # clear garbage before measuring
                tracemalloc.start()
                t0 = time.perf_counter()

                result_val = fn(*args, **kwargs)

                elapsed = time.perf_counter() - t0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                elapsed_times.append(elapsed)

            avg = sum(elapsed_times) / len(elapsed_times)
            show_timing(f"{name} (avg of {repeat})", avg)
            show_mem(f"{name} peak RAM", peak / 1024)
            return result_val
        return wrapper
    return decorator


# =============================================================================
#  SETUP — build a temp file we can actually read
# =============================================================================

def build_temp_file() -> str:
    """
    Write a fake log file to disk. We use delete=False so we control
    when it's cleaned up. We close it immediately after writing so
    Windows doesn't hold a lock on it during the demo reads.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    )
    for i in range(FILE_LINES):
        tmp.write(LINE_TEMPLATE.format(i=i))
    tmp.close()   # ← close NOW. Windows locks open file handles.
    return tmp.name


# =============================================================================
#  APPROACH 1 — readlines(): load everything into RAM at once
# =============================================================================

def approach_readlines(path: str) -> int:
    """
    Naive approach: read the whole file into a list, then iterate.
    This is fine for small files. It becomes a problem — or a crash —
    when the file is larger than available RAM.

    Space complexity: O(N) — the entire file lives in memory.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()        # ← entire file materialised as a list
    # Simulate processing: count ERROR lines
    return sum(1 for line in lines if "ERROR" in line)


# =============================================================================
#  APPROACH 2 — list comprehension: still O(N) in memory
# =============================================================================

def approach_listcomp(path: str) -> int:
    """
    A list comprehension over a file is more Pythonic than readlines(),
    but it has the same memory problem: it builds the entire filtered
    list in memory before you can do anything with it.

    [line for line in f if "ERROR" in line]  ← full list, then iterate

    Space complexity: O(N) — still holds every matching line at once.
    This is the trap many intermediate Python devs fall into:
    list comps look clean but aren't lazy.
    """
    with open(path, encoding="utf-8") as f:
        error_lines = [line for line in f if "ERROR" in line]
    return len(error_lines)


# =============================================================================
#  APPROACH 3 — generator function: O(1) memory
# =============================================================================

def stream_errors(path: str) -> Iterator[str]:
    """
    A generator function. The `yield` keyword is what makes this special.

    How it works:
      - Calling stream_errors() does NOT execute any code yet.
        It returns a generator object (an iterator).
      - Code runs only when you call next() on it (or loop over it).
      - `yield` suspends the function, hands a value to the caller,
        and freezes state. On the next next() call, it resumes from
        exactly where it paused.
      - The `with open(...)` block stays open across all yields,
        and closes cleanly when the generator is exhausted or garbage-collected.

    Space complexity: O(1) — only one line exists at a time.
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "ERROR" in line:
                yield line           # ← pause here, give this line to caller


def approach_generator(path: str) -> int:
    """Consume the generator, counting results — never storing them."""
    return sum(1 for _ in stream_errors(path))


# =============================================================================
#  APPROACH 4 — generator expression: same memory, inline syntax
# =============================================================================

def approach_genexpr(path: str) -> int:
    """
    A generator expression uses () instead of [].

    [x for x in data]   ← list comprehension: eager, builds a list NOW
    (x for x in data)   ← generator expression: lazy, produces on demand

    They look almost identical. The difference is entirely about WHEN
    the work happens and WHERE the results live.

    Use a generator expression when:
      - You only need to iterate once
      - You don't need random access (data[i])
      - The data could be large

    Use a list comprehension when:
      - You need to iterate multiple times
      - You need to index into results
      - You need len(), reversed(), etc.
      - The data is small and speed of repeated access matters
    """
    with open(path, encoding="utf-8") as f:
        # The () makes this a generator — lines are NOT stored
        return sum(1 for line in f if "ERROR" in line)


# =============================================================================
#  THE CRITICAL COMPARISON — generator vs list comp, same task
#
#  This is the section the original lesson was missing. We need to show
#  the SAME logical operation done two ways and see the actual difference.
# =============================================================================

def compare_generator_vs_listcomp(path: str):
    section("🔬  HEAD-TO-HEAD: List Comprehension vs Generator Expression")

    note("Same task: filter ERROR lines, count them.")
    note("Same data. Different memory behaviour.")
    note("Watch the peak RAM numbers — that's the story.")
    separator()

    # List comp version — wrapped in timed() inline using a lambda-style approach
    print("  [ List comprehension ] ← builds the whole result in memory first:")
    print("""
    with open(path) as f:
        error_lines = [line for line in f if "ERROR" in line]
    count = len(error_lines)
""")

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    with open(path, encoding="utf-8") as f:
        error_lines = [line for line in f if "ERROR" in line]
    count_lc = len(error_lines)
    t_lc = time.perf_counter() - t0
    _, peak_lc = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del error_lines   # free it before next test
    gc.collect()

    show_timing("list comp time", t_lc)
    show_mem("list comp peak RAM", peak_lc / 1024)
    result("count", count_lc)
    separator()

    print("  ( Generator expression ) ← processes one line at a time:")
    print("""
    with open(path) as f:
        count = sum(1 for line in f if "ERROR" in line)
""")

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    with open(path, encoding="utf-8") as f:
        count_ge = sum(1 for line in f if "ERROR" in line)
    t_ge = time.perf_counter() - t0
    _, peak_ge = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    show_timing("generator expr time", t_ge)
    show_mem("generator expr peak RAM", peak_ge / 1024)
    result("count", count_ge)
    separator()

    # Verdict
    mem_ratio = peak_lc / peak_ge if peak_ge > 0 else float('inf')
    print("  📊  VERDICT:")
    result("Memory: list comp used", f"{mem_ratio:.1f}× more RAM than generator")
    if t_lc < t_ge:
        result("Speed: list comp was", f"{t_ge/t_lc:.2f}× FASTER (more on this below)")
    else:
        result("Speed: generator was", f"{t_lc/t_ge:.2f}× faster")

    separator()
    note("WHY might the list comp be faster? It allocates a contiguous block")
    note("of memory up front. Python is optimised for this. The generator")
    note("has overhead per yield: function call frame + state save/restore.")
    note("")
    note("The trade-off is NOT speed. It's MEMORY vs SPEED.")
    note("  List comp:  fast access, high RAM, can iterate multiple times")
    note("  Generator:  slow(er), O(1) RAM, single-pass only")
    note("")
    note("When the file is 2GB and you have 4GB RAM, the list comp crashes.")
    note("The generator just... works. That's when you reach for it.")


# =============================================================================
#  WHEN GENERATORS WIN vs WHEN THEY DON'T — the honest comparison
# =============================================================================

def demo_when_generator_wins():
    section("🏆  WHEN TO USE EACH — THE HONEST ANSWER")

    print("""
  REACH FOR A GENERATOR WHEN:
  ─────────────────────────────────────────────────────────────────
  ✓ Data is larger than (or approaches) available RAM
       → log files, NLP corpora, CSV dumps, API pagination streams
  ✓ You only need to iterate ONCE (count, sum, find first match)
  ✓ You want to compose a pipeline and keep it lazy end-to-end
  ✓ You're building an API that returns a stream of results
  ✓ Data is potentially INFINITE (event streams, sensor feeds)

  REACH FOR A LIST COMPREHENSION WHEN:
  ─────────────────────────────────────────────────────────────────
  ✓ Data fits comfortably in RAM
  ✓ You need to iterate the results MORE THAN ONCE
  ✓ You need random access: results[i] or results[-1]
  ✓ You need len(), sorted(), reversed(), or slicing
  ✓ You're passing results to a function that needs a real list
  ✓ Raw speed matters more than memory (list comps are often faster)

  THE KNOWLEDGE THAT IMPRESSES:
  ─────────────────────────────────────────────────────────────────
  "It depends on the access pattern and data size. For a single-pass
   operation on data that might not fit in RAM, I'd use a generator
   for O(1) memory. If I need multiple passes or random access and
   the data fits in RAM, a list comprehension is faster and simpler.
   The real power is chaining generators into a pipeline — each stage
   is lazy, so the whole chain uses O(1) memory regardless of input size."
""")


# =============================================================================
#  ADVANCED — chained generator pipeline with yield from
# =============================================================================

def demo_pipeline(path: str):
    section("✅✅  ADVANCED: The Generator Pipeline")

    note("Real power: chain generators so the WHOLE pipeline is lazy.")
    note("Each stage receives an iterator and yields an iterator.")
    note("Nothing is stored between stages. O(1) memory end-to-end.")
    note("Real NLP use: read → tokenise → filter stopwords → batch → model")
    separator()

    print("""
  Each function takes an iterable and yields transformed items.
  They're composable like Unix pipes: cat file | grep ERROR | awk ...

    def read_file(path):
        with open(path) as f:
            yield from f               # yield from delegates to file iterator

    def strip_lines(lines):
        for line in lines:
            yield line.strip()

    def only_errors(lines):
        for line in lines:
            if "ERROR" in line:
                yield line

    def extract_service(lines):
        for line in lines:
            for token in line.split():
                if token.startswith("service:"):
                    yield token

    def batch(items, size=100):
        buf = []
        for item in items:
            buf.append(item)
            if len(buf) == size:
                yield buf              # yield a full batch
                buf = []
        if buf:
            yield buf                  # yield the remainder

    # Build the pipeline — NO data has moved yet. This is just wiring.
    pipeline = batch(
                   extract_service(
                       only_errors(
                           strip_lines(
                               read_file(path)))))

    # NOW data flows — one item at a time, pulled through each stage
    for b in pipeline:
        feed_to_model(b)
""")

    # Run it for real
    def read_file(path):
        with open(path, encoding="utf-8") as f:
            yield from f

    def strip_lines(lines):
        for line in lines:
            yield line.strip()

    def only_errors(lines):
        for line in lines:
            if "ERROR" in line:
                yield line

    def extract_service(lines):
        for line in lines:
            for token in line.split():
                if token.startswith("service:"):
                    yield token

    def batch(items, size=100):
        buf = []
        for item in items:
            buf.append(item)
            if len(buf) == size:
                yield buf
                buf = []
        if buf:
            yield buf

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    pipeline = batch(extract_service(only_errors(strip_lines(read_file(path)))))
    total_batches = 0
    total_tokens  = 0
    for b in pipeline:
        total_batches += 1
        total_tokens  += len(b)

    t_pipe = time.perf_counter() - t0
    _, peak_pipe = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    show_timing("full pipeline time", t_pipe)
    show_mem("pipeline peak RAM (entire file processed!)", peak_pipe / 1024)
    result("total service tokens extracted", f"{total_tokens:,}")
    result("total batches of 100", total_batches)
    separator()
    note("That peak RAM is for the WHOLE file. The pipeline held one item")
    note("at a time at each stage. Batches of 100 are the largest structure.")
    note("Scale this to a 50GB corpus and the memory number barely changes.")


# =============================================================================
#  BONUS — the timed decorator in action, showing its own value
# =============================================================================

def demo_timed_decorator(path: str):
    section("🎓  THE TIMED DECORATOR — seeing it work")

    note("The @timed() decorator adds measurement without cluttering functions.")
    note("This is the decorator pattern: wrap behaviour around existing code.")
    note("functools.wraps() preserves the original function's name and docstring.")
    separator()

    print("  Applying @timed() to each approach, repeat=3 for averaged timing:\n")

    @timed("readlines (full load)", repeat=TIMING_REPEAT)
    def run_readlines():
        return approach_readlines(path)

    @timed("list comprehension", repeat=TIMING_REPEAT)
    def run_listcomp():
        return approach_listcomp(path)

    @timed("generator function", repeat=TIMING_REPEAT)
    def run_generator():
        return approach_generator(path)

    @timed("generator expression", repeat=TIMING_REPEAT)
    def run_genexpr():
        return approach_genexpr(path)

    r1 = run_readlines()
    r2 = run_listcomp()
    r3 = run_generator()
    r4 = run_genexpr()

    separator()
    note(f"All approaches counted the same lines: {r1} = {r2} = {r3} = {r4}")
    note("The memory column tells you the real story, not the speed column.")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  INTERVIEW PREP · Q10 · GENERATORS & LAZY EVALUATION                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Core question: "How do you process a file larger than available RAM?"
  Core concept:  Lazy evaluation — only compute what you need, when you need it.

  We'll measure BOTH time and memory so you can see the real trade-offs.
""")

    section("Setup: writing test file to disk")
    path = build_temp_file()
    file_mb = os.path.getsize(path) / (1024 * 1024)
    result("File size", f"{file_mb:.1f} MB  ({FILE_LINES:,} lines)")
    note("File is closed immediately after writing — safe to read on Windows.")

    # The four approaches, explained and measured
    demo_timed_decorator(path)
    compare_generator_vs_listcomp(path)
    demo_when_generator_wins()
    demo_pipeline(path)

    # Clean up — file is closed, so os.unlink is safe on Windows
    try:
        os.unlink(path)
        result("Cleanup", "temp file deleted")
    except OSError as e:
        note(f"Could not delete temp file (safe to delete manually): {e}")

    print(f"""
{"=" * 72}
  SUMMARY 

  "A generator uses the yield keyword to produce values one at a time
   instead of building a full list. This gives O(1) memory regardless
   of input size, at the cost of being single-pass only.

   A list comprehension builds the full result eagerly — it's often
   faster for small data and necessary when you need random access or
   multiple iterations.

   The right choice depends on access pattern and data size. For large
   or unbounded data with a single-pass operation, use a generator.
   For small data or multi-pass needs, use a list comprehension.

   The advanced pattern is chaining generators into a pipeline — each
   stage is lazy, so the whole chain uses O(1) memory end-to-end.
   Think of it like Unix pipes: grep | awk | sort, but in Python."
{"=" * 72}
""")


if __name__ == "__main__":
    main()