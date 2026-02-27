"""
=============================================================================
  INTERVIEW PREP · Q11 · THE STRING CONCATENATION TRAP
  Run:  python q11_string_concat.py
=============================================================================

  The Question You'll Be Asked:
    "What's wrong with building a string in a loop with s = s + chunk?"

  The Wrong Answer: Nothing, it works fine.
  The Right Answer: Strings are immutable. Each += copies the whole string.
                    That's O(N²) — it gets exponentially worse as N grows.
  The Great Answer: Use "".join(parts) — one allocation, O(N). Know when
                    StringIO is a better tool, and know the CPython gotcha
                    that makes the naive version LOOK fast in toy scripts.

  This file teaches you:
    1. WHY += on strings is O(N²) — rooted in Python's immutability model
    2. Measured proof: you'll see the curve, not just take our word for it
    3. The join() fix and why it's O(N)
    4. StringIO as an alternative for conditional/streaming writes
    5. The CPython refcount optimisation and why you must never rely on it
=============================================================================
"""

import time
import gc
import functools
import tracemalloc

# =============================================================================
#  TUNABLE PARAMETERS
# =============================================================================

# Sizes to test — we run ALL of these to show how the curve scales
# Keep the largest value under 100_000 unless you're patient
TEST_SIZES   = [1_000, 5_000, 10_000, 25_000, 50_000]
WORD         = "token"          # The chunk we concatenate
TIMING_REPS  = 3                # Averages per measurement

# =============================================================================
#  DISPLAY HELPERS (shared style across all three lesson files)
# =============================================================================

DIVIDER = "\n" + "=" * 72
SECTION = "\n" + "-" * 60

def header(title):      print(f"{DIVIDER}\n  {title}{DIVIDER}")
def section(title):     print(f"{SECTION}\n  {title}{SECTION}")
def result(label, val): print(f"    → {label}: {val}")
def note(msg):          print(f"    ℹ {msg}")
def trap(msg):          print(f"    ⚠  TRAP: {msg}")
def fix(msg):           print(f"    ✓  FIX:  {msg}")
def show_t(lbl, s):     print(f"    ⏱  {lbl}: {s:.5f}s")
def sep():              print()

# =============================================================================
#  TIMING DECORATOR (same pattern as q10 — reusable across your codebase)
# =============================================================================

def timed(label=None, repeat=1):
    """
    Decorator factory — measures wall-clock time and peak RAM.
    repeat > 1 averages multiple runs for stable numbers.

    This is the decorator pattern: behaviour (measurement) is added
    to a function without touching the function's own code.
    functools.wraps() ensures the wrapped function keeps its identity
    (__name__, __doc__) for debugging and introspection.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            name = label or fn.__name__
            times = []
            peak  = 0
            val   = None
            for _ in range(repeat):
                gc.collect()
                tracemalloc.start()
                t0  = time.perf_counter()
                val = fn(*args, **kwargs)
                t   = time.perf_counter() - t0
                _, p = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                times.append(t)
                peak = max(peak, p)
            avg = sum(times) / len(times)
            print(f"    ⏱  {name} (avg {repeat}x): {avg:.5f}s  |  "
                  f"🧠 peak: {peak/1024:.0f} KB")
            return val
        return wrapper
    return decorator


# =============================================================================
#  THE FOUR APPROACHES
# =============================================================================

def concat_naive(words: list) -> str:
    """
    THE TRAP: s = s + chunk in a loop.

    Python strings are immutable — once created, a string object
    cannot be modified. s + chunk doesn't append; it allocates a
    brand new string, copies ALL of s, then copies chunk.

    In a loop of N iterations:
      Iteration 1: copy 1 char
      Iteration 2: copy 2 chars
      Iteration 3: copy 3 chars
      ...
      Iteration N: copy N chars

    Total bytes copied: 1 + 2 + 3 + ... + N = N(N+1)/2 = O(N²)

    Double N → 4× the work. That's the quadratic trap.
    """
    s = ""
    for word in words:
        s = s + word + " "
    return s


def concat_augmented(words: list) -> str:
    """
    THE LOOK-ALIKE TRAP: s += chunk.

    This is syntactically different from s = s + chunk but semantically
    identical in terms of immutability. Same O(N²) worst case.

    CPython has an optimisation: if the string's reference count is 1
    (only one variable points to it), it MAY resize in-place.
    This is an implementation detail, not a language guarantee.
    We'll demonstrate exactly when it breaks.
    """
    s = ""
    for word in words:
        s += word + " "
    return s


def concat_join(words: list) -> str:
    """
    THE FIX: "".join(iterable)

    join() does two passes internally:
      Pass 1: iterate to calculate total required length
      Pass 2: allocate ONE buffer of that length, fill it

    Total work: O(N). One allocation. No copies.

    This is the canonical Python idiom. Any senior engineer seeing
    a string-building loop will immediately ask "why not join()?"
    """
    return " ".join(words)


def concat_stringio(words: list) -> str:
    """
    THE ALTERNATIVE: io.StringIO

    StringIO wraps a C-level byte buffer. Each write() appends to
    that buffer directly — no Python string allocation per call.

    When to prefer StringIO over join():
      - You can't collect everything into a list first
        (streaming data, conditional writes, mixed separators)
      - You're building a string across multiple functions
        (pass the buffer around, call write() anywhere)
      - You want a file-like interface (write, seek, tell)

    When to prefer join():
      - You already have a list of parts
      - Simple separator logic
      - Slightly faster and more readable for straightforward cases
    """
    from io import StringIO
    buf = StringIO()
    for word in words:
        buf.write(word)
        buf.write(" ")
    return buf.getvalue()


# =============================================================================
#  THE SCALING DEMO — showing the O(N²) curve in real numbers
# =============================================================================

def demo_scaling():
    section("📈  THE SCALING DEMO — watch O(N²) emerge in real time")

    note("We run all four approaches at increasing sizes.")
    note("If naive += were O(N), doubling N would double the time.")
    note("Watch what actually happens — it roughly quadruples.")
    sep()

    print(f"  {'Size':>8}  {'naive +=':<14} {'join()':<14} {'StringIO':<14}  ratio (naive/join)")
    print(f"  {'-'*8}  {'-'*13} {'-'*13} {'-'*13}  {'-'*18}")

    for n in TEST_SIZES:
        words = [WORD] * n

        # naive
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(TIMING_REPS): concat_naive(words)
        t_naive = (time.perf_counter() - t0) / TIMING_REPS

        # join
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(TIMING_REPS): concat_join(words)
        t_join = (time.perf_counter() - t0) / TIMING_REPS

        # stringio
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(TIMING_REPS): concat_stringio(words)
        t_sio = (time.perf_counter() - t0) / TIMING_REPS

        ratio = t_naive / t_join if t_join > 0 else 0
        print(f"  {n:>8,}  {t_naive:<14.5f} {t_join:<14.5f} {t_sio:<14.5f}  {ratio:.1f}×")

    sep()
    note("The ratio column should grow as N grows — that's the quadratic curve.")
    note("join() stays roughly proportional to N. naive gets exponentially worse.")


# =============================================================================
#  THE CPYTHON REFCOUNT TRAP — the most dangerous 'gotcha' in this topic
# =============================================================================

def demo_cpython_gotcha():
    section("⚠  THE CPYTHON GOTCHA — why naive code sometimes LOOKS fast")

    note("CPython has an optimisation: if a string's reference count is 1,")
    note("it may extend the buffer in-place rather than allocating a new one.")
    note("This makes naive += appear fast in simple scripts. It's a lie.")
    sep()

    trap("The optimisation BREAKS in all of these common situations:")
    print("""
    # 1. Two variables pointing at the same string
    s = ""
    alias = s          ← refcount is now 2; optimisation disabled
    for w in words:
        s += w         ← must copy — alias still points to old s

    # 2. String stored in a container
    s = ""
    log.append(s)      ← refcount 2; s is in the list AND your variable
    for w in words:
        s += w         ← must copy every time

    # 3. String passed to a function
    def inspect(x): pass
    s = ""
    for w in words:
        inspect(s)     ← inspect holds a reference during the call
        s += w         ← refcount is 2 inside the loop body

    # 4. Different Python implementation (PyPy, Jython, GraalPy)
    ← no such optimisation exists; always O(N²)
""")
    fix("join() doesn't rely on any of this. It's O(N) by design, always.")

    sep()
    note("Interview insight: mentioning this gotcha signals you've been")
    note("burned by it, or read deep enough to know about it. Either way,")
    note("it's the kind of detail that separates a good answer from a great one.")


# =============================================================================
#  HEAD-TO-HEAD WITH THE TIMED DECORATOR
# =============================================================================

def demo_timed(n: int):
    section(f"🎓  TIMED DECORATOR DEMO at N={n:,}")
    words = [WORD] * n

    @timed("naive s = s + chunk", repeat=TIMING_REPS)
    def run_naive(): return concat_naive(words)

    @timed("naive s += chunk   ", repeat=TIMING_REPS)
    def run_aug():   return concat_augmented(words)

    @timed("join()             ", repeat=TIMING_REPS)
    def run_join():  return concat_join(words)

    @timed("StringIO           ", repeat=TIMING_REPS)
    def run_sio():   return concat_stringio(words)

    r1 = run_naive()
    r2 = run_aug()
    r3 = run_join()
    r4 = run_sio()

    sep()
    # Verify correctness — strip trailing space before comparing
    # (naive adds a trailing space per item; join() doesn't)
    assert r1.strip() == r2.strip() == r3.strip() == r4.strip(), \
        "Results differ — something is wrong"
    result("All results identical (after strip)", "✓")
    note("Peak RAM shows join() and StringIO barely allocate at measurement time")
    note("because the result string is returned, not stored in a growing buffer.")


# =============================================================================
#  WHEN TO USE EACH
# =============================================================================

def demo_decision():
    section("🏆  WHEN TO USE EACH")
    print("""
  USE join() WHEN:
  ─────────────────────────────────────────────────────────────────
  ✓ You have a list (or any iterable) of string parts already
  ✓ Simple, consistent separator
  ✓ Most common case — this is the canonical Python idiom
  ✓ Readability: "\\n".join(lines) is self-documenting

  USE StringIO WHEN:
  ─────────────────────────────────────────────────────────────────
  ✓ You can't pre-collect all parts (streaming, conditional writes)
  ✓ You need to build a string across multiple functions/calls
  ✓ You need file-like interface (seek, tell, pass as a file handle)
  ✓ Mixing different separators per write

  NEVER USE += in a loop WHEN:
  ─────────────────────────────────────────────────────────────────
  ✗ Building any non-trivial string (more than ~100 iterations)
  ✗ Processing NLP data (documents, corpora, prompt construction)
  ✗ Performance matters — it will degrade quadratically

  THE INTERVIEW ANSWER:
  ─────────────────────────────────────────────────────────────────
  "Python strings are immutable. s += chunk allocates a new string
   and copies all existing bytes every iteration — that's O(N²).
   The fix is ''.join(parts) which calculates total length, allocates
   once, and fills in O(N). CPython has a refcount optimisation that
   can make += look fast in simple scripts, but it breaks as soon as
   you have two references to the string — so never rely on it."
""")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  INTERVIEW PREP · Q11 · THE STRING CONCATENATION TRAP                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Core question: "What's wrong with s = s + chunk in a loop?"
  Core concept:  Immutability forces a full copy on every iteration → O(N²).

  We'll prove this with actual timing data across multiple sizes.
""")

    demo_scaling()
    demo_timed(TEST_SIZES[-1])    # run the decorator demo at the largest size
    demo_cpython_gotcha()
    demo_decision()

    print(f"""
{"=" * 72}
  SUMMARY 

  "Strings in Python are immutable. s = s + chunk creates a brand
   new string object every iteration and copies everything — that's
   O(N²). The fix is ''.join(list_of_parts): it pre-calculates total
   length, allocates once, and fills the buffer in a single O(N) pass.

   For conditional or streaming builds where you can't pre-collect
   into a list, io.StringIO gives a file-like buffer with O(1) appends.

   CPython sometimes optimises += when refcount is 1, but this is an
   implementation detail that breaks the moment you have two references
   to the string — never rely on it."
{"=" * 72}
""")


if __name__ == "__main__":
    main()