"""
=============================================================================
  INTERVIEW PREP · Q12 · THE GLOBAL INTERPRETER LOCK (GIL)
  Run:  python q12_gil.py
=============================================================================

  The Question You'll Be Asked:
    "Why isn't my multi-threaded Python faster on a 16-core machine?"

  The Wrong Answer: My code must have a bug.
  The Right Answer: The GIL prevents multiple threads from running Python
                    bytecode simultaneously. For CPU-bound work, use
                    multiprocessing instead — each process has its own GIL.
  The Great Answer: Know WHEN threads ARE the right tool (I/O-bound tasks),
                    understand why the GIL exists, and know that C extensions
                    like NumPy/PyTorch release it — so the rule has nuance.

  This file teaches you:
    1. What the GIL is and WHY it exists (not just "it's a lock")
    2. Measured proof that threads don't help CPU-bound work
    3. Measured proof that multiprocessing does
    4. When threads ARE correct: I/O-bound tasks, GIL release during I/O
    5. The modern concurrent.futures API and the decision table
=============================================================================
"""

import time
import threading
import multiprocessing
import functools
import gc
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# =============================================================================
#  TUNABLE PARAMETERS
# =============================================================================

CPU_WORKLOAD  = 3_000_000   # Iterations per worker — pure Python busy loop
NUM_WORKERS   = 4           # Threads / processes to spawn
TIMING_REPS   = 2           # Repeat count for averaged timings

# Simulated I/O delay in seconds per "task" for the I/O-bound demo
# This represents a database query, API call, or network request wait
IO_TASK_DELAY = 0.15
IO_TASK_COUNT = 8           # Number of concurrent "I/O tasks" to run

# =============================================================================
#  DISPLAY HELPERS
# =============================================================================

DIVIDER = "\n" + "=" * 72
SECTION = "\n" + "-" * 60

def header(title):      print(f"{DIVIDER}\n  {title}{DIVIDER}")
def section(title):     print(f"{SECTION}\n  {title}{SECTION}")
def result(label, val): print(f"    → {label}: {val}")
def note(msg):          print(f"    ℹ {msg}")
def trap(msg):          print(f"    ⚠  TRAP: {msg}")
def fix(msg):           print(f"    ✓  FIX:  {msg}")
def show_t(lbl, s):     print(f"    ⏱  {lbl}: {s:.3f}s")
def sep():              print()

# =============================================================================
#  TIMING DECORATOR
# =============================================================================

def timed(label=None, repeat=1):
    """
    Same decorator pattern from Q10 and Q11 — wrap any function to measure
    its wall-clock time. Reusing the same pattern across your codebase is
    itself good Python engineering: DRY, consistent, predictable.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            name = label or fn.__name__
            times = []
            val   = None
            for _ in range(repeat):
                gc.collect()
                t0  = time.perf_counter()
                val = fn(*args, **kwargs)
                times.append(time.perf_counter() - t0)
            avg = sum(times) / len(times)
            print(f"    ⏱  {name} (avg {repeat}x): {avg:.3f}s")
            return val
        return wrapper
    return decorator


# =============================================================================
#  WORKER FUNCTIONS
#
#  These must be defined at module level (not inside functions) for
#  multiprocessing to work on Windows and macOS. The 'spawn' start
#  method used on those platforms imports this module fresh in each
#  child process — nested functions don't survive that import.
# =============================================================================

def cpu_worker(n: int) -> int:
    """
    Pure Python CPU-bound work.
    Simulates NLP feature extraction: a tight counting loop
    that keeps the CPU busy with no I/O or waiting.
    """
    count = 0
    for _ in range(n):
        count += 1
    return count


def io_worker(task_id: int) -> str:
    """
    Simulated I/O-bound work.
    Represents a database query, REST API call, or S3 read —
    tasks where the CPU spends most of its time waiting, not computing.
    """
    time.sleep(IO_TASK_DELAY)    # ← stands in for a real network wait
    return f"task-{task_id} done"


# =============================================================================
#  PART 1 — WHAT IS THE GIL AND WHY DOES IT EXIST?
# =============================================================================

def explain_the_gil():
    section("📚  WHAT IS THE GIL?")

    print("""
  The Global Interpreter Lock (GIL) is a mutex — a mutual exclusion lock —
  inside CPython (the standard Python interpreter).

  It ensures that only ONE thread executes Python bytecode at a time,
  even on a machine with 16 cores.

  WHY DOES IT EXIST?
  ─────────────────────────────────────────────────────────────────
  CPython manages memory using reference counting. Every Python object
  has a counter: how many variables currently point to it. When that
  count hits zero, the object is freed.

  Without the GIL, two threads could simultaneously modify the same
  object's reference count, corrupt it, and cause a use-after-free
  memory error or crash. The GIL prevents this by only allowing one
  thread to touch Python objects at a time.

  It's a pragmatic engineering trade-off: simpler, safer memory
  management at the cost of true thread parallelism.

  WHAT THE GIL MEANS FOR YOU:
  ─────────────────────────────────────────────────────────────────
  ✗ Python threads CANNOT run CPU-bound code in true parallel
  ✗ Adding threads to a CPU-bound task often makes it SLOWER
    (GIL contention + thread context switching overhead)

  ✓ Python threads CAN run concurrently for I/O-bound work
    (the GIL is RELEASED during blocking I/O calls)
  ✓ C extensions (NumPy, PyTorch, spaCy) release the GIL
    during their heavy computation — so threading can help there
""")


# =============================================================================
#  PART 2 — CPU-BOUND: THREADS FAIL, PROCESSES WIN
# =============================================================================

def demo_cpu_bound():
    section("🔬  CPU-BOUND: Threads vs Processes vs Serial")

    note(f"Task: run cpu_worker({CPU_WORKLOAD:,} iterations) × {NUM_WORKERS} workers")
    note("This simulates NLP tokenisation, regex parsing, feature extraction.")
    sep()

    # --- SERIAL BASELINE ---
    print("  [ Serial — no concurrency ]")
    t0 = time.perf_counter()
    for _ in range(NUM_WORKERS):
        cpu_worker(CPU_WORKLOAD)
    t_serial = time.perf_counter() - t0
    show_t("serial baseline", t_serial)
    sep()

    # --- THREADING ---
    print("  [ Threading — shared GIL, no true parallelism for CPU work ]")
    trap("Threads contend for the GIL. Only one runs at a time.")
    trap("Context switching adds overhead. Often slower than serial.")
    print()

    t0 = time.perf_counter()
    threads = [
        threading.Thread(target=cpu_worker, args=(CPU_WORKLOAD,))
        for _ in range(NUM_WORKERS)
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    t_threaded = time.perf_counter() - t0
    show_t(f"threaded ({NUM_WORKERS} threads)", t_threaded)
    thread_vs_serial = t_threaded / t_serial
    result("vs serial", f"{thread_vs_serial:.2f}× {'(SLOWER — GIL overhead)' if thread_vs_serial > 1.05 else '(negligible gain)'}")
    sep()

    # --- MULTIPROCESSING ---
    print("  [ Multiprocessing — each process has its OWN GIL ]")
    fix("Separate processes = separate interpreters = true parallelism.")
    fix("Workers run simultaneously on different cores.")
    print()

    t0 = time.perf_counter()
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        pool.map(cpu_worker, [CPU_WORKLOAD] * NUM_WORKERS)
    t_mp = time.perf_counter() - t0
    show_t(f"multiprocessing ({NUM_WORKERS} processes)", t_mp)
    mp_speedup = t_serial / t_mp
    result("speedup vs serial", f"{mp_speedup:.2f}× faster")
    sep()

    note("Speedup < N because: process spawn time + IPC serialisation overhead.")
    note("On a real NLP task (regex, tokenisation, parsing) the gap is larger")
    note("because actual work dominates over the overhead.")


# =============================================================================
#  PART 3 — I/O-BOUND: THREADS ARE CORRECT
# =============================================================================

def demo_io_bound():
    section("✅  I/O-BOUND: Where Threading Is THE Right Answer")

    note("The GIL is RELEASED during blocking I/O — network, disk, DB calls.")
    note(f"Task: {IO_TASK_COUNT} simulated I/O operations, each taking {IO_TASK_DELAY}s")
    note("(think: 8 database queries or 8 API calls)")
    sep()

    # --- SERIAL ---
    print("  [ Serial I/O — one at a time, fully sequential ]")
    t0 = time.perf_counter()
    for i in range(IO_TASK_COUNT):
        io_worker(i)
    t_serial_io = time.perf_counter() - t0
    show_t("serial I/O", t_serial_io)
    result("expected", f"~{IO_TASK_COUNT * IO_TASK_DELAY:.1f}s  (tasks run one after another)")
    sep()

    # --- THREADING FOR I/O ---
    print("  [ Threading for I/O — GIL released during sleep/wait ]")
    fix("While one thread sleeps (waiting on I/O), others run freely.")
    fix("True concurrency for I/O — the GIL is released during the wait.")
    print()

    t0 = time.perf_counter()
    threads = [
        threading.Thread(target=io_worker, args=(i,))
        for i in range(IO_TASK_COUNT)
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    t_threaded_io = time.perf_counter() - t0
    show_t(f"threaded I/O ({IO_TASK_COUNT} threads)", t_threaded_io)
    result("expected", f"~{IO_TASK_DELAY:.2f}s  (all tasks overlap)")
    speedup = t_serial_io / t_threaded_io
    result("actual speedup", f"{speedup:.1f}×")
    sep()

    note("The threaded version completes in roughly the time of ONE task,")
    note("not all of them — because the waiting happens in parallel.")
    note("This is where Python threads genuinely shine.")


# =============================================================================
#  PART 4 — CONCURRENT.FUTURES: THE MODERN API
# =============================================================================

def demo_concurrent_futures():
    section("✅✅  ADVANCED: concurrent.futures — The Modern API")

    note("concurrent.futures gives a unified interface for both threads and")
    note("processes. Swap ThreadPoolExecutor ↔ ProcessPoolExecutor and your")
    note("code structure stays identical. Only the concurrency model changes.")
    sep()

    print("""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    # CPU-bound NLP work → ProcessPoolExecutor (bypass the GIL)
    docs = ["large document..."] * NUM_WORKERS

    def tokenise(doc):
        return doc.split()     # your actual NLP function

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        token_lists = list(ex.map(tokenise, docs))
        #              ↑ blocks until all done, preserves order

    # I/O-bound work → ThreadPoolExecutor (GIL released during I/O)
    with ThreadPoolExecutor(max_workers=IO_TASK_COUNT) as ex:
        futures = list(ex.map(io_worker, range(IO_TASK_COUNT)))

    # The swap is ONE word: Process ↔ Thread
    # Everything else stays the same.
""")

    fix("ex.map() is simpler than managing Pool directly.")
    fix("Results are returned in submission order, not completion order.")
    fix("Context manager handles shutdown/cleanup automatically.")
    sep()

    note("ADVANCED: submit() + as_completed() for heterogeneous tasks")
    print("""
    from concurrent.futures import as_completed

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(io_worker, i): i for i in range(IO_TASK_COUNT)}
        for future in as_completed(futures):     # ← yields as each finishes
            task_id = futures[future]
            result  = future.result()
            print(f"Task {task_id} completed: {result}")
    # Use this when tasks have different durations and you want to
    # process results as soon as they arrive (streaming inference, etc.)
""")


# =============================================================================
#  DECISION TABLE
# =============================================================================

def demo_decision():
    section("📊  THE DECISION TABLE")

    print("""
  ┌─────────────────────────┬────────────────────────────┬────────────────────────────┐
  │ Task type               │ Right tool                 │ Why                        │
  ├─────────────────────────┼────────────────────────────┼────────────────────────────┤
  │ CPU-bound               │ multiprocessing /          │ Each process has its own   │
  │ (regex, tokenisation,   │ ProcessPoolExecutor        │ GIL — true parallelism     │
  │ feature extraction)     │                            │                            │
  ├─────────────────────────┼────────────────────────────┼────────────────────────────┤
  │ I/O-bound               │ threading /                │ GIL released during I/O   │
  │ (API calls, DB reads,   │ ThreadPoolExecutor         │ waits — real concurrency   │
  │ file reads, S3)         │                            │                            │
  ├─────────────────────────┼────────────────────────────┼────────────────────────────┤
  │ Many concurrent         │ asyncio                    │ Single thread, event loop, │
  │ I/O connections         │                            │ zero thread overhead       │
  │ (web scraping, chat)    │                            │                            │
  ├─────────────────────────┼────────────────────────────┼────────────────────────────┤
  │ NumPy / PyTorch /       │ threading CAN work         │ C extensions release the   │
  │ spaCy heavy ops         │                            │ GIL during computation     │
  └─────────────────────────┴────────────────────────────┴────────────────────────────┘

  THE QUICK MENTAL MODEL:
  ─────────────────────────────────────────────────────────────────
  Is the bottleneck the CPU? → multiprocessing
  Is the bottleneck waiting? → threading or asyncio
  Are you using NumPy/PyTorch? → threading might be fine (they release the GIL)
  Lots of short I/O tasks? → asyncio (lowest overhead per task)
""")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  INTERVIEW PREP · Q12 · THE GLOBAL INTERPRETER LOCK (GIL)                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Core question: "Why isn't my multi-threaded Python faster on many cores?"
  Core concept:  The GIL serialises Python bytecode. Know when to use
                 multiprocessing (CPU) vs threading (I/O) vs asyncio (async I/O).

  We'll measure the difference between threads and processes on both
  CPU-bound and I/O-bound tasks so you can see the rule play out live.
""")

    explain_the_gil()
    demo_cpu_bound()
    demo_io_bound()
    demo_concurrent_futures()
    demo_decision()

    print(f"""
{"=" * 72}
  SUMMARY — What to say in your interview

  "The GIL is a mutex in CPython that only lets one thread execute
   Python bytecode at a time, even on multi-core hardware. It exists
   to protect CPython's reference-counted memory model.

   For CPU-bound tasks — regex, tokenisation, parsing — the GIL means
   threads won't help. Use multiprocessing: each process gets its own
   interpreter and its own GIL, so they run truly in parallel.

   For I/O-bound tasks — API calls, database reads, file I/O — the GIL
   is released while the thread waits. Threading gives real concurrency
   here. concurrent.futures.ThreadPoolExecutor is the clean modern API.

   One nuance: NumPy, PyTorch, and spaCy release the GIL during their
   heavy C-level computation, so threading can work for those even on
   CPU-intensive code — because the hot path isn't Python bytecode."
{"=" * 72}
""")


if __name__ == "__main__":
    # Required on Windows/macOS — 'spawn' start method needs this guard
    # to prevent recursive subprocess spawning when the module is imported
    multiprocessing.freeze_support()
    main()