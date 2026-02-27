"""
=============================================================================
  PYTHON ENGINEERING CHALLENGE
  NLP Pipeline Log Analyser
  Time target: 10–15 minutes
=============================================================================

  SETUP
  ─────
  You have one file to work with: nlp_pipeline.log
  It is a log from a multi-service NLP pipeline.

  Each line follows this format:

      YYYY-MM-DD HH:MM:SS [LEVEL] service=NAME message text...

  Example lines:
      2024-01-15 08:42:01 [ERROR] service=tokenise vocab lookup failed: token 'bert' not in index
      2024-01-15 08:42:03 [WARN]  service=search  slow tokenisation: 4821ms exceeds threshold
      2024-01-15 08:42:05 [INFO]  service=embed   processed 1873 tokens in request
      2024-01-15 08:42:07 [DEBUG] service=rank    attention head 42 activated

  The log contains ~120,000 lines across multiple services and log levels.

=============================================================================
  YOUR TASK
=============================================================================

  Write a function called main() that produces the following output:

  ── 1. TOP ERRORS BY SERVICE ─────────────────────────────────────────────
  A summary showing, for each service, how many ERROR lines it produced.
  Print them sorted highest to lowest. Only include services with at least
  one ERROR.

  ── 2. SLOWEST SERVICES (from timing data) ───────────────────────────────
  Some log messages contain a timing value in milliseconds, written as
  an integer immediately followed by "ms" somewhere in the message text.
  Examples:
      "model inference timeout after 4821ms"
      "pipeline stage 'corpus' completed in 5515ms"
      "slow tokenisation: 312ms exceeds threshold"

  For each service, collect all ms values you can extract. Compute the
  mean ms per service and print them sorted slowest to fastest.
  Only include services where at least one ms value was found.

  ── 3. ERROR RATE OVER TIME ──────────────────────────────────────────────
  Group all log lines by date (YYYY-MM-DD). For each date, compute:
      error_rate = ERROR lines on that date / total lines on that date

  Print each date and its error rate as a percentage, sorted by date.

=============================================================================
  CONSTRAINTS & EXPECTATIONS
=============================================================================

  • The log file is larger than you would want to load naively all at once.
    Think carefully about how you read it.

  • You must not use pandas or any external library. stdlib only.
    (re, collections, datetime, statistics, itertools — all fine.)

  • String building, if you do any, should be done correctly.

  • If there is work that could benefit from concurrency, consider it.
    You do not have to implement it, but be ready to discuss the choice.

  • Your code should be clean enough that an interviewer can read it.

=============================================================================
  DELIVERABLE FORMAT
=============================================================================

  Output should look roughly like this (your numbers will differ):

      === TOP ERRORS BY SERVICE ===
      tokenise   :   987 errors
      search     :   823 errors
      ...

      === MEAN LATENCY BY SERVICE (ms) ===
      embed      :  5821.3 ms avg
      rank       :  5790.1 ms avg
      ...

      === ERROR RATE BY DATE ===
      2024-01-01 :  8.2%
      2024-02-01 :  7.9%
      ...

=============================================================================
  SCAFFOLDING — do not modify the decorator or the __main__ block.
  Write your solution inside main() and any helper functions you define
  above it. LOG_FILE is set for you.
=============================================================================
"""

import time
import functools
import gc
import tracemalloc


# ── Benchmarking decorator ────────────────────────────────────────────────────
def benchmark(fn):
    """
    Wraps a function to report wall-clock time and peak heap memory
    after it returns. Drop this on any top-level function.

    Uses functools.wraps to preserve the wrapped function's identity —
    its __name__ and __doc__ are visible to debuggers and introspection tools.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()

        result = fn(*args, **kwargs)

        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\n{'─' * 60}")
        print(f"  ⏱  {fn.__name__}() completed in {elapsed:.3f}s")
        print(f"  🧠 peak memory: {peak / 1_048_576:.2f} MB")
        print(f"{'─' * 60}\n")
        return result
    return wrapper


# ── Your solution goes here ───────────────────────────────────────────────────
import re
from collections import Counter, defaultdict

SERVICE_PATTERN = re.compile(r'service=(\S+)')
MS_PATTERN = re.compile(r'(\d+)ms\b')
LOG_FILE = "nlp_pipeline.log"

def stream_lines(path):
    with open(path, encoding="utf-8") as f:
        yield from f

@benchmark
def main():
    error_counts   = Counter()
    latency_totals = defaultdict(float)
    latency_counts = defaultdict(int)
    date_totals    = defaultdict(int)
    date_errors    = defaultdict(int)

    for line in stream_lines(LOG_FILE):
        parts   = line.split()
        if len(parts) < 3:
            continue
        date    = parts[0]                    # 'YYYY-MM-DD'
        level   = parts[2].strip('[]')        # '[ERROR]' → 'ERROR'
        svc_m   = SERVICE_PATTERN.search(line)
        if not svc_m:
            continue
        service = svc_m.group(1)
        ms_m    = MS_PATTERN.search(line)

        date_totals[date] += 1
        if level == "ERROR":
            error_counts[service] += 1
            date_errors[date]     += 1
        if ms_m:
            latency_totals[service] += int(ms_m.group(1))
            latency_counts[service] += 1

    # --- output ---
    print("=== TOP ERRORS BY SERVICE ===")
    for svc, n in error_counts.most_common():
        print(f"  {svc:<12}: {n:>6} errors")

    print("\n=== MEAN LATENCY BY SERVICE (ms) ===")
    means = sorted(
        ((s, latency_totals[s] / latency_counts[s]) for s in latency_totals),
        key=lambda x: x[1], reverse=True
    )
    for svc, avg in means:
        print(f"  {svc:<12}: {avg:>8.1f} ms avg")

    print("\n=== ERROR RATE BY DATE ===")
    for date in sorted(date_totals):
        rate = date_errors.get(date, 0) / date_totals[date] * 100
        print(f"  {date} : {rate:>5.1f}%")
        
# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
