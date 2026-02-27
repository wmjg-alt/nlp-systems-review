"""Generate the challenge corpus file."""
import random

random.seed(42)

SERVICES = ["auth", "search", "recommend", "ingest", "parse", "tokenise", "embed", "rank"]
LEVELS   = ["ERROR", "WARN", "INFO", "DEBUG"]
WEIGHTS  = [0.08, 0.17, 0.55, 0.20]

MESSAGES = {
    "ERROR": [
        "model inference timeout after {ms}ms",
        "vocab lookup failed: token '{tok}' not in index",
        "embedding dimension mismatch: expected 768 got {ms}",
        "null pointer in tokeniser pipeline at offset {ms}",
        "OOM during batch encoding: batch_size={ms}",
    ],
    "WARN": [
        "slow tokenisation: {ms}ms exceeds threshold",
        "cache miss rate high: {ms}% in last window",
        "fallback to subword for unknown token '{tok}'",
        "retry {ms} on downstream embedding service",
        "deprecated pipeline stage '{tok}' still in use",
    ],
    "INFO": [
        "processed {ms} tokens in request",
        "pipeline stage '{tok}' completed in {ms}ms",
        "loaded vocab of {ms} entries",
        "batch {ms} dispatched to embedding queue",
        "request '{tok}' routed to ranker",
    ],
    "DEBUG": [
        "attention head {ms} activated",
        "token '{tok}' mapped to id {ms}",
        "cache hit for key '{tok}'",
        "normalised input length: {ms} chars",
        "subword split produced {ms} pieces",
    ],
}

TOKENS = ["transformer", "bert", "gpt", "corpus", "stopword", "lemma",
          "bigram", "trigram", "tfidf", "cosine", "embedding", "softmax",
          "attention", "decoder", "encoder", "vocab", "tokeniser", "padding"]

lines = []
for i in range(120_000):
    ts_sec  = 1_700_000_000 + i * 2
    h, rem  = divmod(ts_sec % 86400, 3600)
    m, s    = divmod(rem, 60)
    date    = f"2024-{(i//43800)%12+1:02d}-{(i//1440)%28+1:02d}"
    ts      = f"{date} {h:02d}:{m:02d}:{s:02d}"
    level   = random.choices(LEVELS, WEIGHTS)[0]
    service = random.choice(SERVICES)
    msg_tpl = random.choice(MESSAGES[level])
    msg     = msg_tpl.format(ms=random.randint(1, 9999), tok=random.choice(TOKENS))
    lines.append(f"{ts} [{level}] service={service} {msg}\n")

with open("nlp_pipeline.log", "w") as f:
    f.writelines(lines)

size_mb = sum(len(l) for l in lines) / 1_048_576
print(f"Written {len(lines):,} lines  ({size_mb:.1f} MB)  →  nlp_pipeline.log")

