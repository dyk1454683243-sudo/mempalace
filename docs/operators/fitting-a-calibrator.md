# Fitting a confidence calibrator

`search_memories` can attach a `confidence` field — a calibrated
`P(this hit is relevant)` in `[0, 1]` — to every vector hit it returns.
The field is **only** populated when a calibrator JSON is configured;
unconfigured callers see no `confidence` field at all. The system never
fakes a probability from raw cosine similarity.

This doc walks through what the calibrator does, why it's a per-corpus
artifact (not a one-size-fits-all default), how to fit one with
`scripts/fit_calibrator.py`, and how to interpret the resulting
ECE / Brier numbers.

## Background — why a calibrator at all

`mempalace_search` returns each vector hit with a `similarity` field —
defined as `max(0, 1 - cosine_distance)`. It has the **range** of a
probability (`[0, 1]`) but **no statistical guarantee** of being one. A
similarity of `0.6` does not mean "60% of hits at this score are
relevant" — it just means "the embedding model placed query and doc
fairly close in vector space". Whether `0.6` corresponds to 90%, 50%, or
5% relevance is an empirical property of *your* corpus, *your* queries,
and *your* embedding model.

The calibrator (`mempalace/calibration.py`) is an isotonic regression
that maps raw similarity → calibrated `P(relevant)`. It is fit on a
labeled probe set (queries with known-correct expected documents), then
applied at search time as a single bisect per hit (sub-microsecond,
CPU-only, zero new dependencies).

## Why we don't ship a one-size-fits-all default

Cosine-similarity distributions vary by:

* **embedding model** — MiniLM-L6 vs. multi-encoder RRF have different
  geometry; an `0.6` for one is not an `0.6` for the other
* **corpus composition** — code-heavy vs. prose-heavy palaces yield
  different similarity floors
* **query style** — keyword queries cluster at different ranges than
  full-sentence queries
* **drawer chunking** — finer chunks → tighter local matches → higher
  similarities on average

We ship one fitted calibrator (`mempalace/data/calibrators/jp_realm_v1.json`)
as a reference artifact, fit against JP's ~390K-drawer production
corpus. It is **not** wired as a default — pointing every user at it
would shipped a calibrator that's wrong for them. Users opt in by
setting `MEMPALACE_CALIBRATION_PATH`, or by running the fit script
against their own corpus.

## The fit procedure

### Inputs

A **probe set** — a JSON list of `(query, expected_source_file)` pairs
where the query is a natural-language question and the expected file is
the basename of the document that *should* be the top hit.

The fork ships one at `scripts/probes_v2_git_derived.json` (200 probes
auto-derived from this repo's git history). The format:

```json
{
  "probes": [
    {
      "query": "Trigram index + complete schema in _create_table + test isolation",
      "expected": "postgres.py",
      "why": "5f9d087 fix(migrate,backend): trigram index + complete schema..."
    },
    ...
  ]
}
```

Labels are basename-only — `Path(hit.source_file).name ==
Path(expected).name` is the relevance test. This matches the identity
rule used by every other harness in the repo
(`rank_of_target` in `scripts/eval_fusion_ab.py`).

### Running the script

```bash
source ~/.config/palace-daemon/env  # sets PALACE_API_KEY + PALACE_DAEMON_URL
python scripts/fit_calibrator.py \
    --probes scripts/probes_v2_git_derived.json \
    --limit-per-probe 20 \
    --out mempalace/data/calibrators/<your_label>.json \
    --label "<your_label>"
```

The script issues one `GET /search?q=...&limit=20` per probe against the
configured daemon, collects every vector hit's `(similarity, relevant?)`
pair, and fits an isotonic regression. Total runtime scales with
`n_probes × per-query daemon latency`; 200 probes at ~5s each ≈ 15min.

### Wiring it in

```bash
export MEMPALACE_CALIBRATION_PATH=$PWD/mempalace/data/calibrators/<your_label>.json
mempalace search "some query"  # hits now carry confidence:
```

Or in `~/.mempalace/config.json`:

```json
{ "calibration_path": "/abs/path/to/<your_label>.json" }
```

A long-lived process (the daemon) picks up an in-place re-fit without a
restart — the loader keys its cache on `(path, mtime)`. Drop a new JSON
at the same path and the next search query uses it.

## What the report tells you

The script prints (and optionally writes to `--report-json`) a fit
report:

```
=== Calibration fit report ===
  Label              : jp_realm_v1
  Probes             : 200
  Vector pairs       : <N>
  Positive rate      : <P>  (<K> positives)
  Breakpoints (PAV)  : <M>
  Brier  before/after: 0.XXXX  →  0.YYYY  (Δ ...; lower is better)
  ECE    before/after: 0.XXXX  →  0.YYYY  (Δ ...; lower is better)
```

* **Brier score** — mean squared error between predicted confidence and
  binary outcome. Bounded `[0, 1]`; `0` is perfect, `0.25` is the score
  of always predicting `0.5`. **Lower is better.**
* **ECE (Expected Calibration Error)** — mean absolute gap between
  bucket-mean confidence and bucket-empirical relevance, weighted by
  bucket size. `0` is perfectly calibrated. **Lower is better.**
* **Before** — using raw similarity as the confidence value (the
  baseline a naive system would emit if it pretended similarity were a
  probability — which mempalace deliberately does not).
* **After** — using the fitted isotonic map on the same pairs.

### Caveats

* **In-sample fit.** The after-numbers are evaluated on the same pairs
  the calibrator was fit on. A held-out comparison is more rigorous, but
  the probe set is small (200) and isotonic regression on that scale has
  very few effective degrees of freedom (one breakpoint per pooled
  block), so optimism bias is small.
* **Sparse positives.** When the expected file is rare in the corpus,
  the positive rate is low (often <2%) and the calibrator will route
  most similarities to a low confidence. That is the **correct** answer
  given the data — high-confidence hits in a corpus where most candidate
  documents are irrelevant *should* be rare.
* **Probe quality dominates.** A probe set whose `expected` files don't
  exist in the corpus, or whose `query` text doesn't reflect how real
  users phrase the question, will fit a calibrator that doesn't
  generalize. Use git-derived probes (auto-generated from your own
  commits) or a hand-curated set you trust.

## Refitting

Re-run the script whenever the corpus changes materially — after a big
mine, after switching embedding models, after introducing a new wing
with different content type. The cached calibrator picks up the new
JSON on next load (mtime-keyed).

## See also

* `mempalace/calibration.py` — the isotonic + PAV implementation
* `tests/test_calibration.py` — unit tests for the math
* `tests/test_searcher_confidence.py` — end-to-end integration test
* `docs/research/uncertainty-aware-retrieval.md` — design rationale
* Issue [techempower-org/mempalace#250] — the fit-and-commit follow-up
  to the calibration plumbing (PR #167)
