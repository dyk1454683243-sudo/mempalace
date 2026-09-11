#!/usr/bin/env python3
"""fit_calibrator.py — fit a confidence calibrator on labeled search hits.

Issue techempower-org/mempalace#250. PR #167 shipped the calibration
plumbing (``mempalace/calibration.py`` + ``MEMPALACE_CALIBRATION_PATH``
wiring); this script is the missing piece — running the eval harness on
our own corpus, fitting an isotonic calibrator, and reporting Brier /
Expected Calibration Error before vs. after.

Per the test-on-our-corpus rule (feedback_test_retrieval_against_our_corpus):
calibration distributions don't transfer between corpora. The shipped
calibrator is named ``jp_realm_v1`` and is **not** wired as a default —
it's specific to JP's production palace. Users opt in by pointing
``MEMPALACE_CALIBRATION_PATH`` at it, or re-run this script against
their own corpus.

How labels are assigned
-----------------------
Each probe is ``(query, expected_source_file)``. For every vector hit
(``similarity is not None``) returned by ``search_memories``, we label

    relevant = 1  iff  basename(hit.source_file) == basename(expected)

The basename rule matches ``rank_of_target`` in ``eval_fusion_ab.py`` —
same identity used by every other harness in the repo. The label is
sparse-but-truthful: most vector hits in a large production corpus are
*not* the expected file, so most labels are 0. That's what calibration
expects — a well-fit isotonic map will route low-similarity hits to low
confidence and high-similarity hits to higher confidence, even when the
positive class is rare.

How the metrics are reported
----------------------------
* ``Brier_before`` / ``ECE_before`` — using the raw ``similarity`` value
  directly as a confidence. This is the baseline the system would emit
  if no calibrator were configured (and we treated similarity as a
  probability — which the codebase deliberately does **not** do; we
  only emit a ``confidence`` field when a calibrator is present).
* ``Brier_after`` / ``ECE_after`` — using ``cal.apply(similarity)`` for
  the freshly-fit calibrator on the same pairs.

The after numbers are an in-sample fit; a held-out comparison would be
more rigorous but the probe set is 200 items and isotonic regression on
that scale has very few effective degrees of freedom (one breakpoint
per pooled block — typically < 30), so the optimism is small.

Usage
-----
::

    source ~/.config/palace-daemon/env  # sets PALACE_API_KEY + PALACE_DAEMON_URL
    python scripts/fit_calibrator.py \\
        --probes scripts/probes_v2_git_derived.json \\
        --limit-per-probe 20 \\
        --out mempalace/data/calibrators/jp_realm_v1.json \\
        --label "jp_realm_v1"

The script is CPU-only and offline-deterministic *given* a fixed
``(probes, hits)`` input — daemon nondeterminism (e.g. encoder updates,
ANN drift) is the only source of run-to-run variation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── pure helpers (unit-testable, no I/O) ──────────────────────────────────


def label_from_hit(hit: dict, expected_source_file: str) -> int:
    """Return ``1`` if the hit's source-file basename matches ``expected``.

    Basename match is the same identity rule used by ``rank_of_target`` in
    ``scripts/eval_fusion_ab.py`` — a probe's expected path and a hit's
    ``source_file`` compare equal regardless of directory prefix.
    """
    sf = (hit.get("source_file") or "").strip()
    return 1 if Path(sf).name == Path(expected_source_file).name else 0


def collect_pairs(
    hits: Sequence[dict],
    expected_source_file: str,
) -> list:
    """Extract ``(similarity, relevant)`` pairs from one probe's hits.

    Only hits with ``similarity is not None`` participate — BM25-only and
    graph-only hits carry no vector score and can't be calibrated. The
    field is None when ``matched_via`` is ``bm25_*`` or ``graph_*``.
    """
    pairs: list = []
    for h in hits:
        sim = h.get("similarity")
        if sim is None:
            continue
        pairs.append((float(sim), label_from_hit(h, expected_source_file)))
    return pairs


@dataclass
class CalibrationReport:
    """Numbers + provenance for a single fit run."""

    n_probes: int
    n_pairs: int
    n_positive: int
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    label: str
    out_path: str
    breakpoints: int = 0
    elapsed_secs: float = 0.0
    per_probe_summary: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_probes": self.n_probes,
            "n_pairs": self.n_pairs,
            "n_positive": self.n_positive,
            "positive_rate": (round(self.n_positive / self.n_pairs, 4) if self.n_pairs else 0.0),
            "brier_before": round(self.brier_before, 4),
            "brier_after": round(self.brier_after, 4),
            "brier_delta": round(self.brier_after - self.brier_before, 4),
            "ece_before": round(self.ece_before, 4),
            "ece_after": round(self.ece_after, 4),
            "ece_delta": round(self.ece_after - self.ece_before, 4),
            "label": self.label,
            "out_path": self.out_path,
            "breakpoints": self.breakpoints,
            "elapsed_secs": round(self.elapsed_secs, 2),
        }


def compute_before_after(pairs: Sequence[tuple], cal) -> tuple:
    """Return ``(brier_before, ece_before, brier_after, ece_after)``.

    Before: raw similarity used as confidence (the naive identity baseline
    a system without a calibrator would emit if it pretended similarity
    were a probability — which mempalace deliberately does not).
    After: ``cal.apply(similarity)`` on the same pairs.
    """
    from mempalace.calibration import brier_score, expected_calibration_error

    if not pairs:
        raise ValueError("compute_before_after: empty pairs is undefined")
    sims = [p[0] for p in pairs]
    outs = [p[1] for p in pairs]
    confs_after = [cal.apply(s) for s in sims]
    return (
        brier_score(sims, outs),
        expected_calibration_error(sims, outs, n_bins=10),
        brier_score(confs_after, outs),
        expected_calibration_error(confs_after, outs, n_bins=10),
    )


# ── orchestration (pure given a search_fn) ────────────────────────────────


def run_fit(
    probes: Sequence[dict],
    search_fn: Callable[[str, int], list],
    limit_per_probe: int,
    label: str,
    out_path: Optional[str] = None,
) -> tuple:
    """Drive the fit. ``search_fn(query, limit) -> list_of_hits``.

    Returns ``(Calibrator, CalibrationReport)``. ``search_fn`` is injected
    so tests can run this with a stub and never touch the daemon. When
    ``out_path`` is set the calibrator is also saved to disk.
    """
    from mempalace.calibration import fit_calibrator

    t0 = time.time()
    all_pairs: list = []
    per_probe: list = []
    for i, p in enumerate(probes):
        query = p["query"]
        expected = p["expected"]
        hits = search_fn(query, limit_per_probe) or []
        pairs = collect_pairs(hits, expected)
        all_pairs.extend(pairs)
        per_probe.append(
            {
                "i": i,
                "query": query,
                "expected": expected,
                "n_hits": len(hits),
                "n_vec_hits": len(pairs),
                "n_positive_vec": sum(1 for _, r in pairs if r),
            }
        )

    if not all_pairs:
        raise RuntimeError(
            "run_fit: no vector hits collected across all probes — "
            "is the daemon returning matched_via=bm25_postgres only?"
        )

    cal = fit_calibrator(all_pairs, source=label)
    brier_before, ece_before, brier_after, ece_after = compute_before_after(all_pairs, cal)

    report = CalibrationReport(
        n_probes=len(probes),
        n_pairs=len(all_pairs),
        n_positive=sum(1 for _, r in all_pairs if r),
        brier_before=brier_before,
        brier_after=brier_after,
        ece_before=ece_before,
        ece_after=ece_after,
        label=label,
        out_path=out_path or "",
        breakpoints=len(cal.x),
        elapsed_secs=time.time() - t0,
        per_probe_summary=per_probe,
    )

    if out_path:
        cal.save(out_path)

    return cal, report


# ── daemon HTTP shim (only path that hits the network) ────────────────────


def _make_daemon_search_fn(
    daemon_url: str,
    api_key: str,
    timeout: float = 60.0,
    retries: int = 3,
    retry_backoff: float = 5.0,
):
    """Return a ``(query, limit) -> hits`` closure backed by daemon ``/search``.

    Kept in this module rather than reaching into ``mempalace.searcher`` so
    the script remains a standalone harness — ``search_memories`` would
    need a configured palace path / backend and bypassing the daemon
    misses the production-corpus signal we're calibrating against.

    Each query gets ``retries`` attempts with linear ``retry_backoff *
    attempt`` second sleeps between them. The daemon under contention
    (GPU pinned by a backfill or another harness) commonly returns slow
    enough to trip the default urllib timeout; one transient stall must
    not nuke a 15-minute fit.
    """
    base = daemon_url.rstrip("/")

    def search(query: str, limit: int) -> list:
        q = urllib.parse.quote(query, safe="")
        url = f"{base}/search?q={q}&limit={int(limit)}"
        last_exc: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers={"X-API-Key": api_key})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = json.load(r)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return data.get("results") or []
                return []
            except (TimeoutError, OSError, urllib.error.URLError) as e:
                last_exc = e
                if attempt < retries:
                    time.sleep(retry_backoff * attempt)
                    continue
                # Final attempt failed — return [] so the fit continues
                # rather than aborting on a transient daemon hiccup. The
                # caller will see this probe in per_probe_summary as
                # n_hits=0 and the overall fit just has slightly fewer
                # pairs to work with.
                print(
                    f"warn: {retries} attempts to /search failed for "
                    f"query {query!r}: {e}; recording zero hits and continuing",
                    file=sys.stderr,
                )
                return []
        # Unreachable, but keep mypy/ruff happy.
        if last_exc is not None:
            return []
        return []

    return search


# ── CLI ───────────────────────────────────────────────────────────────────


def _load_probes(path: str) -> list:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "probes" in raw:
        return list(raw["probes"])
    if isinstance(raw, list):
        return raw
    raise ValueError(f"probe file {path}: expected a list or a dict with a 'probes' key")


def _print_summary(report: CalibrationReport) -> None:
    d = report.as_dict()
    print()
    print("=== Calibration fit report ===")
    print(f"  Label              : {d['label']}")
    print(f"  Probes             : {d['n_probes']}")
    print(f"  Vector pairs       : {d['n_pairs']}")
    print(f"  Positive rate      : {d['positive_rate']:.4f}  ({d['n_positive']} positives)")
    print(f"  Breakpoints (PAV)  : {d['breakpoints']}")
    print(
        f"  Brier  before/after: {d['brier_before']:.4f}  →  {d['brier_after']:.4f}  "
        f"(Δ {d['brier_delta']:+.4f}; lower is better)"
    )
    print(
        f"  ECE    before/after: {d['ece_before']:.4f}  →  {d['ece_after']:.4f}  "
        f"(Δ {d['ece_delta']:+.4f}; lower is better)"
    )
    print(f"  Elapsed            : {d['elapsed_secs']}s")
    if d["out_path"]:
        print(f"  Calibrator JSON    : {d['out_path']}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probes",
        default=str(_REPO_ROOT / "scripts" / "probes_v2_git_derived.json"),
        help="JSON probe set ({probes:[{query,expected,why},...]} or a list).",
    )
    parser.add_argument(
        "--limit-per-probe",
        type=int,
        default=20,
        help="Hits to request per probe (default 20).",
    )
    parser.add_argument(
        "--out",
        default=str(_REPO_ROOT / "mempalace" / "data" / "calibrators" / "jp_realm_v1.json"),
        help="Where to write the fitted Calibrator JSON.",
    )
    parser.add_argument(
        "--label",
        default="jp_realm_v1",
        help="Provenance label baked into the calibrator JSON.",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Optional path to write a full report JSON alongside the calibrator.",
    )
    parser.add_argument(
        "--probe-limit",
        type=int,
        default=0,
        help="Cap probes processed (0 = all). Useful for smoke tests.",
    )
    parser.add_argument(
        "--daemon-url",
        default=os.environ.get("PALACE_DAEMON_URL", ""),
        help="Palace daemon base URL (default env PALACE_DAEMON_URL).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PALACE_API_KEY", ""),
        help="Daemon X-API-Key (default env PALACE_API_KEY).",
    )
    args = parser.parse_args(argv)

    if not args.daemon_url:
        print("No --daemon-url and PALACE_DAEMON_URL unset.", file=sys.stderr)
        return 2
    if not args.api_key:
        print("No --api-key and PALACE_API_KEY unset.", file=sys.stderr)
        return 2

    probes = _load_probes(args.probes)
    if args.probe_limit > 0:
        probes = probes[: args.probe_limit]

    print(f"Probes:      {len(probes)} from {args.probes}")
    print(f"Daemon:      {args.daemon_url}")
    print(f"Limit/probe: {args.limit_per_probe}")
    print(f"Output:      {args.out}")
    print()

    search_fn = _make_daemon_search_fn(args.daemon_url, args.api_key)
    cal, report = run_fit(
        probes=probes,
        search_fn=search_fn,
        limit_per_probe=args.limit_per_probe,
        label=args.label,
        out_path=args.out,
    )
    _print_summary(report)

    if args.report_json:
        d = report.as_dict()
        d["per_probe"] = report.per_probe_summary
        Path(args.report_json).write_text(json.dumps(d, indent=2), encoding="utf-8")
        print(f"  Full report JSON   : {args.report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
