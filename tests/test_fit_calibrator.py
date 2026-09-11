"""Tests for scripts/fit_calibrator.py (techempower-org/mempalace#250).

Covers the pure pieces — label assignment, pair extraction, the
``run_fit`` orchestration with a stubbed ``search_fn`` — so we exercise
the harness without ever touching the daemon, the network, or the live
corpus.

The daemon HTTP shim (``_make_daemon_search_fn``) is intentionally
**not** tested here — it's a thin urllib wrapper whose only contract is
"GET /search with X-API-Key and return data['results']". Exercising it
would need either a real daemon or a mocked HTTP layer; either is
out-of-scope for unit coverage. The orchestration tests below feed a
stub directly into ``run_fit`` so the harness logic is fully covered.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_FIT_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fit_calibrator.py"

# Load the script as a module — it's not on sys.path by default because
# scripts/ isn't a package. importlib.util keeps the test self-contained.
_spec = importlib.util.spec_from_file_location("fit_calibrator", _FIT_SCRIPT)
fit_calibrator = importlib.util.module_from_spec(_spec)
sys.modules["fit_calibrator"] = fit_calibrator
_spec.loader.exec_module(fit_calibrator)


class TestLabelFromHit:
    def test_basename_match_yields_one(self):
        hit = {"source_file": "/abs/path/to/searcher.py", "similarity": 0.42}
        assert fit_calibrator.label_from_hit(hit, "searcher.py") == 1

    def test_basename_match_ignores_directories(self):
        hit = {"source_file": "mempalace/searcher.py", "similarity": 0.42}
        assert fit_calibrator.label_from_hit(hit, "/totally/other/dir/searcher.py") == 1

    def test_non_match_yields_zero(self):
        hit = {"source_file": "mempalace/cli.py", "similarity": 0.42}
        assert fit_calibrator.label_from_hit(hit, "searcher.py") == 0

    def test_missing_source_file_yields_zero(self):
        hit = {"similarity": 0.42}
        assert fit_calibrator.label_from_hit(hit, "searcher.py") == 0

    def test_empty_source_file_yields_zero(self):
        hit = {"source_file": "", "similarity": 0.42}
        assert fit_calibrator.label_from_hit(hit, "searcher.py") == 0


class TestCollectPairs:
    def test_drops_bm25_only_hits(self):
        # BM25 / graph-only hits carry similarity=None and must be excluded —
        # the calibrator can't map them.
        hits = [
            {"source_file": "searcher.py", "similarity": 0.8},
            {"source_file": "searcher.py", "similarity": None, "matched_via": "bm25_postgres"},
            {"source_file": "other.py", "similarity": 0.3},
        ]
        pairs = fit_calibrator.collect_pairs(hits, "searcher.py")
        assert pairs == [(0.8, 1), (0.3, 0)]

    def test_empty_hits_yields_empty(self):
        assert fit_calibrator.collect_pairs([], "x.md") == []

    def test_all_hits_relevant(self):
        hits = [
            {"source_file": "x.md", "similarity": 0.1},
            {"source_file": "/some/x.md", "similarity": 0.5},
            {"source_file": "deep/x.md", "similarity": 0.9},
        ]
        pairs = fit_calibrator.collect_pairs(hits, "x.md")
        assert all(r == 1 for _, r in pairs)
        assert [s for s, _ in pairs] == [0.1, 0.5, 0.9]


class TestComputeBeforeAfter:
    def test_perfectly_separable_after_is_better(self):
        # Build a clean signal: low sim → 0, high sim → 1. The raw
        # similarity is *not* zero-on-zero, so brier_before > 0; the
        # isotonic fit collapses it to a step function, so brier_after
        # should be lower (or equal — never worse on the training set).
        from mempalace.calibration import fit_calibrator as fit

        pairs = [(s / 100.0, 1 if s >= 60 else 0) for s in range(0, 100)]
        cal = fit(pairs, source="unit")
        bb, eb, ba, ea = fit_calibrator.compute_before_after(pairs, cal)
        # All four are non-negative.
        assert all(v >= 0.0 for v in (bb, eb, ba, ea))
        # Calibrated should not be worse on the training set.
        assert ba <= bb + 1e-6
        assert ea <= eb + 1e-6

    def test_empty_pairs_raises(self):
        from mempalace.calibration import Calibrator

        with pytest.raises(ValueError):
            fit_calibrator.compute_before_after([], Calibrator())

    def test_returns_four_floats(self):
        from mempalace.calibration import fit_calibrator as fit

        pairs = [(0.2, 0), (0.8, 1)]
        cal = fit(pairs)
        out = fit_calibrator.compute_before_after(pairs, cal)
        assert len(out) == 4
        assert all(isinstance(v, float) for v in out)


class TestRunFit:
    def test_run_fit_with_stub_search_fn(self, tmp_path):
        # Two probes; the stub returns three hits per query, one of which
        # matches the expected source_file.
        probes = [
            {"query": "a", "expected": "a.md"},
            {"query": "b", "expected": "b.md"},
        ]

        def stub_search(query, limit):
            target = f"{query}.md"
            return [
                {"source_file": target, "similarity": 0.9},
                {"source_file": "noise1.md", "similarity": 0.5},
                {"source_file": "noise2.md", "similarity": 0.2},
            ]

        out = tmp_path / "cal.json"
        cal, report = fit_calibrator.run_fit(
            probes=probes,
            search_fn=stub_search,
            limit_per_probe=3,
            label="test-stub",
            out_path=str(out),
        )

        # 2 probes × 3 vector hits = 6 pairs; 2 positives.
        assert report.n_probes == 2
        assert report.n_pairs == 6
        assert report.n_positive == 2
        assert report.label == "test-stub"

        # Calibrator file written, loadable, and identifies its source.
        assert out.exists()
        d = json.loads(out.read_text())
        assert d["source"] == "test-stub"
        assert d["n_samples"] == 6

        # High-similarity hits should calibrate higher than low ones.
        assert cal.apply(0.9) >= cal.apply(0.2)

    def test_run_fit_skips_bm25_only_hits(self):
        # Single probe whose hits are all BM25 (similarity=None) plus one
        # vector hit. Only the vector hit lands in the training set.
        probes = [{"query": "q", "expected": "target.py"}]

        def stub_search(query, limit):
            return [
                {"source_file": "target.py", "similarity": 0.7},
                {"source_file": "target.py", "similarity": None, "matched_via": "bm25_postgres"},
                {"source_file": "noise.md", "similarity": None, "matched_via": "graph_age"},
            ]

        _, report = fit_calibrator.run_fit(
            probes=probes,
            search_fn=stub_search,
            limit_per_probe=10,
            label="bm25-skip-test",
        )
        assert report.n_pairs == 1
        assert report.n_positive == 1

    def test_run_fit_raises_when_no_vector_hits(self):
        # Daemon returning only BM25 hits would yield no calibration data —
        # raise rather than save a degenerate empty calibrator.
        probes = [{"query": "q", "expected": "x.md"}]

        def stub_search(query, limit):
            return [
                {"source_file": "x.md", "similarity": None, "matched_via": "bm25_postgres"},
            ]

        with pytest.raises(RuntimeError, match="no vector hits"):
            fit_calibrator.run_fit(
                probes=probes,
                search_fn=stub_search,
                limit_per_probe=10,
                label="no-vec",
            )

    def test_run_fit_per_probe_summary_shape(self):
        probes = [{"query": "q", "expected": "t.md"}]

        def stub_search(query, limit):
            return [
                {"source_file": "t.md", "similarity": 0.6},
                {"source_file": "other.md", "similarity": 0.3},
            ]

        _, report = fit_calibrator.run_fit(
            probes=probes,
            search_fn=stub_search,
            limit_per_probe=5,
            label="shape-test",
        )
        assert len(report.per_probe_summary) == 1
        row = report.per_probe_summary[0]
        assert row["query"] == "q"
        assert row["expected"] == "t.md"
        assert row["n_hits"] == 2
        assert row["n_vec_hits"] == 2
        assert row["n_positive_vec"] == 1


class TestLoadProbes:
    def test_loads_dict_with_probes_key(self, tmp_path):
        p = tmp_path / "probes.json"
        p.write_text(json.dumps({"probes": [{"query": "q", "expected": "e"}]}))
        out = fit_calibrator._load_probes(str(p))
        assert out == [{"query": "q", "expected": "e"}]

    def test_loads_bare_list(self, tmp_path):
        p = tmp_path / "probes.json"
        p.write_text(json.dumps([{"query": "q", "expected": "e"}]))
        out = fit_calibrator._load_probes(str(p))
        assert out == [{"query": "q", "expected": "e"}]

    def test_rejects_unsupported_shape(self, tmp_path):
        p = tmp_path / "probes.json"
        p.write_text(json.dumps({"not_probes": []}))
        with pytest.raises(ValueError):
            fit_calibrator._load_probes(str(p))


class TestReportAsDict:
    def test_as_dict_shape(self):
        r = fit_calibrator.CalibrationReport(
            n_probes=10,
            n_pairs=100,
            n_positive=5,
            brier_before=0.25,
            brier_after=0.05,
            ece_before=0.30,
            ece_after=0.04,
            label="lbl",
            out_path="/x/y.json",
            breakpoints=12,
            elapsed_secs=42.5,
        )
        d = r.as_dict()
        # Sanity: rounded deltas are computed, positive_rate is correct.
        assert d["brier_delta"] == round(0.05 - 0.25, 4)
        assert d["ece_delta"] == round(0.04 - 0.30, 4)
        assert d["positive_rate"] == round(5 / 100, 4)
        assert d["label"] == "lbl"
        assert d["breakpoints"] == 12
        assert d["elapsed_secs"] == 42.5

    def test_as_dict_zero_pairs_safe(self):
        r = fit_calibrator.CalibrationReport(
            n_probes=0,
            n_pairs=0,
            n_positive=0,
            brier_before=0.0,
            brier_after=0.0,
            ece_before=0.0,
            ece_after=0.0,
            label="empty",
            out_path="",
        )
        assert r.as_dict()["positive_rate"] == 0.0
