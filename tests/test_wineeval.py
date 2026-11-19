from pathlib import Path

import pytest

import wineeval as w

rows = w.load_corpus(Path(__file__).parent / "fixture.csv")


def test_labels_match_all_filters_case_insensitively():
    assert w.relevant(rows, {"region": "argentina"}) == {0, 1}
    assert w.relevant(rows, {"region": "Argentina", "name": "Catena"}) == {0}
    assert w.relevant(rows, {"variety": "Sparkling"}) == {3}
    # labels come from metadata fields, not notes: row 4 mentions Argentina only in its notes
    assert 4 not in w.relevant(rows, {"region": "Argentina"})


def test_query_kind():
    assert w.query_kind("Malbec from Mendoza", {"region": "Mendoza"}) == "literal"
    assert w.query_kind("Argentinian red", {"region": "Argentina", "variety": "Red Wine"}) == "paraphrase"
    assert w.query_kind("bubbly", {"variety": "Sparkling & Champagne"}) == "paraphrase"


def test_recall_is_capped_by_relevant_set_size():
    assert w.recall_at_k([0, 9, 1], {0, 1}, 5) == 1.0
    assert w.recall_at_k([0, 9, 1], {0, 1}, 2) == 0.5
    assert w.recall_at_k(list(range(10)), set(range(100)), 5) == 1.0
    assert w.recall_at_k([7, 8], {0}, 2) == 0.0


def test_reciprocal_rank():
    assert w.reciprocal_rank([5, 3, 1], {1, 3}) == 0.5
    assert w.reciprocal_rank([5, 3, 1], {9}) == 0.0


def test_rrf():
    # b is 1st and 2nd: beats a (1st and 3rd) and c (only in one list)
    assert w.rrf([["a", "b", "c"], ["b", "d", "a"]], k=60) == ["b", "a", "d", "c"]
    assert w.rrf([[2, 1], [1, 2]]) == [1, 2]  # exact tie broken by id


def test_bm25_ranks_the_matching_doc_first():
    bm = w.BM25([w.doc_text(r) for r in rows])
    assert w.rank(bm.scores("Maipo Valley"))[0] == 2
    assert w.rank(bm.scores("brioche"))[0] == 3
    assert not bm.scores("zzz").any()


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = w.bootstrap_ci([0, 1] * 20)
    assert lo < 0.5 < hi
    assert w.bootstrap_ci([0.3] * 10) == pytest.approx((0.3, 0.3))


def test_every_query_in_the_real_set_has_labels():
    import json
    qs = json.loads((w.ROOT / "queries.json").read_text())
    corpus = w.load_corpus()
    assert 40 <= len(qs) <= 60
    assert all(w.relevant(corpus, q["filters"]) for q in qs)


def test_bm25_reproduces_the_published_row():
    # BM25 is deterministic and needs no model, so the README/results.md BM25 numbers can be checked in CI
    import json
    corpus = w.load_corpus()
    qs = json.loads((w.ROOT / "queries.json").read_text())
    bm = w.BM25([w.doc_text(r) for r in corpus])
    runs = [w.rank(bm.scores(q["q"])) for q in qs]
    rels = [w.relevant(corpus, q["filters"]) for q in qs]
    got = {m: round(sum(f(r, rel) for r, rel in zip(runs, rels)) / len(qs), 2) for m, f in w.METRICS.items()}
    assert got == {"recall@5": 0.66, "recall@10": 0.66, "MRR": 0.71}
    assert "| BM25 | 0.66 (0.55 to 0.77) | 0.66 (0.55 to 0.77) | 0.71 (0.59 to 0.82) |" in (w.ROOT / "results.md").read_text()


def _run_main_in(tmp_path, monkeypatch, dense):
    import shutil
    shutil.copy(w.ROOT / "queries.json", tmp_path / "queries.json")
    monkeypatch.setattr(w, "ROOT", tmp_path)  # load_corpus keeps its default path to the real CSV
    monkeypatch.setattr(w, "dense_rankings", dense)
    w.main()
    return tmp_path / "results.md"


def test_bm25_only_run_does_not_overwrite_the_published_results(tmp_path, monkeypatch, capsys):
    # without the dense extra the run has no dense or hybrid rows; writing it would
    # replace the published three-retriever tables with a BM25-only file
    out = _run_main_in(tmp_path, monkeypatch, lambda docs, queries: None)
    assert not out.exists()
    printed = capsys.readouterr()
    assert "| BM25 | 0.66 (0.55 to 0.77)" in printed.out
    assert "results.md not written" in printed.err


def test_full_run_writes_results(tmp_path, monkeypatch):
    # a stand-in dense ranker (doc id order) is enough to exercise the write path
    out = _run_main_in(tmp_path, monkeypatch, lambda docs, queries: [list(range(len(docs)))] * len(queries))
    text = out.read_text()
    assert "| hybrid (RRF, k=60) minus BM25 |" in text
    assert text.count("| BM25 |") == 3  # one table per stratum


def test_readme_states_the_real_number_of_distinct_label_sets():
    # queries that share a relevant set are not independent, which the README's Limitations quantify
    import json
    from collections import Counter
    corpus = w.load_corpus()
    qs = json.loads((w.ROOT / "queries.json").read_text())
    sets = Counter(frozenset(w.relevant(corpus, q["filters"])) for q in qs)
    twice = sum(1 for c in sets.values() if c == 2)
    assert max(sets.values()) == 2
    readme = " ".join((w.ROOT / "README.md").read_text().split())
    assert f"{len(qs)} queries over {len(sets)} distinct label sets ({twice} sets are queried twice" in readme
