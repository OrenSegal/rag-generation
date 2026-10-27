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
