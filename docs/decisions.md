# Decisions

Five choices that shape what the numbers in the README mean. The commits
record what was decided, not what else was weighed. The alternatives below
were written on 2026-11-07 as the obvious options and why the chosen one fits;
"not recorded" marks options nobody wrote about at the time.

## 1. Labels come from metadata filters, not from reading the tasting notes

- **Context.** A retrieval eval needs a relevant set per query. The CSV has
  structured fields (name, region, variety) and free-text notes.
- **Decision.** Each query carries a filter over the metadata; a row is
  relevant when every filter value is a case-insensitive substring of its
  field (`relevant()` in `wineeval.py`). Notes are never used for labels.
- **Alternatives.** Hand-judging top-k results per retriever: worse, because
  labels would then depend on which retriever surfaced a row, and judging
  1,365 rows by hand per query is not feasible. An LLM judge: not recorded.
- **Consequences.** Labels are reproducible and retriever-independent, but
  only as good as the metadata (a Shiraz not named as one is "not relevant"),
  and literal queries are close to string matching, which favors BM25. Both
  are stated in the README.
- **Evidence.** 57530c9; `test_labels_match_all_filters_case_insensitively`.

## 2. Query kind is computed, not hand-assigned

- **Context.** The headline result depends on whether a query names its label.
- **Decision.** `query_kind()` marks a query *literal* if it shares a token
  with a filter value (ignoring "wine", "red", "white"), else *paraphrase*.
- **Alternatives.** Tagging each query by hand: not recorded as considered;
  a rule cannot drift toward the result I expected.
- **Consequences.** The split is reproducible, but the rule was written with
  the queries, so the strata are still a post-hoc lens (README, Limitations).
- **Evidence.** 57530c9; `test_query_kind`.

## 3. recall@k is capped at min(k, |relevant|), and named "capped recall"

- **Context.** Relevant sets range from 3 to 91 rows. Plain recall@5 on a
  91-row label can never exceed 5/91.
- **Decision.** Divide hits in the top k by min(k, |relevant|), so 1.0 is
  reachable for every query and large labels do not drag the mean down.
- **Alternatives.** Uncapped recall (fails for the reason above); precision@k
  or nDCG: not recorded.
- **Consequences.** Scores are comparable across label sizes, but they are not
  the textbook recall, so they are not comparable with numbers elsewhere. For a
  label with at least k rows the value is precision@k (51 of 55 queries at
  k=5), so the columns are labelled "capped recall@k" and defined under the
  table.
- **Evidence.** 57530c9; `test_recall_is_capped_by_relevant_set_size`.

## 4. Compare retrievers with paired bootstrap differences

- **Context.** Per-retriever intervals overlap heavily at n=55, and the
  retrievers are scored on the same queries.
- **Decision.** Report "X minus BM25" rows: bootstrap the per-query difference
  (2,000 resamples, seed 0), which removes query difficulty shared by both.
- **Alternatives.** Eyeballing overlapping marginal CIs (too conservative);
  a paired t-test or Wilcoxon: not recorded.
- **Consequences.** Tighter, honest intervals for the comparison that matters.
  Queries that reuse a label set are not independent, so the intervals are
  somewhat too narrow; no multiple-comparison correction is applied.
- **Evidence.** 57530c9; `main()` in `wineeval.py`.

## 5. Pin the deterministic part of the published results in a test

- **Context.** `results.md` is generated, and a README number can drift from
  the code that produced it.
- **Decision.** A test recomputes the BM25 row and compares it with the
  published values. Dense and hybrid rows are not pinned, because dense scores
  can change in the last decimal across torch versions and hardware, and CI
  does not install the 90 MB model.
- **Alternatives.** Pin every row (flaky across machines); pin nothing (README
  could drift silently).
- **Consequences.** A BM25 or label change that moves the headline fails CI.
  Since then `scores.json` holds the per-query scores of all three retrievers
  and a test rebuilds `results.md` from it, so every published mean and
  interval is checked against committed scores, and the BM25 scores are
  recomputed. CI still does not run the dense model: whether `scores.json`
  itself reproduces is checked by running `python wineeval.py` and then
  `git diff --exit-code results.md scores.json`.
- **Evidence.** e9c971c (PR #5).
