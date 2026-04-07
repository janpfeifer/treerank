# treerank
Experiments using trees to rank / retrieve content.



## Data Format

The command-line tool `cmd/embed` can be used to generated the dataset used for this experiments.

It reads from MS MARCO dataset in HuggingFace and extracts the _queries_ and the _passages_ + _is_selected_ (whether 
the passage was selected)
into 4 separate sharded files per split of the dataset. Each split ("train", "validation", "test") has its own 
subdirectory, and within each the following 4 files:

1. `queries.bin`: The embedded queries sorted by ids. Each query occupies exactly **3840 float32** stored in 
little-endian, so **15360 bytes** -- this way they allow for random access, given an id.
2. `passages.bin`: The embedded passages sorted by ids. Each passage occupies 
exactly **3840 float32**, so **15360 bytes**.
3. `query_indices.bin`: The indices of the passages for each query. Each query has a variable number of passages,
but at most 10. 
   So this file is a sequence of int32 values (4 bytes each) representing the 10 indices of the passages for each
   query, so 40 bytes per query. 
   It has the same number of records (queries) as `queries.bin`. 
   An index -1 is a placeholder for a missing passage.
4. `query_is_selected.bin`: The is_selected flags for each passage of a query. 
   There are 10 is_selected values of type bool (one byte each) for each row, and padding is filled with 0 as well.
   It has the same number of records as `queries.bin`.

In the process it deduplicates passages (based on a hash of the passage text), creates indices to the
_queries_ and _passages_ and only keeps the first occurrence of each passage, and share the indices when a
passage occurs for more than one query. 

## Performance Metrics

### MRR (Mean-Reciprocal-Rank) formulation

- All passages of all examples in a dataset split are considered as negatives
  - But only for the split, so when evaluating on "validation", none of the passages from "train" or "test" are considered.
- MRR uses the best ranked *is_selected* passage. So if for a query there is more than one selected passage,
  the RR for the query is 1/n, where n is the max(rank for all selected passages).
- The RR is then thresholded at 1/10, so if the best ranked *is_selected* passage is at a rank > 10, it is scored as 0.
- Queries with no *is_selected* passage don't participate on the mean of MRR.

### KaLM-Gemma3 12B (BFloat16), on NVidia RTX 5090

- Ranked by pure cosine-similarity of the query and passage.

#### Newer version:

- Task: "MSMARCO"
- Bucketing strategy: Two-bit bucket budget: 8192 tokens.
- Padding-ratio: ~20%
- Total time: ???
- MRR: ~0.48

#### Old version

Task not set (should be "MSMARCO"), which yields much worse retrieval.

- Bucketing strategy: Power-of-1.4, batch budget: 8192 tokens.
- Padding-ratio: ~20%
- Total time: 2h32m
- MRR: ~0.30

```
┌────────────────┬────────┬───────────────┐
│ Metric         │ Total  │     Speed     │
├────────────────┼────────┼───────────────┤
│ Queries        │ 101.1K │   11  items/s │
│ Passages       │ 980.5K │  107  items/s │
│ Tokens         │  86.5M │ 9.5K tokens/s │
│ Non-Pad Tokens │  72.2M │ 7.9K tokens/s │
└────────────────┴────────┴───────────────┘
```

