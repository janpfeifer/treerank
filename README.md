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
