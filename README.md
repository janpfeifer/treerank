# TreeRank

TreeRank is a small research project to train an alternative type of deep retrieval model.

## Overview

It aims to discretize the encoding of queries and passages to a series of classifications using bagging/boosting:

- Each model pair (one for queries, one for documents) classifies a query/document to one of C classes -- number of 
  classes is a hyperparameter, let's say 1000.
- There will be M models (number of models again is configurable, let's say 2000), trained in mix of "boosting" 
  and "bagging" (training on different random subsets of the data and different random initializations).
- The final retrieval is based on the number of matching classes of document to query (Hamming distance).
- The documents model includes a separate binary classification head to predict whether a document is relevant to a
  query. This is used to filter out irrelevant documents before the main classification. There is threshold
  for which documents are considered irrelevant, and assumed not to match any query class.
- The document model is bootstrapped from the query model, but otherwise is optimized differently and can drift.
  This is deliberately, and the idea is that the document model will learn to project the types of results from
  the queries.

The "Tree" prefix in the name project was inspired by the fact that we wanted to do the models
as a trees (and the ensemble as forests). However, for simplicity, the first version will be purely 
neural network based (using FNNs)

## Dataset: MS Marco v2.1

For development and benchmarking we will use the MS Marco dataset, version 2.1. 
While the leaderboard is closed, the dataset is still relevant to this example.
It is downloaded from [1].

For the dataset, the documents are called "passages", and refer to passages of text in websites
that may answer the queries.

Each query comes with 10 passages, with zero to 10 of them being "selected" -- in most cases there is only
one selected passage (but there are up to 7 selected). The non-selected are considered "hard-negatives".

The dataset is split into _train_ (~800K queries and 8M passages), _validation_ (~100K queries and 1M passages).
The test split doesn't have any labels, hence cannot be used.

All training and validation will be done using the _train_ split, and we use the _validation_ split for testing.

Links:

- [1] https://huggingface.co/datasets/microsoft/ms_marco

## Embedding text: KaLM-Gemma3 12B model, embedding to 3840 dimensions.

TreeRank assumes there is a pre-trained embedding model that can be used to embed queries and passages.

For development, we are using KaLM-Gemma3 12B model [2], which is one of the top ones on HuggingFace leaderboard.

Links:

- [1] https://kalm-embedding.github.io/
- [2] https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511

## Metrics

The metric for retrieval used with MS Marco is MRR (Mean Reciprocal Rank) with top-10 recall, measured with all
the passages of the validation dataset (~100K queries and ~1M passages, so 1M negatives for each query).

Our initial baseline is simply using the cosine distance of the embeddings from KaLM-Gemma3 12B model. 
With 1M negatives, the KaLM-Gemma3-12B only retrieval scores an MRR of **0.3362**.

A true baseline would be to train a thin layer—like a 2-layer MLP or even a simple linear projection -- 
with a contrastive loss (like InfoNCE) on top of the KaLM-Gemma3 embeddings, to factor out that fact
that TreeRank is actually training (and the pure KaLM-Gemma3 is not).

We don't want to fine-tune the full Gemma3 model, becuase that is so much more expensive that we consider that this
is another class of solutions entirely.

## Training

The training methodology for each (of the M) model would be:

1. Find a random even split of the queries: add a "uniformity loss" on each class based on the
   square number of queries classified
   to that class, and train the queries model first. This could take the form as an entropy loss
   $-\sum_{p_i}{p_i log(p_i)}$ or as a GINI loss $\sum_{p_i}{p_i^2}$.

2. Bootstrap the "documents" model from the queries model, but then train it such that the "selected" documents
   ("passages" for MSMARCO) fall in the same class and the queries.

3. A final round of training both models (query and document) simultaneously, or alternatingly.

Each model pair would be trained with a subset of the queries and passages (including hard negatives),
it would be highly regularized (L2, dropout), and likely it will simply use gradient decent, there won't
be any batching (except that dropout may work somewhat like batching).
Controlling the number of steps in (B) and (C) may work as a regularization as well. I'm hoping this would make them
"weak"  and independent enough for the boosting/bagging process to work.

The weight of each query/passage example will be $(1+e)/(1+n)$, where $n$ is the number of models of the ensemble
trained so far and $e$ is the number of those where the query/passage didn't get mapped to the same class.
The linear decay on the weight should prevent false-positives in the label to lead to explosive weights (if
we were using some exponential weighting scheme).

The binary relevance ("no-retrieval") model: this will be also an ensemble. So a document can be ignore by one model 
of the ensemble but not by the others. 
Effectively, a document could be indexed as an extra class, or simply not indexed for that particular model, if it
is deemed un-retrievable (non-relevant). But it could still be considered retrievable by other models, so
it is not definitive. 
And it works for new documents being indexed, without the need to retrain.


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
- MRR: ~0.4878

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
