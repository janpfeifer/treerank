package datamanager

import (
	"math/rand/v2"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

// SampleQueries samples numQueries queries without replacement (all queryIDs are unique) from the dataset.
//
// It returns:
//
//   - queryIDs: global IDs (indices in the set of all queries in the dataset) of the sampled queries.
//     They are unique and sorted in ascending order.
func (d *DataManager) SampleQueries(numQueries, numPassages int) (queryIDs []int) {
	queryIDSet := sets.Make[int](numQueries)
	for len(queryIDSet) < numQueries {
		queryID := rand.IntN(d.NumQueries)
		queryIDSet.Insert(queryID)
	}
	queryIDs = xslices.SortedKeys(queryIDSet)
	return queryIDs
}

// TopKPassagesForQueries returns the top k passages for each query in queryIDs.
//
//   - queries: tensor of shape [len(queryIDs), embedding_dim] containing the query embeddings of
//     the queries to score.
//   - passages: tensor of shape [numPassages, embedding_dim] containing the all the passage embeddings
//     considered.
//
// It returns:
//
//   - topKPassageIDs: tensor of shape [len(queryIDs), k]Int32 containing the indices of the top k passages for each query.
//     It is initialized (padded) with -1, in case there are less than k passages.
//   - topKScores: tensor of shape [len(queryIDs), k]Float32 containing the scores (cosine-similarity) of the top k passages for each query.
//     It is initialized (padded) with -inf, in case there are less than k passages.
func (d *DataManager) TopKPassagesForQueries(backend backends.Backend, queries *tensors.Tensor, passages *tensors.Tensor, k int) (topKPassageIDs, topKScores *tensors.Tensor) {
	return
}
