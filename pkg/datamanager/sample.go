package datamanager

import (
	"fmt"
	"math/rand/v2"
	"time"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/humanize"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

// SampleQueriesAndPassages samples numQueries and their associated passages, along with topKPassages and random passages
// up to numPassages. It returns the sampled queries, passages, and re-mapped querying tensors.
func (d *DataManager) SampleQueriesAndPassages(backend backends.Backend, numQueries, numPassages, topKPassages int) (
	queries, passages, queriesIsSelected, queriesPassageIDs *tensors.Tensor, err error) {
	// (A) Sample queryIDs
	queryIDs := d.SampleQueries(numQueries, numPassages)

	// Load queries
	queries, err = d.LoadQueriesByIDs(backend, queryIDs...)
	if err != nil {
		return
	}

	// Load queriesIsSelected
	queriesIsSelected, err = d.LoadIsSelectedForQueries(backend, queryIDs...)
	if err != nil {
		return
	}

	// passageIDsSet to hold all our unique passage IDs
	passageIDsSet := sets.Make[int32](numPassages)

	// Add passages directly associated with the sampled queries
	associatedPassageIDsLocal, originalQueriesPassageIDsLocal, err := d.LoadPassagesForQueries(backend, queryIDs...)
	if err != nil {
		return
	}
	for _, passageID := range associatedPassageIDsLocal {
		passageIDsSet.Insert(int32(passageID))
	}

	// (B) Add topK passages associated with the sampled queries
	var topKPassageIDs *tensors.Tensor
	topKPassageIDs, _, err = d.TopKPassagesForQueries(backend, queries, topKPassages, 0, false)
	if err != nil {
		originalQueriesPassageIDsLocal.FinalizeAll()
		return
	}
	err = tensors.ConstFlatData(topKPassageIDs, func(flatTopKPassageIDs []int32) {
		for _, passageID := range flatTopKPassageIDs {
			if passageID >= 0 {
				passageIDsSet.Insert(int32(passageID))
			}
		}
	})
	topKPassageIDs.FinalizeAll() // free up memory since we don't need the tensor anymore
	if err != nil {
		originalQueriesPassageIDsLocal.FinalizeAll()
		return
	}

	// (C) Sample unique random passages until the total number of sampled messages reach numPassages
	for len(passageIDsSet) < numPassages {
		passageID := rand.IntN(d.NumPassages)
		passageIDsSet.Insert(int32(passageID))
	}

	// Now proceed to load them
	finalPassageIDs := xslices.SortedKeys(passageIDsSet)
	finalPassageIDsInt := make([]int, len(finalPassageIDs))
	for i, v := range finalPassageIDs {
		finalPassageIDsInt[i] = int(v)
	}
	passages, err = d.LoadPassagesByIDs(backend, finalPassageIDsInt...)
	if err != nil {
		originalQueriesPassageIDsLocal.FinalizeAll()
		return
	}

	// Create reverse mapping
	globalToSampledIdx := make(map[int32]int32, len(finalPassageIDs))
	for idx, id := range finalPassageIDs {
		globalToSampledIdx[id] = int32(idx)
	}

	flatQueriesPassageIDs := make([]int32, numQueries*PassagesPerQuery)
	err = tensors.ConstFlatData(originalQueriesPassageIDsLocal, func(flatOriginalQueriesPassageIDs []int32) {
		for i, localPassageIdx := range flatOriginalQueriesPassageIDs {
			if localPassageIdx >= 0 {
				globalPassageID := associatedPassageIDsLocal[localPassageIdx]
				flatQueriesPassageIDs[i] = globalToSampledIdx[int32(globalPassageID)]
			} else {
				flatQueriesPassageIDs[i] = -1
			}
		}
	})
	originalQueriesPassageIDsLocal.FinalizeAll()
	if err != nil {
		return
	}

	queriesPassageIDs = tensors.FromFlatDataAndDimensions(flatQueriesPassageIDs, numQueries, PassagesPerQuery)

	return
}

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
//   - backend: where to run the computations.
//   - queries: tensor of shape [len(queryIDs), embedding_dim] containing the query embeddings of
//     the queries to score.
//   - k: the number of top passages to return for each query.
//   - passagesBatchSize: number of passages to process at a time. If 0, it will be set to 128, a good default.
//   - showProgress: true if the progress (along an ETA) should be printed to the console.
//
// It returns:
//
//   - topKPassageIDs: tensor of shape [len(queryIDs), k]Int32 containing the indices of the top k passages for each query.
//     It is initialized (padded) with -1, in case there are less than k passages.
//   - topKScores: tensor of shape [len(queryIDs), k]Float32 containing the scores (cosine-similarity) of the top k passages for each query.
//     It is initialized (padded) with -inf, in case there are less than k passages.
func (d *DataManager) TopKPassagesForQueries(
	backend backends.Backend, queries *tensors.Tensor, k int,
	passagesBatchSize int, showProgress bool) (
	topKPassageIDs, topKScores *tensors.Tensor, err error) {
	dtypeFloat := queries.DType()
	dtypeInt := dtypes.Int32

	if passagesBatchSize == 0 {
		passagesBatchSize = 128
	}
	updateTopKExec, err := NewExec(backend, func(queries, passagesBatch, passagesBatchBaseID, currentTopKPassageIDs, currentTopKScores *Node) (*Node, *Node) {
		g := queries.Graph()
		numQueries := queries.Shape().Dim(0)
		numPassagesBatch := passagesBatch.Shape().Dim(0)

		// 1. Scores for the new batch: cosine similarity
		newScores := CrossCosineSimilarity(queries, passagesBatch, -1, 0)
		currentTopKScores = Concatenate([]*Node{currentTopKScores, newScores}, -1)

		// 2. Indices for the new batch
		newIndices := Add(Iota(g, shapes.Make(dtypeInt, numQueries, numPassagesBatch), 1), passagesBatchBaseID)
		currentTopKPassageIDs = Concatenate([]*Node{currentTopKPassageIDs, newIndices}, -1)

		// 3. Sort currentTopK... by their descending scores (similarity)
		compareScores := NewClosure(g, func(g *Graph) []*Node {
			lhsScore := Parameter(g, "lhsScore", shapes.Make(dtypeFloat))
			rhsScore := Parameter(g, "rhsScore", shapes.Make(dtypeFloat))
			_ = Parameter(g, "lhsIndex", shapes.Make(dtypeInt))
			_ = Parameter(g, "rhsIndex", shapes.Make(dtypeInt))
			return []*Node{GreaterThan(lhsScore, rhsScore)}
		})
		sorted := SortFunc(compareScores, 1, false, currentTopKScores, currentTopKPassageIDs)
		currentTopKScores = sorted[0]
		currentTopKPassageIDs = sorted[1]

		// 4. Truncate to take only the TopK
		if currentTopKScores.Shape().Dim(1) > k {
			currentTopKScores = Slice(currentTopKScores, AxisRange(), AxisRangeFromStart(k))
			currentTopKPassageIDs = Slice(currentTopKPassageIDs, AxisRange(), AxisRangeFromStart(k))
		}
		return currentTopKPassageIDs, currentTopKScores
	})
	if err != nil {
		return nil, nil, err
	}
	defer updateTopKExec.Finalize()

	// Initialize topKPassageIDs and topKScores with -1 and -inf respectively.
	numQueries := queries.Shape().Dim(0)
	numPassages := d.NumPassages
	res, err := ExecOnceN(backend, func(g *Graph) (*Node, *Node) {
		ids := BroadcastToDims(Scalar(g, dtypeInt, -1), numQueries, k)
		scores := BroadcastToDims(Infinity(g, dtypeFloat, -1), numQueries, k)
		return ids, scores
	})
	if err != nil {
		return nil, nil, err
	}
	topKPassageIDs = res[0]
	topKScores = res[1]

	// Loop variables:
	start := time.Now()
	lastUpdate := start
	var passagesBaseIdx int

	// Print progress status function.
	printStatusFn := func() {
		prefix, eta := " -", "Unknown"
		elapsed := time.Since(start)
		remaining := numPassages - passagesBaseIdx
		if remaining <= 0 {
			// Last update, we append a new-line.
			eta = fmt.Sprintf("Elapsed time: %s\n", humanize.Duration(time.Since(start)))
			prefix = "✅"
		} else if passagesBaseIdx > 0 && elapsed > 0 {
			speed := float64(passagesBaseIdx) / elapsed.Seconds()
			eta = "ETA: " + humanize.Duration(time.Duration(float64(remaining)/speed)*time.Second)
		}
		fmt.Printf("\r%s Scoring %s queries x %s / %s passages (%.1f%%) -- %s %s",
			prefix, humanize.Count(numQueries), humanize.Count(passagesBaseIdx), humanize.Count(numPassages),
			float64(passagesBaseIdx)/float64(numPassages)*100, eta, humanize.EraseToEndOfLine)
		lastUpdate = time.Now()
	}

	// Loop over passages, a batch at a time.
	for passagesBaseIdx = 0; passagesBaseIdx < numPassages; passagesBaseIdx += passagesBatchSize {
		if showProgress && time.Since(lastUpdate) > 1*time.Second {
			printStatusFn()
			lastUpdate = time.Now()
		}
		thisBatchSize := min(passagesBatchSize, numPassages-passagesBaseIdx)
		passagesBatch, err := d.LoadPassagesBatch(backend, passagesBaseIdx, thisBatchSize)
		if err != nil {
			return nil, nil, err
		}
		donatedIDs, err := DonateTensorBuffer(topKPassageIDs, backend, 0)
		if err != nil {
			return nil, nil, err
		}
		donatedScores, err := DonateTensorBuffer(topKScores, backend, 0)
		if err != nil {
			return nil, nil, err
		}
		res, err = updateTopKExec.Exec(queries, passagesBatch, int32(passagesBaseIdx), donatedIDs, donatedScores)
		if err != nil {
			return nil, nil, err
		}
		topKPassageIDs = res[0]
		topKScores = res[1]
		passagesBatch.FinalizeAll()
	}
	if showProgress {
		printStatusFn()
	}

	return topKPassageIDs, topKScores, nil
}
