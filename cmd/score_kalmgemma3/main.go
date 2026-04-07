package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/humanize"
	"github.com/janpfeifer/treerank/internal/datamanager"
	"k8s.io/klog/v2"
)

var (
	flagData = flag.String("data", "", "Data directory where files should be generated.")
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	if *flagData == "" {
		klog.Exit("-data <data directory> must be specified")
	}

	dm := must1(datamanager.New(*flagData))
	defer dm.Close()
	backend := must1(backends.New())
	defer backend.Finalize()

	fmt.Printf("Dataset in %s:\n", *flagData)
	fmt.Printf("  - Queries:  %d\n", dm.NumQueries)
	fmt.Printf("  - Passages: %d\n", dm.NumPassages)

	// Investigate the distribution of selected passages per query.
	queryIsSelected := must1(dm.LoadQueryIsSelected(backend))
	perCount := must1(ExecOnce(backend, func(queryIsSelected *Node) *Node {
		g := queryIsSelected.Graph()
		numSelected := GreaterThan(queryIsSelected, ScalarZero(g, queryIsSelected.DType()))
		countSelected := ReduceSum(ConvertDType(numSelected, dtypes.Int8), 1)

		perCount := IotaFull(g, shapes.Make(dtypes.Int8, 1, 10))      // Shape: [1, 10]
		perCount = Equal(ExpandAxes(countSelected, -1), perCount)     // Shape: [NumQueries, 10]Bool
		perCount = ReduceSum(ConvertDType(perCount, dtypes.Int32), 0) // Shape: [10]Int32
		return perCount
	}, queryIsSelected))
	fmt.Printf("  - Queries: count of #selected: %v\n", perCount.Value().([]int32))

	// Update MRR computation by selecting the topK passages for each query.
	const K = 10
	updateMRRExec := must1(NewExec(backend, func(queries, passagesBatch, passagesBaseIdx, currentTopKPassagesPerQueryIndices, currentTopKPassagesPerQueryScores *Node) (*Node, *Node) {
		// fmt.Printf("Shapes: %s, %s, %s, %s, %s\n", queries.Shape(), passagesBatch.Shape(), passagesBaseIdx.Shape(), currentTopKPassagesPerQueryIndices.Shape(), currentTopKPassagesPerQueryScores.Shape())
		g := queries.Graph()
		dtype := queries.DType()
		numQueries := queries.Shape().Dim(0)
		numPassagesBatch := passagesBatch.Shape().Dim(0)

		// 1. Scores for the new batch: cosine similarity, shape [NumQueries, NumPassagesBatch]
		newScores := CrossCosineSimilarity(queries, passagesBatch, -1, 0)
		currentTopKPassagesPerQueryScores = Concatenate([]*Node{currentTopKPassagesPerQueryScores, newScores}, -1) // [NumQueries, K + NumPassagesBatch]

		// 2. Indices for the new batch:
		newIndices := Add(Iota(g, shapes.Make(dtypes.Int32, numQueries, numPassagesBatch), 1), passagesBaseIdx)
		currentTopKPassagesPerQueryIndices = Concatenate([]*Node{currentTopKPassagesPerQueryIndices, newIndices}, -1) // [NumQueries, K + NumPassagesBatch]

		// 4. Sort currentTopK... by their descending scores (similarity)
		compareScores := NewClosure(g, func(g *Graph) []*Node {
			lhsScore := Parameter(g, "lhsScore", shapes.Make(dtype))
			rhsScore := Parameter(g, "rhsScore", shapes.Make(dtype))
			_ = Parameter(g, "lhsIndex", shapes.Make(dtypes.Int32))
			_ = Parameter(g, "rhsIndex", shapes.Make(dtypes.Int32))
			return []*Node{GreaterThan(lhsScore, rhsScore)}
		})
		sorted := SortFunc(compareScores, 1, false, currentTopKPassagesPerQueryScores, currentTopKPassagesPerQueryIndices)
		currentTopKPassagesPerQueryScores = sorted[0]
		currentTopKPassagesPerQueryIndices = sorted[1]

		// 5. Truncate to take only the TopK
		if currentTopKPassagesPerQueryScores.Shape().Dim(1) > K {
			currentTopKPassagesPerQueryScores = Slice(currentTopKPassagesPerQueryScores, AxisRange(), AxisRangeFromStart(K))
			currentTopKPassagesPerQueryIndices = Slice(currentTopKPassagesPerQueryIndices, AxisRange(), AxisRangeFromStart(K))
		}

		return currentTopKPassagesPerQueryIndices, currentTopKPassagesPerQueryScores
	}))

	// Calculate the MRR of the queries.
	queries := must1(dm.LoadQueries(backend))
	const passagesBatchSize = 32

	// currentTopKPassagesPerQuery starts with 0 passages selected, and keeps being updated with the topK for each batch.
	currentTopKPassagesPerQueryIndices := must1(tensors.FromShapeForBackend(backend, 0, shapes.Make(dtypes.Int32, dm.NumQueries, 0)))
	currentTopKPassagesPerQueryScores := must1(tensors.FromShapeForBackend(backend, 0, shapes.Make(dtypes.Float32, dm.NumQueries, 0)))
	start := time.Now()
	lastUpdate := start
	var passagesBaseIdx int
	printStatusFn := func() {
		eta := "Unknown"
		elapsed := time.Since(start)
		remaining := dm.NumPassages - passagesBaseIdx
		if passagesBaseIdx > 0 && elapsed > 0 {
			speed := float64(passagesBaseIdx) / elapsed.Seconds()
			eta = humanize.Duration(time.Duration(float64(remaining)/speed) * time.Second)
		}
		if remaining == 0 {
			eta = ""
		} else {
			eta = " -- ETA: " + eta
		}
		fmt.Printf("\r- Processed %s / %s passages (%.1f%%)%s %s",
			humanize.Count(passagesBaseIdx), humanize.Count(dm.NumPassages),
			float64(passagesBaseIdx)/float64(dm.NumPassages)*100, eta, humanize.EraseToEndOfLine)
		lastUpdate = time.Now()
	}

	// Loop at a batch of passages at a time.
	for passagesBaseIdx = 0; passagesBaseIdx < dm.NumPassages; passagesBaseIdx += passagesBatchSize {
		if time.Since(lastUpdate) > 1*time.Second {
			printStatusFn()
			lastUpdate = time.Now()
		}
		thisBatchSize := min(passagesBatchSize, dm.NumPassages-passagesBaseIdx)
		passagesBatch := must1(dm.LoadPassagesBatch(backend, passagesBaseIdx, thisBatchSize))
		currentTopKPassagesPerQueryIndices, currentTopKPassagesPerQueryScores = must2(updateMRRExec.Exec2(
			queries, passagesBatch, int32(passagesBaseIdx),
			must1(DonateTensorBuffer(currentTopKPassagesPerQueryIndices, backend, 0)),
			must1(DonateTensorBuffer(currentTopKPassagesPerQueryScores, backend, 0))))
	}
	printStatusFn()
	fmt.Println()
	fmt.Printf("- Elapsed time: %s\n", humanize.Duration(time.Since(start)))
	updateMRRExec.Finalize()

	// Now that we have the TopK passages for each query, we can compute the MRR, by checking if any of them
	// are the selected passages for the query.
	queryIndices := must1(dm.LoadQueryIndices(backend))
	mrr := must1(ExecOnce(backend, func(queryIndices, queryIsSelected, currentTopKPassagesPerQueryIndices *Node) *Node {
		g := queryIndices.Graph()
		dtypeInt := dtypes.Int32
		dtypeFloat := dtypes.Float32
		zeroInt := ScalarZero(g, dtypeInt)
		zeroFloat := ScalarZero(g, dtypeFloat)
		oneFloat := ScalarOne(g, dtypeFloat)
		numQueries := queryIndices.Shape().Dim(0)

		// 2. isSelected: shape [NumQueries, 10]Bool
		isSelected := GreaterThan(queryIsSelected, ScalarZero(g, queryIsSelected.DType()))
		countSelected := ReduceSum(ConvertDType(isSelected, dtypes.Int32), -1) // Shape [NumQueries]

		// 3. Query indices: keep only the selected ones, replace the others with -1.
		selectedIndices := Where(isSelected, queryIndices, Scalar(g, dtypeInt, -1))

		// 4. Check if any of the topK passages are selected.
		expandedTopK := ExpandAxes(currentTopKPassagesPerQueryIndices, -1) // [NumQueries, K, 1]
		expandedSelectedIndices := ExpandAxes(selectedIndices, 1)          // [NumQueries, 1, 10]
		matchedTopK := Equal(expandedSelectedIndices, expandedTopK)        // [NumQueries, K, 10]
		matchedTopK = ReduceLogicalOr(matchedTopK, -1)                     // [NumQueries, K]  // If a topK matches any of the selections it is good!

		// 5. RR (reciprocal rank, 1/(rank+1)) score of each passage -- later we select only those selected.
		rrScores := Reciprocal(OnePlus(Iota(g, shapes.Make(dtypeFloat, numQueries, K), 1))) // [NumQueries, K], values 1/i, 1 <= i <= K
		rrScores = Where(matchedTopK, rrScores, zeroFloat)                                  // Set to zero for values that didn't match.

		// 6. Take only the RR of the best (maximum) selected of the possible correct passages.
		rrScores = ReduceMax(rrScores, -1)

		// 7. Calculate the MRR by taking the mean over only the queries with non-zero selections.
		//    (This may differ from the leaderboard, I'm not sure if they include tests with no selections and then count these as 0)
		validQueries := ReduceSum(ConvertDType(GreaterThan(countSelected, zeroInt), dtypeFloat)) // Scalar denominator for the mean.
		validQueries = Max(validQueries, oneFloat)                                               // Avoid division by zero.
		mrr := Div(ReduceSum(rrScores), validQueries)
		return mrr
	}, queryIndices, queryIsSelected, currentTopKPassagesPerQueryIndices))
	fmt.Printf("MRR: %s\n", mrr)
}

func must(err error) {
	if err != nil {
		klog.Errorf("Must failed: %+v", err)
		panic(err)
	}
}

func must1[T any](v T, err error) T {
	must(err)
	return v
}

func must2[T1, T2 any](v1 T1, v2 T2, err error) (T1, T2) {
	must(err)
	return v1, v2
}
