package main

import (
	"flag"
	"fmt"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
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

	// Calculate the MRR of the queries.
	queries := must1(dm.LoadQueries(backend))
	passages := must1(dm.LoadPassages(backend))
	queryIndices := must1(dm.LoadQueryIndices(backend))
	mrr := must1(ExecOnce(backend, func(queries, passages, queryIndices, queryIsSelected *Node) *Node {
		g := queries.Graph()
		dtype := queries.DType()
		zero := ScalarZero(g, dtype)
		one := ScalarOne(g, dtype)
		numQueries := queries.Shape().Dim(0)
		numPassages := passages.Shape().Dim(0)

		// 1. Similarity: shape [NumQueries, NumPassages]
		similarity := CrossCosineSimilarity(queries, passages, -1, 0)

		// 2. isSelected: shape [NumQueries, 10]Bool
		isSelected := GreaterThan(queryIsSelected, ScalarZero(g, queryIsSelected.DType()))
		isSelected = ConvertDType(isSelected, dtypes.Int32)
		countSelected := ReduceSum(isSelected, -1)
		_ = countSelected

		// 3. scatteredIsSelected: shape [NumQueries, NumPassages]
		queryIndices = Max(queryIndices, ScalarZero(g, queryIndices.DType())) // [NumQueries, 10]
		queryIndices = Stack([]*Node{
			Iota(g, shapes.Make(dtypes.Int32, numQueries, 10), 0), // row numbers (queryID)
			queryIndices, // column numbers (query's passageID)
		}, -1)
		scatteredIsSelected := Scatter(queryIndices, isSelected, shapes.Make(dtypes.Int32, numQueries, numPassages), false, true)
		scatteredIsSelected.AssertDims(numQueries, numPassages)

		// 4. Sorte scatteredIsSelected by similarities (descending).
		compareSimilarities := NewClosure(g, func(g *Graph) []*Node {
			lhsSimilarity := Parameter(g, "lhsSimilarity", shapes.Make(dtype))
			rhsSimilarity := Parameter(g, "rhsSimilarity", shapes.Make(dtype))
			_ = Parameter(g, "lhsIsSelected", shapes.Make(dtypes.Int32))
			_ = Parameter(g, "rhsIsSelected", shapes.Make(dtypes.Int32))
			return []*Node{GreaterThan(lhsSimilarity, rhsSimilarity)}
		})
		sorted := SortFunc(compareSimilarities, 1, false, similarity, scatteredIsSelected)
		isSelectedSortedBySimilarity := sorted[1]

		// 5. RR (reciprocal rank, 1/(rank+1)) score of each passage -- later we select only those selected.
		rrScores := Reciprocal(OnePlus(Iota(g, shapes.Make(dtype, numQueries, numPassages), 1)))
		rrScores = Where(GreaterThan(isSelectedSortedBySimilarity, ScalarZero(g, isSelectedSortedBySimilarity.DType())),
			rrScores, zero)

		// 6. Take only the RR of the best (maximum) selected of the possible correct passages.
		rrScores = ReduceMax(rrScores, -1)

		// 7. Discard selections below Top-K, with K=10 (used in the official leaderboard)
		minRR := Scalar(g, dtype, 1.0/10.0)
		rrScores = Where(LessThan(rrScores, minRR), zero, rrScores)

		// 8. Calculate the MRR by taking the mean over only the queries with non-zero selections.
		//    (This may differe from the leaderboard, I'm not sure if they include tests with no selections and then count these as 0)
		validQueries := Max(ReduceSum(ConvertDType(GreaterThan(countSelected, ScalarZero(g, countSelected.DType())), dtype)), one)
		mrr := Div(ReduceSum(rrScores), validQueries)
		return mrr
	}, queries, passages, queryIndices, queryIsSelected))

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
