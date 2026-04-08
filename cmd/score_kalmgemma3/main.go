package main

import (
	"flag"
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/gomlx/go-huggingface/datasets"
	"github.com/gomlx/go-huggingface/examples/msmarco"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/humanize"
	"github.com/janpfeifer/treerank/internal/datamanager"
	"github.com/parquet-go/parquet-go"
	"github.com/spf13/pflag"
	"k8s.io/klog/v2"
)

var (
	flagData    = pflag.String("data", "", "Data directory where files should be generated.")
	flagQueries = pflag.IntSlice("queries", nil, "Queries to score (indices). If empty, all queries are scored. "+
		"If set, it will also display the query text, as well as the Top-1 passage text and the selected passages.")
	flagMSMarcoSplit = flag.String("msmarco_split", msmarco.ValidationSplit,
		"Split to read from MS MARCO dataset (e.g. 'train', 'validation', 'test') used to diplay the queries. "+
			"It must match the split used to generate the data.")
)

func main() {
	klog.InitFlags(nil)
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine) // Add klog flags.
	pflag.Parse()

	if *flagData == "" {
		klog.Exit("--data <data directory> must be specified")
	}

	dm := must1(datamanager.New(*flagData))
	defer dm.Close()
	backend := must1(backends.New())
	defer backend.Finalize()

	// Print dataset and selected queries info.
	fmt.Printf("Dataset in %s:\n", *flagData)
	fmt.Printf("  - Total queries:  %d\n", dm.NumQueries)
	fmt.Printf("  - Total passages: %d\n", dm.NumPassages)
	queryIDs := *flagQueries
	if len(queryIDs) > 0 {
		for _, queryID := range queryIDs {
			if queryID < 0 || queryID >= dm.NumQueries {
				klog.Exitf("queryID %d (passed by --queries) is out of bounds [0, %d)", queryID, dm.NumQueries)
			}
		}
		fmt.Printf("  - Queries to score: %v\n", queryIDs)
	}

	// Investigate the distribution of selected passages per query.
	queryIsSelected := must1(dm.LoadQueryIsSelected(backend))
	PrintCountNumberSelected(backend, queryIsSelected)
	if len(queryIDs) > 0 {
		queryIsSelected = must1(ExecOnce(backend, func(queryIsSelected, queriesToScore *Node) *Node {
			return Gather(queryIsSelected, ExpandAxes(queriesToScore, -1))
		}, queryIsSelected, queryIDs))
		if queryIsSelected.Rank() != 2 && queryIsSelected.Shape().Dim(0) != len(queryIDs) {
			klog.Exitf("Expected %d queries selected, got %s", len(queryIDs), queryIsSelected.Shape())
		}
	}

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

	// Load queries.
	var queries *tensors.Tensor
	if len(queryIDs) == 0 {
		queries = must1(dm.LoadAllQueries(backend))
	} else {
		queries = must1(dm.LoadQueriesByIDs(backend, queryIDs...))
	}
	const passagesBatchSize = 32

	// currentTopKPassagesPerQuery starts with 0 passages selected, and keeps being updated with the topK for each batch.
	currentTopKPassagesPerQueryIndices := must1(tensors.FromShapeForBackend(backend, 0, shapes.Make(dtypes.Int32, queries.Shape().Dim(0), 0)))
	currentTopKPassagesPerQueryScores := must1(tensors.FromShapeForBackend(backend, 0, shapes.Make(dtypes.Float32, queries.Shape().Dim(0), 0)))
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

	// Loop at a batch of passages at a time, scoring them against the queries loaded.
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

	// Load queryIndices: map of queryID -> list of passageIDs
	allQueryIndices := must1(dm.LoadQueryIndices(backend))
	queryIndices := allQueryIndices
	if len(queryIDs) > 0 {
		queryIndices = must1(ExecOnce(backend, func(queryIndices, queryIDs *Node) *Node {
			return Gather(queryIndices, ExpandAxes(queryIDs, -1))
		}, queryIndices, queryIDs))
		if queryIndices.Rank() != 2 && queryIndices.Shape().Dim(0) != len(queryIDs) {
			klog.Exitf("Expected %d queries indices, got %s", len(queryIDs), queryIndices.Shape())
		}
	}

	// Now that we have the TopK passages for each query, we can compute the MRR, by checking if any of them
	// are the selected passages for the query.
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

	if len(queryIDs) > 0 {
		PrintQueries(queryIDs, queryIsSelected, allQueryIndices, currentTopKPassagesPerQueryIndices)
	}
}

// PrintQueries prints the text of the queries, the selected passages and the top passage of each query.
func PrintQueries(queryIDs []int, queryIsSelected, allQueryIndices, topKPassagesPerQueryIndices *tensors.Tensor) {
	// Get top scored passages for each query
	topPassageIDs := make([]int, len(queryIDs))
	tensors.ConstFlatData(topKPassagesPerQueryIndices, func(flatTopK []int32) {
		stride := topKPassagesPerQueryIndices.Shape().Dim(1)
		for i := range queryIDs {
			topPassageIDs[i] = int(flatTopK[i*stride])
		}
	})

	// Get query IDs that hold the top passages -- they are not necessarily the same as the queryIDs we are listing.
	topPassageQueryIDs := make([]int, len(queryIDs))
	topPassageQuerySubIndex := make([]int, len(queryIDs))
	tensors.ConstFlatData(allQueryIndices, func(flatQueryIndices []int32) {
		stride := allQueryIndices.Shape().Dim(1)
		numFound := 0
		for queryIndicesFlatIdx, passageID := range flatQueryIndices {
			for topIdx, topPassageID := range topPassageIDs {
				if topPassageID != int(passageID) {
					continue
				}
				topPassageQueryIDs[topIdx] = queryIndicesFlatIdx / stride
				topPassageQuerySubIndex[topIdx] = queryIndicesFlatIdx % stride
				numFound++
			}
			if numFound == len(topPassageIDs) {
				break
			}
		}
		if numFound != len(topPassageIDs) {
			klog.Fatalf("Expected %d top passages, found %d", len(topPassageIDs), numFound)
		}
	})

	// Get reader from dataset, so we can fetch the text of the queries and passages.
	ds := datasets.New(msmarco.ID)
	dsInfo := must1(ds.Info())
	if !MapHas(dsInfo.DatasetInfo, msmarco.Config) || !MapHas(dsInfo.DatasetInfo[msmarco.Config].Splits, *flagMSMarcoSplit) {
		klog.Fatalf("Dataset %q doesn't contents for config=%q / split=%q", ds.ID, msmarco.Config, *flagMSMarcoSplit)
	}
	reader := must1(datasets.CreateParquetReader[msmarco.MsMarcoRecord](ds, msmarco.Config, *flagMSMarcoSplit))
	defer reader.Close()

	separator := lipgloss.NewStyle().Foreground(lipgloss.Color("238")).Render(strings.Repeat("─", 120))
	queryStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#80FF80"))
	passageStyle := lipgloss.NewStyle().Faint(true)

	for queryIDIdx, queryID := range queryIDs {
		record := getQueryRecord(reader, queryID)
		fmt.Println(separator)
		fmt.Printf("Query #%d: %s\n", queryID, queryStyle.Render(record.Query))

		countSelected := 0
		for i, isSelected := range record.Passages.IsSelected {
			if isSelected > 0 {
				fmt.Printf("  - Selected #%d: %s\n", countSelected+1,
					passageStyle.Render(record.Passages.PassageText[i]))
				countSelected++
			}
		}
		if countSelected == 0 {
			fmt.Printf("  - No passages selected\n")
		}

		fmt.Printf("  - Top scored passage (from query #%d, passage #%d):\n",
			topPassageQueryIDs[queryIDIdx], topPassageQuerySubIndex[queryIDIdx])
		topRecord := getQueryRecord(reader, topPassageQueryIDs[queryIDIdx])
		topPassage := topRecord.Passages.PassageText[topPassageQuerySubIndex[queryIDIdx]]
		fmt.Printf("    %s\n", passageStyle.Render(topPassage))
	}
	fmt.Println(separator)
}

func getQueryRecord(reader *parquet.GenericReader[msmarco.MsMarcoRecord], rowID int) msmarco.MsMarcoRecord {
	batch := make([]msmarco.MsMarcoRecord, 1)
	reader.SeekToRow(int64(rowID))
	n := must1(reader.Read(batch))
	if n != 1 {
		klog.Fatalf("Expected 1 record, got %d", n)
	}
	return batch[0]
}

func MapHas[K comparable, V any](m map[K]V, k K) bool {
	_, ok := m[k]
	return ok
}

func PrintCountNumberSelected(backend backends.Backend, queryIsSelected *tensors.Tensor) {
	perCount := must1(ExecOnce(backend, func(queryIsSelected *Node) *Node {
		g := queryIsSelected.Graph()
		numSelected := GreaterThan(queryIsSelected, ScalarZero(g, queryIsSelected.DType()))
		countSelected := ReduceSum(ConvertDType(numSelected, dtypes.Int8), 1)

		perCount := IotaFull(g, shapes.Make(dtypes.Int8, 1, 10))      // Shape: [1, 10]
		perCount = Equal(ExpandAxes(countSelected, -1), perCount)     // Shape: [NumQueries, 10]Bool
		perCount = ReduceSum(ConvertDType(perCount, dtypes.Int32), 0) // Shape: [10]Int32
		return perCount
	}, queryIsSelected))
	counts := perCount.Value().([]int32)
	fmt.Printf("  - Distribution of selected passages: ([<num_selected_passages>:<num_queries>])\n")
	fmt.Printf("    - ")
	comma := ""
	for numSelected, count := range counts {
		if count > 0 {
			fmt.Printf("%s[%d:%d]", comma, numSelected, count)
			comma = ", "
		}
	}
	fmt.Println()
}
