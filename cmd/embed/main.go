package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math/bits"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/lipgloss/table"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/go-huggingface/datasets"
	"github.com/gomlx/go-huggingface/examples/kalmgemma3"
	"github.com/gomlx/go-huggingface/examples/msmarco"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/go-huggingface/tokenizers"
	tapi "github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/go-huggingface/tokenizers/bucket"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/janpfeifer/treerank/internal/humanize"
	"k8s.io/klog/v2"
)

var (
	flagData         = flag.String("data", "", "Data directory where files should be generated.")
	flagLimit        = flag.Int("limit", -1, "Limit the number of queries indexed. Set <= 0 to use all.")
	flagMSMarcoSplit = flag.String("msmarco_split", msmarco.ValidationSplit, "Split to read from MS MARCO dataset (e.g. 'train', 'validation', 'test')")
	flagTask         = flag.String("task", "", "Task selection (for queries), it adds a prompt accordingly. "+
		"If empty no prompt is prepended. "+
		"Set to '?' or 'list' to list supported values.")
)

type Work struct {
	IsQuery bool
	ID      int32
}

func MapHas[K comparable, V any](m map[K]V, k K) bool {
	_, ok := m[k]
	return ok
}

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	if *flagData == "" {
		klog.Fatalf("-data flag is required to generate datasets")
	}

	repo, err := kalmgemma3.LoadRepo()
	if err != nil {
		klog.Fatalf("Failed to init repo: %v", err)
	}

	taskPrompts := taskSelection(repo) // load task prompts if needed

	model := mustRunWithElapsedTime("Loading model configurations", func() (*transformer.Model, error) {
		return transformer.LoadModel(repo)
	})

	tokenizer := mustRunWithElapsedTime("Loading tokenizer", func() (tokenizers.Tokenizer, error) {
		return model.GetTokenizer()
	})

	backend := mustRunWithElapsedTime("Initializing backend", func() (backends.Backend, error) {
		return backends.New()
	})
	ctx := context.New()

	mustRunWithElapsedTime("Loading variables into context", func() (any, error) {
		return nil, model.LoadContext(backend, ctx)
	})

	padID := 0
	if id, err := tokenizer.SpecialTokenID(tapi.TokPad); err == nil {
		padID = id
	}

	embedExec, err := context.NewExec(backend, ctx.Checked(false), func(ctx *context.Context, tokens *graph.Node) *graph.Node {
		fmt.Printf("\n\t - Compiling execution graph for %s (%s tokens)\n",
			tokens.Shape(), humanize.Count(int64(tokens.Shape().Size())))
		constPadID := graph.Scalar(tokens.Graph(), tokens.DType(), padID)
		mask := graph.NotEqual(tokens, constPadID)
		x := model.SentenceEmbeddingGraph(ctx, tokens, mask)
		return graph.ConvertDType(x, dtypes.Float32)
	})

	// Prepare output files in split sub-directory.
	splitDir := filepath.Join(*flagData, *flagMSMarcoSplit)
	if err := os.MkdirAll(splitDir, 0755); err != nil {
		klog.Fatalf("Failed to create split directory: %v", err)
	}

	openBin := func(name string) *os.File {
		f, err := os.OpenFile(filepath.Join(splitDir, name), os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0644)
		if err != nil {
			klog.Fatalf("Failed to create file %s: %v", name, err)
		}
		return f
	}

	fQueries := openBin("queries.bin")
	defer fQueries.Close()
	fPassages := openBin("passages.bin")
	defer fPassages.Close()
	fQueryIndices := openBin("query_indices.bin")
	defer fQueryIndices.Close()
	fQueryIsSelected := openBin("query_is_selected.bin")
	defer fQueryIsSelected.Close()
	passageToID := make(map[string]int32, 1000)

	// Structured concurrency (keep track of goroutines).
	var wg sync.WaitGroup

	// Start bucket runner in a separate goroutine.
	bucketsInputChan := make(chan bucket.SentenceRef)
	bucketsOutputChan := make(chan bucket.Bucket, 10)
	bkt := bucket.New(tokenizer).
		ByPowerBudget(8*1024, 16, 1.4).
		WithMaxParallelization(-1)
	wg.Go(func() {
		bkt.Run(bucketsInputChan, bucketsOutputChan)
	})

	// Dataset preparation and stats.
	ds := datasets.New(msmarco.ID)
	limit := *flagLimit
	var numQueriesRead int32
	var numPassagesRead int32
	dsInfo, err := ds.Info()
	if err != nil {
		klog.Fatalf("Failed to get dataset info: %v", err)
	}
	if !MapHas(dsInfo.DatasetInfo, msmarco.Config) || !MapHas(dsInfo.DatasetInfo[msmarco.Config].Splits, *flagMSMarcoSplit) {
		klog.Fatalf("Dataset %q doesn't contents for config=%q / split=%q", ds.ID, msmarco.Config, *flagMSMarcoSplit)
	}
	splitInfo := dsInfo.DatasetInfo[msmarco.Config].Splits[*flagMSMarcoSplit]
	totalQueries := splitInfo.NumExamples
	fmt.Printf("- Dataset %q, split %q: %d queries in total\n", ds.ID, *flagMSMarcoSplit, totalQueries)
	if limit > 0 {
		limit = min(limit, int(totalQueries))
	}

	// Start goroutine that feeds the bucket runner with queries and passages.
	// It also sequentially saves the fQueryIndices and fQueryIsSelected files.
	wg.Go(func() {
		defer close(bucketsInputChan)
		count := 0
		for record, err := range datasets.IterParquetFromDataset[msmarco.MsMarcoRecord](ds, msmarco.Config, *flagMSMarcoSplit) {
			if err != nil {
				klog.Fatalf("Dataset iterator error: %v", err)
			}

			queryID := numQueriesRead
			numQueriesRead++
			query := record.Query
			if *flagTask != "" {
				query = taskPrompts.BuildQueryPrompt(record.Query, *flagTask)
			}

			bucketsInputChan <- bucket.SentenceRef{
				Sentence:  query,
				Reference: Work{IsQuery: true, ID: queryID},
			}

			var indices [10]int32
			var isSelected [10]byte
			for i := range indices {
				indices[i] = -1
			}

			// There should be at most 10 passages per query in the datasets, but
			// just in case we enforce the limit.
			pLens := min(len(record.Passages.PassageText), 10)
			for queryPassageIdx := range pLens {
				text := record.Passages.PassageText[queryPassageIdx]
				if text == "" {
					continue
				}

				// Registers new passage if it doesn't exist yet.
				passageID, exists := passageToID[text]
				if !exists {
					passageID = numPassagesRead
					numPassagesRead++
					passageToID[text] = passageID
					bucketsInputChan <- bucket.SentenceRef{
						Sentence:  text,
						Reference: Work{IsQuery: false, ID: passageID},
					}
				}
				indices[queryPassageIdx] = passageID
				if queryPassageIdx < len(record.Passages.IsSelected) && record.Passages.IsSelected[queryPassageIdx] > 0 {
					isSelected[queryPassageIdx] = 1
				}
			}

			if err := binary.Write(fQueryIndices, binary.LittleEndian, indices); err != nil {
				klog.Fatalf("Failed to write indices: %v", err)
			}
			if err := binary.Write(fQueryIsSelected, binary.LittleEndian, isSelected); err != nil {
				klog.Fatalf("Failed to write is selected: %v", err)
			}

			count++
			if limit > 0 && count >= limit {
				break
			}
		}
	})

	// Process batches and save embeddings.
	const embeddingDim = 3840
	const bytePerFloat = 4
	embeddingByteLen := embeddingDim * bytePerFloat

	startTime := time.Now()
	var numTokensProcessed, numNonPadTokensProcessed int64
	var numSentencesProcessed, numQueriesProcessed int
	expectedNumQueries := limit
	if expectedNumQueries <= 0 {
		expectedNumQueries = int(totalQueries)
	}
	var emaSpeed float64
	var emaInitialized bool

	wg.Go(func() {

		fmt.Printf("- Starting processing:\n")
		lastReportTime := time.Now()
		var queriesPerSecond float64
		for bk := range bucketsOutputChan {
			if bk.Error != nil {
				klog.Fatalf("Tokenization error: %v", bk.Error)
			}

			batchSize := bk.Shape.BatchSize
			seqLen := bk.Shape.SentenceLength

			rawData := dtypes.UnsafeByteSlice(bk.Batch)
			var dtype dtypes.DType
			switch bits.UintSize {
			case 32:
				dtype = dtypes.Int32
			case 64:
				dtype = dtypes.Int64
			default:
				klog.Fatalf("Unsupported int of %d-bits architecture", bits.UintSize)
			}
			inputTensor, err := tensors.FromRaw(backend, 0, shapes.Make(dtype, batchSize, seqLen), rawData)
			if err != nil {
				klog.Fatalf("Failed to create input tensor: %+v", err)
			}

			batchStartTime := time.Now()
			var outTensor *tensors.Tensor
			errPanic := exceptions.TryCatch[error](func() {
				outTensor, err = embedExec.Exec1(inputTensor)
			})
			if errPanic != nil {
				fmt.Println()
				klog.Fatalf("Panic on execute embeddings for %s: %+v", inputTensor.Shape(), errPanic)
			}
			if err != nil {
				fmt.Println()
				klog.Fatalf("Failed to execute embeddings for %s: %+v", inputTensor.Shape(), err)
			}
			numTokensProcessed += int64(len(bk.Batch))
			numNonPadTokensProcessed += int64(bk.NonPadTokens)
			outTensor.ConstFlatData(func(flatAny any) {
				flatFloat32 := flatAny.([]float32)
				for i := range batchSize {
					if bk.References[i] == nil {
						continue
					}
					ref, ok := bk.References[i].(Work)
					if !ok {
						continue
					}

					startIdx := i * embeddingDim
					embedRow := flatFloat32[startIdx : startIdx+embeddingDim]

					offset := int64(ref.ID) * int64(embeddingByteLen)
					var file *os.File
					if ref.IsQuery {
						file = fQueries
					} else {
						file = fPassages
					}

					err := binary.Write(ioWriterAt{file, offset}, binary.LittleEndian, embedRow)
					if err != nil {
						klog.Fatalf("Failed writing embedding: %v", err)
					}
				}
			})
			inputTensor.FinalizeAll()
			outTensor.FinalizeAll()

			// Moving average of (non-padding) tokens per second speed.
			batchDuration := time.Since(batchStartTime).Seconds()
			if batchDuration > 0 {
				currentSpeed := float64(bk.NonPadTokens) / batchDuration
				if !emaInitialized {
					emaSpeed = currentSpeed
					emaInitialized = true
				} else {
					emaSpeed = 0.1*currentSpeed + 0.9*emaSpeed
				}
			}

			// Count queries and sentences processed.
			numSentencesProcessed += batchSize
			for i := range batchSize {
				if bk.References[i] != nil && bk.References[i].(Work).IsQuery {
					numQueriesProcessed++
				}
			}

			// Report progress every second.
			if time.Since(lastReportTime) > time.Second {
				lastReportTime = time.Now()

				// ETA estimation.
				queriesPerSecond = float64(numQueriesProcessed) / time.Since(startTime).Seconds()
				eta := "Unknown"
				if numQueriesProcessed > 0 {
					remainingSeconds := float64(expectedNumQueries-numQueriesProcessed) / queriesPerSecond
					eta = humanize.Duration(time.Duration(int64(remainingSeconds*1e9)) * time.Nanosecond)
				}
				fmt.Printf("\r  - Processed %s / %s queries (%s, %s non-padding) -- ETA %s ...%s",
					humanize.Count(int64(numQueriesProcessed)), humanize.Count(int64(expectedNumQueries)), humanize.Speed(queriesPerSecond, "queries"),
					humanize.Speed(emaSpeed, "tokens"), eta, humanize.EraseToEndOfLine)
			}
		}
		fmt.Printf("\r  - Processed %s / %s queries (%s, %s non-padding) -- done.%s\n",
			humanize.Count(int64(numQueriesProcessed)), humanize.Count(int64(expectedNumQueries)), humanize.Speed(queriesPerSecond, "queries"),
			humanize.Speed(emaSpeed, "tokens"), humanize.EraseToEndOfLine)
	})

	wg.Wait()
	elapsed := time.Since(startTime)
	fmt.Printf("Total duration: %v\n", humanize.Duration(elapsed))

	// Print nice report table with counts and speeds.
	names := []string{"Queries", "Passages", "Tokens", "Non-Pad Tokens"}
	totals := []string{humanize.Count(int64(numQueriesRead)), humanize.Count(int64(numPassagesRead)), humanize.Count(numTokensProcessed), humanize.Count(numNonPadTokensProcessed)}
	speeds := []string{
		humanize.Speed(float64(numQueriesRead)/elapsed.Seconds(), " items"),
		humanize.Speed(float64(numPassagesRead)/elapsed.Seconds(), " items"),
		humanize.Speed(float64(numTokensProcessed)/elapsed.Seconds(), "tokens"),
		humanize.Speed(float64(numNonPadTokensProcessed)/elapsed.Seconds(), "tokens"),
	}
	baseStyle := lipgloss.NewStyle().Padding(0, 1)
	t := table.New().
		Border(lipgloss.NormalBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(lipgloss.Color("238"))).
		Headers("Metric", "Total", "Speed").
		StyleFunc(func(row, col int) lipgloss.Style {
			s := baseStyle
			if col > 0 && row != table.HeaderRow {
				s = s.Align(lipgloss.Right)
			}
			if row == table.HeaderRow {
				headerStyle := s.Foreground(lipgloss.Color("252")).Bold(true)
				if col > 0 {
					headerStyle = headerStyle.Align(lipgloss.Center)
				}
				return headerStyle
			}
			return s
		})
	for i := range names {
		t.Row(names[i], totals[i], speeds[i])
	}
	fmt.Println(t)
}

func mustRunWithElapsedTime[T any](name string, f func() (T, error)) T {
	fmt.Printf("%s...", name)
	start := time.Now()
	ret, err := f()
	if err != nil {
		klog.Fatalf("failed: %v\n", err)
	}
	fmt.Printf("done (%s)\n", humanize.Duration(time.Since(start)))
	return ret
}

type ioWriterAt struct {
	f      *os.File
	offset int64
}

func (w ioWriterAt) Write(p []byte) (n int, err error) {
	return w.f.WriteAt(p, w.offset)
}

func taskSelection(repo *hub.Repo) kalmgemma3.TaskPrompts {
	var taskPrompts kalmgemma3.TaskPrompts
	if *flagTask == "" {
		return taskPrompts
	}

	var err error
	taskPrompts, err = kalmgemma3.LoadTaskPrompts(repo)
	if err != nil {
		klog.Fatalf("Failed to load task prompts: %+v", err)
	}
	if *flagTask == "?" || *flagTask == "list" {
		fmt.Printf("Available task prompts:\n")
		for task, prompt := range taskPrompts {
			fmt.Printf("  %s: %s\n", task, prompt)
		}
		os.Exit(0)
	}
	if _, ok := taskPrompts[*flagTask]; !ok {
		klog.Fatalf("Unknown task prompt key: %s", *flagTask)
	}
	return taskPrompts
}
