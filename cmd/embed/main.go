package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"hash/maphash"
	"math/bits"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"sync/atomic"
	"syscall"
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
	"github.com/gomlx/gomlx/pkg/support/humanize"
	"k8s.io/klog/v2"
)

var (
	flagData         = flag.String("data", "", "Data directory where files should be generated.")
	flagLimit        = flag.Int("limit", -1, "Limit the number of queries indexed. Set <= 0 to use all.")
	flagMSMarcoSplit = flag.String("msmarco_split", msmarco.ValidationSplit, "Split to read from MS MARCO dataset (e.g. 'train', 'validation', 'test')")
	flagTask         = flag.String("task", "MSMARCO", "Task selection (for queries), it adds a prompt accordingly. "+
		"If empty no prompt is prepended. "+
		"Set to '?' or 'list' to list supported values.")
	flagResume = flag.Bool("resume", false, "Resume generation from where it stopped.")
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
		// fmt.Printf("\n\t - Compiling execution graph for %s (%s tokens)\n",
		// 	tokens.Shape(), humanize.Count(int64(tokens.Shape().Size())))
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
		flags := os.O_CREATE | os.O_RDWR
		if !*flagResume {
			flags |= os.O_TRUNC
		}
		f, err := os.OpenFile(filepath.Join(splitDir, name), flags, 0644)
		if err != nil {
			klog.Fatalf("Failed to create file %s: %v", name, err)
		}
		return f
	}

	fQueries := openBin("queries.bin")
	defer fQueries.Close()
	fPassages := openBin("passages.bin")
	defer fPassages.Close()
	fQueryIndices := openBin("queries_passage_ids.bin")
	defer fQueryIndices.Close()
	fQueryIsSelected := openBin("queries_is_selected.bin")
	defer fQueryIsSelected.Close()
	var passageDict *PassageDictionary
	var numQueriesRead int32

	if *flagResume {
		dictPath := filepath.Join(splitDir, "passage_dictionary.bin")
		var err error
		passageDict, err = LoadPassageDictionary(dictPath)
		if err != nil {
			klog.Fatalf("Failed to load passage dictionary: %v", err)
		}

		qStat, _ := fQueries.Stat()
		nq1 := qStat.Size() / (3840 * 4) // embeddingDim * 4 bytes

		qiStat, _ := fQueryIndices.Stat()
		nq2 := qiStat.Size() / 40 // 10 int32s = 40 bytes

		qsStat, _ := fQueryIsSelected.Stat()
		nq3 := qsStat.Size() / 10 // 10 bytes = 10 bytes

		if nq1 != nq2 || nq1 != nq3 {
			klog.Fatalf("Cannot resume: inconsistent file sizes! queries.bin=%d, indices=%d, selected=%d queries", nq1, nq2, nq3)
		}

		numQueriesRead = int32(nq1)
		fmt.Printf("- Resuming generation from query %d (loaded %d passages from dictionary).\n", numQueriesRead, passageDict.Len())

		if _, err := fQueryIndices.Seek(int64(numQueriesRead)*40, 0); err != nil {
			klog.Fatalf("Failed to seek queries_passage_ids.bin: %v", err)
		}
		if _, err := fQueryIsSelected.Seek(int64(numQueriesRead)*10, 0); err != nil {
			klog.Fatalf("Failed to seek queries_is_selected.bin: %v", err)
		}

		pStat, _ := fPassages.Stat()
		np := pStat.Size() / (3840 * 4)
		if np < int64(passageDict.Len()) {
			klog.Fatalf("Cannot resume: passages.bin size (%d embeddings) is surprisingly smaller than dictionary size (%d)", np, passageDict.Len())
		}
	} else {
		passageDict = NewPassageDictionary()
	}

	// Structured concurrency (keep track of goroutines).
	var wg sync.WaitGroup

	// Start bucket runner in a separate goroutine.
	bucketsInputChan := make(chan bucket.SentenceRef, 5)
	bucketsOutputChan := make(chan bucket.Bucket, 10)
	bkt := bucket.New(tokenizer).
		ByTwoBitBucketBudget(8*1024, 16).
		WithMaxParallelization(-1)
	wg.Go(func() {
		bkt.Run(bucketsInputChan, bucketsOutputChan)
	})

	// Dataset preparation and stats.
	ds := datasets.New(msmarco.ID)
	limit := *flagLimit
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
	if *flagTask != "" {
		fmt.Printf("  - Task: %q -> %q\n", *flagTask, taskPrompts.BuildQueryPrompt("...", *flagTask))
	}
	if limit > 0 {
		limit = min(limit, int(totalQueries))
	}

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	var interrupted atomic.Bool

	var wgInterrupt sync.WaitGroup
	wgInterrupt.Go(func() {
		_, ok := <-sigChan
		if !ok {
			return
		}
		fmt.Printf("\n🛑 Interrupted: stopping reading dataset early, flushing buffers (it will take a minute or so). " +
			"\n   Interrupt again (control+C) to abort immediately and lose all progress.\n")
		interrupted.Store(true)

		_, ok = <-sigChan
		if !ok {
			return
		}
		fmt.Printf("\n🛑 Interrupted again: aborting immediately! Anything saved can't be resumed later.\n")
		os.Exit(1)
	})

	// Start goroutine that feeds the bucket runner with queries and passages.
	// It also sequentially saves the fQueryIndices and fQueryIsSelected files.
	wg.Go(func() {
		defer close(bucketsInputChan)
		if limit > 0 && int(numQueriesRead) >= limit {
			fmt.Printf("- Target limit of %d queries already reached.\n", limit)
			return
		}

		startAt := int64(numQueriesRead)
		for record, err := range datasets.IterParquetFromDatasetAt[msmarco.MsMarcoRecord](ds, msmarco.Config, *flagMSMarcoSplit, startAt) {
			if interrupted.Load() {
				break
			}

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
				passageID, isNew := passageDict.GetOrAdd(text)
				if isNew {
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

			if limit > 0 && int(numQueriesRead) >= limit {
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
	initialQueriesProcessed := int(numQueriesRead)
	initialSentencesProcessed := initialQueriesProcessed + passageDict.Len()
	var emaSpeed float64
	var emaInitialized bool

	wg.Go(func() {

		fmt.Printf("- Starting processing:\n")
		lastReportTime := time.Now()
		var sentencesPerSecond float64
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
				sentencesPerSecond = float64(numSentencesProcessed) / time.Since(startTime).Seconds()
				eta := "Unknown"
				totalSentences := initialSentencesProcessed + numSentencesProcessed
				expectedTotalSentences := 11 * expectedNumQueries
				if numQueriesProcessed > 0 {
					remainingSeconds := float64(expectedTotalSentences-totalSentences) / sentencesPerSecond
					eta = humanize.Duration(time.Duration(int64(remainingSeconds*1e9)) * time.Nanosecond)
				}
				fmt.Printf("\r   - Processed %s / %s queries+passages (%s, %s non-padding) -- ETA %s ...%s",
					humanize.Count(int64(totalSentences)), humanize.Count(int64(expectedTotalSentences)), humanize.Speed(sentencesPerSecond, "queries+passages"),
					humanize.Speed(emaSpeed, "tokens"), eta, humanize.EraseToEndOfLine)
			}
		}
		totalSentences := initialSentencesProcessed + numSentencesProcessed
		expectedTotalSentences := 11 * expectedNumQueries
		fmt.Printf("\r  ✅ Processed %s / %s queries+passages (%s, %s non-padding) -- done.%s\n",
			humanize.Count(int64(totalSentences)), humanize.Count(int64(expectedTotalSentences)), humanize.Speed(sentencesPerSecond, "queries+passages"),
			humanize.Speed(emaSpeed, "tokens"), humanize.EraseToEndOfLine)
	})

	wg.Wait()
	elapsed := time.Since(startTime)
	fmt.Printf("Total duration: %v\n", humanize.Duration(elapsed))

	// Save passage dictionary, if one wants to continue or merge datasets later.
	dictPath := filepath.Join(splitDir, "passage_dictionary.bin")
	fmt.Printf("- Saving PassageDictionary to %q\n", dictPath)
	if err := passageDict.Save(dictPath); err != nil {
		klog.Errorf("Failed to save passage dictionary: %v", err)
	}

	// Print nice report table with counts and speeds.
	numPassagesRead := passageDict.Len()
	names := []string{"Queries", "Passages", "Tokens", "Non-Pad Tokens"}
	totals := []string{humanize.Count(int64(numQueriesRead)), humanize.Count(int64(numPassagesRead)), humanize.Count(numTokensProcessed), humanize.Count(numNonPadTokensProcessed)}
	speeds := []string{
		humanize.Speed(float64(numQueriesProcessed)/elapsed.Seconds(), " items"),
		humanize.Speed(float64(numSentencesProcessed-numQueriesProcessed)/elapsed.Seconds(), " items"),
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

	signal.Stop(sigChan)
	close(sigChan)
	wgInterrupt.Wait()
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

// PassageDictionary encapsulates mapping of passages to unique integer IDs.
type PassageDictionary struct {
	mapPassages map[uint64]int32
	hasher      maphash.Hash
}

func NewPassageDictionary() *PassageDictionary {
	return &PassageDictionary{
		mapPassages: make(map[uint64]int32, 1000),
	}
}

func (pd *PassageDictionary) GetOrAdd(text string) (id int32, isNew bool) {
	pd.hasher.Reset()
	pd.hasher.WriteString(text)
	hash := pd.hasher.Sum64()

	var ok bool
	id, ok = pd.mapPassages[hash]
	if !ok {
		id = int32(len(pd.mapPassages))
		pd.mapPassages[hash] = id
		isNew = true
	}
	return
}

func (pd *PassageDictionary) Len() int {
	return len(pd.mapPassages)
}

func (pd *PassageDictionary) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	size := int64(len(pd.mapPassages))
	if err := binary.Write(f, binary.LittleEndian, size); err != nil {
		return err
	}

	for hash, id := range pd.mapPassages {
		if err := binary.Write(f, binary.LittleEndian, hash); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, id); err != nil {
			return err
		}
	}
	return nil
}

func LoadPassageDictionary(path string) (*PassageDictionary, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	pd := NewPassageDictionary()
	var size int64
	if err := binary.Read(f, binary.LittleEndian, &size); err != nil {
		return nil, err
	}

	for i := int64(0); i < size; i++ {
		var hash uint64
		var id int32
		if err := binary.Read(f, binary.LittleEndian, &hash); err != nil {
			return nil, err
		}
		if err := binary.Read(f, binary.LittleEndian, &id); err != nil {
			return nil, err
		}
		pd.mapPassages[hash] = id
	}
	return pd, nil
}
