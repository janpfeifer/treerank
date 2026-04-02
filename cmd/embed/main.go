package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/gomlx/go-huggingface/datasets"
	"github.com/gomlx/go-huggingface/examples/kalmgemma3"
	"github.com/gomlx/go-huggingface/examples/msmarco"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/go-huggingface/tokenizers"
	tapi "github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/go-huggingface/tokenizers/bucket"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"k8s.io/klog/v2"
)

var (
	flagRepository   = flag.String("repo", kalmgemma3.Repository, "Path to the repository")
	flagData         = flag.String("data", "", "Data directory where files should be generated.")
	flagLimit        = flag.Int("limit", -1, "Limit the number of queries indexed. Set <= 0 to use all.")
	flagMSMarcoSplit = flag.String("msmarco_split", msmarco.ValidationSplit, "Split to read from MS MARCO dataset (e.g. 'train', 'validation', 'test')")
)

type Work struct {
	IsQuery bool
	ID      int32
}

func main() {
	flag.Parse()

	if *flagData == "" {
		klog.Fatalf("-data flag is required to generate datasets")
	}

	repo, err := kalmgemma3.LoadRepo()
	if err != nil {
		klog.Fatalf("Failed to init repo: %v", err)
	}

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

	mustRunWithElapsedTime("Uploading variables to device", func() (any, error) {
		for v := range ctx.IterVariables() {
			t, err := v.Value()
			if err != nil {
				return nil, err
			}
			if err = t.MaterializeOnDevice(backend, false, 0); err != nil {
				return nil, err
			}
			t.FinalizeLocal()
			runtime.GC()
		}
		return nil, nil
	})

	padID := 0
	if id, err := tokenizer.SpecialTokenID(tapi.TokPad); err == nil {
		padID = id
	}

	embedExec := mustRunWithElapsedTime("Compiling execution graph", func() (*context.Exec, error) {
		exec, err := context.NewExec(backend, ctx.Checked(false), func(ctx *context.Context, tokens *graph.Node) *graph.Node {
			mask := graph.NotEqual(tokens, graph.Const(tokens.Graph(), int32(padID)))
			x := model.SentenceEmbeddingGraph(ctx, tokens, mask)
			return graph.ConvertDType(x, dtypes.Float32)
		})
		return exec, err
	})

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

	inputChan := make(chan bucket.SentenceRef)
	outputChan := make(chan bucket.Bucket, 10)

	bkt := bucket.New(tokenizer).
		ByPower(32, 8, 2).
		WithMaxDelay(100*time.Millisecond, true).
		WithMaxParallelization(-1)

	go bkt.Run(inputChan, outputChan)

	var numQueries int32
	var numPassages int32
	producerDone := make(chan struct{})

	go func() {
		defer close(producerDone)
		defer close(inputChan)
		ds := datasets.New(msmarco.ID)

		count := 0
		limit := *flagLimit

		for record, err := range datasets.IterParquetFromDataset[msmarco.MsMarcoRecord](ds, msmarco.Config, *flagMSMarcoSplit) {
			if err != nil {
				klog.Fatalf("Dataset iterator error: %v", err)
			}

			qID := numQueries
			numQueries++

			inputChan <- bucket.SentenceRef{
				Sentence:  record.Query,
				Reference: Work{IsQuery: true, ID: qID},
			}

			var indices [10]int32
			for i := range indices {
				indices[i] = -1
			}
			var isSelected [10]byte
			for i := range isSelected {
				isSelected[i] = 0
			}

			pLens := len(record.Passages.PassageText)
			if pLens > 10 {
				pLens = 10
			}

			for i := 0; i < pLens; i++ {
				text := record.Passages.PassageText[i]
				if text == "" {
					continue
				}

				pID, exists := passageToID[text]
				if !exists {
					pID = numPassages
					numPassages++
					passageToID[text] = pID
					inputChan <- bucket.SentenceRef{
						Sentence:  text,
						Reference: Work{IsQuery: false, ID: pID},
					}
				}
				indices[i] = pID
				if i < len(record.Passages.IsSelected) && record.Passages.IsSelected[i] > 0 {
					isSelected[i] = 1
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
	}()

	fmt.Println("Starting processing...")

	const embeddingDim = 3840
	const bytePerFloat = 4
	embeddingByteLen := embeddingDim * bytePerFloat

	startTime := time.Now()
	var totalRecordsProcessed int32
	var emaSpeed float64
	var emaInitialized bool

	for bk := range outputChan {
		if bk.Error != nil {
			klog.Fatalf("Tokenization error: %v", bk.Error)
		}

		batchSize := bk.Shape.BatchSize
		seqLen := bk.Shape.SentenceLength

		flatData := xslices.Map(bk.Batch, func(i int) int32 { return int32(i) })
		inputTensor := tensors.FromFlatDataAndDimensions(flatData, batchSize, seqLen)

		batchStartTime := time.Now()
		results, err := embedExec.Exec(inputTensor)
		if err != nil {
			klog.Fatalf("Failed to execute embeddings: %v", err)
		}

		outTensor := results[0]
		outTensor.ConstFlatData(func(flatAny any) {
			flatFloat32 := flatAny.([]float32)

			for i := 0; i < batchSize; i++ {
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
		results[0].FinalizeAll()

		totalRecordsProcessed += int32(batchSize)
		
		batchDuration := time.Since(batchStartTime).Seconds()
		if batchDuration > 0 {
			currentSpeed := float64(batchSize) / batchDuration
			if !emaInitialized {
				emaSpeed = currentSpeed
				emaInitialized = true
			} else {
				emaSpeed = 0.1*currentSpeed + 0.9*emaSpeed
			}
		}

		if totalRecordsProcessed%100 == 0 || totalRecordsProcessed < 10 {
			expectedTotal := float64(numQueries + numPassages)
			var expectedRemaining string = "Unknown"
			if expectedTotal > float64(totalRecordsProcessed) && emaSpeed > 0 {
				remains := int((expectedTotal - float64(totalRecordsProcessed)) / emaSpeed)
				expectedRemaining = (time.Duration(remains) * time.Second).String()
			}
			fmt.Printf("Processed %d embed requests... [%.2f it/s] Expected remaining time (approx): %s\n", 
				totalRecordsProcessed, emaSpeed, expectedRemaining)
		}
	}

	<-producerDone

	fmt.Printf("\nDone.\nTotal duration: %v\n", time.Since(startTime))
	fmt.Printf("Total Queries: %d\n", numQueries)
	fmt.Printf("Total Unique Passages: %d\n", numPassages)
}

func mustRunWithElapsedTime[T any](name string, f func() (T, error)) T {
	fmt.Printf("%s...", name)
	start := time.Now()
	ret, err := f()
	if err != nil {
		klog.Fatalf("failed: %v\n", err)
	}
	fmt.Printf("done (%v)\n", time.Since(start))
	return ret
}

type ioWriterAt struct {
	f      *os.File
	offset int64
}

func (w ioWriterAt) Write(p []byte) (n int, err error) {
	return w.f.WriteAt(p, w.offset)
}
