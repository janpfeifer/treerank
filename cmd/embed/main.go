package main

import (
	"flag"
	"fmt"
	"iter"
	"runtime"
	"strings"
	"time"

	"encoding/binary"
	"os"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"

	"github.com/charmbracelet/lipgloss"
	"github.com/gomlx/go-huggingface/datasets"
	"github.com/gomlx/go-huggingface/examples/kalmgemma3"
	"github.com/gomlx/go-huggingface/examples/msmarco"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"k8s.io/klog/v2"
)

var (
	flagRepository = flag.String("repo", kalmgemma3.Repository, "Path to the repository")

	flagTokens = flag.Bool("tokens", false, "Print tokens for each sentence")

	flagEmbeddings          = flag.Bool("embeddings", false, "Compute embeddings for each sentence")
	outputSentencesPerShard = flag.Int("output_sentences_per_shard", 100_000, "Number of sentences per shard")
	outputFile              = flag.String("output", "", "Output file(s) for embeddings (default 100K sentences per file). "+
		"The name is suffixed with 'XXXXX-of-XXXXX' for the shards.")

	flagMSMarco      = flag.Bool("msmarco", false, "Pull sentences from MS MARCO dataset")
	flagMSMarcoSplit = flag.String("msmarco_split", msmarco.ValidationSplit, "Split to read from MS MARCO dataset (e.g. 'train', 'validation', 'test')")
	flagNumSentences = flag.Int("num_sentences", 10, "Number of sentences to pull from MS MARCO dataset. Set to any value <= 0 and it will use all of them.")
	flagPassages     = flag.Bool("passages", false, "Pull passages instead of queries from MS MARCO dataset")
)

func main() {
	flag.Parse()

	var err error
	var repo *hub.Repo
	if *flagTokens || *flagEmbeddings {
		fmt.Println("Initializing repository:", *flagRepository)
		repo, err = kalmgemma3.LoadRepo()
		if err != nil {
			klog.Fatalf("Failed to init repo: %v", err)
		}
	}

	var model *transformer.Model
	if *flagEmbeddings {
		model = mustRunWithElapsedTime("Loading model configurations", func() (*transformer.Model, error) {
			return transformer.LoadModel(repo)
		})
	}

	var tokenizer tokenizers.Tokenizer
	if *flagTokens || *flagEmbeddings {
		if model != nil {
			tokenizer = mustRunWithElapsedTime("Loading tokenizer", func() (tokenizers.Tokenizer, error) {
				return model.GetTokenizer()
			})
		} else {
			tokenizer = mustRunWithElapsedTime("Loading tokenizer", func() (tokenizers.Tokenizer, error) {
				return tokenizers.New(repo)
			})
		}
	}

	var backend backends.Backend
	var ctx *context.Context
	var embedExec *context.Exec

	if *flagEmbeddings {
		backend = mustRunWithElapsedTime("Initializing backend", func() (backends.Backend, error) {
			return backends.New()
		})
		ctx = context.New()

		mustRunWithElapsedTime("Loading variables into context", func() (any, error) {
			return nil, model.LoadContext(ctx)
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

		embedExec = mustRunWithElapsedTime("Compiling execution graph", func() (*context.Exec, error) {
			exec, err := context.NewExec(backend, ctx.Checked(false), func(ctx *context.Context, tokens *graph.Node) *graph.Node {
				x := model.SentenceEmbeddingGraph(ctx, tokens)
				return graph.ConvertDType(x, dtypes.Float32)
			})
			return exec, err
		})
	}

	var fOutput *os.File
	if *flagEmbeddings && *outputFile != "" {
		fOutput, err = os.Create(*outputFile)
		if err != nil {
			klog.Fatalf("Failed to create output file: %v", err)
		}
		defer fOutput.Close()
	}

	var sentencesIterator iter.Seq2[string, error]
	if *flagMSMarco {
		sentencesIterator = iterMSMarco()
	} else {
		sentencesIterator = iterArgs()
	}

	// Loop over sentences:
	count := 0
	for sentence, err := range sentencesIterator {
		if err != nil {
			klog.Fatalf("Failed to get sentence: %+v", err)
		}
		fmt.Printf("Sentence %03d: %s\n", count+1, sentence)

		var tokenIDs []int
		if *flagTokens || *flagEmbeddings {
			tokenIDs = tokenizer.EncodeWithOptions(sentence, true)
		}

		if *flagTokens {
			style := lipgloss.NewStyle().Underline(true)
			tokens := xslices.Map(tokenIDs, func(tokenID int) string {
				text := tokenizer.Decode([]int{tokenID})
				text = strings.ReplaceAll(text, " ", style.Render(" "))
				text = strings.ReplaceAll(text, " ", style.Render(" "))
				return text
			})
			fmt.Printf("\t- Tokens: \"%s\"\n", strings.Join(tokens, "\", \""))
			fmt.Printf("\t- Token IDs: %v\n", tokenIDs)
		}

		if *flagEmbeddings {
			tokenIDsInt32 := xslices.Map(tokenIDs, func(id int) int32 { return int32(id) })
			inputTensor := tensors.FromValue([][]int32{tokenIDsInt32})
			results, err := embedExec.Exec(inputTensor)
			if err != nil {
				klog.Fatalf("Failed to execute embeddings: %v", err)
			}
			outTensor := results[0]
			outTensor.ConstFlatData(func(flatAny any) {
				flat := flatAny.([]float32)
				limit := 10
				if len(flat) < limit {
					limit = len(flat)
				}
				fmt.Printf("\t- Embeddings[:%d] (out of %d): %v...\n", limit, len(flat), flat[:limit])

				if fOutput != nil {
					err := binary.Write(fOutput, binary.LittleEndian, flat)
					if err != nil {
						klog.Fatalf("Failed writing embeddings: %v", err)
					}
				}
			})
			inputTensor.FinalizeAll()
			results[0].FinalizeAll()
		}

		count++
		if *flagNumSentences > 0 && count >= *flagNumSentences {
			break
		}
	}

	fmt.Printf("\nDone.\nTotal sentences processed: %d\n", count)
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

func iterArgs() iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		for _, arg := range flag.Args() {
			if !yield(arg, nil) {
				return
			}
		}
	}
}

func iterMSMarco() iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		ds := datasets.New(msmarco.ID)
		count := 0
		limit := *flagNumSentences
		for record, err := range datasets.IterParquetFromDataset[msmarco.MsMarcoRecord](ds, msmarco.Config, *flagMSMarcoSplit) {
			if err != nil {
				yield("", err)
				return
			}
			if *flagPassages {
				if klog.V(1).Enabled() {
					klog.Infof("- Record: %+v\n", record)
					klog.Infof("\t- %d Answers, %d WellFormedAnswers, %d PassageText, %d URLs\n",
						len(record.Answers), len(record.WellFormedAnswers), len(record.Passages.PassageText), len(record.Passages.URL))
				}
				if len(record.Passages.PassageText) == 0 {
					// An empty record still counts as 1 sentence for our limit: to avoid looping over a whole dataset that doesn't have
					// any passages (or when the parsing is failing)
					count++
					if limit > 0 && count >= limit {
						return
					}
				}
				for _, passage := range record.Passages.PassageText {
					if passage == "" {
						continue
					}
					if !yield(passage, nil) {
						return
					}
					count++
					if limit > 0 && count >= limit {
						return
					}
				}
			} else {
				if !yield(record.Query, nil) {
					return
				}
				count++
				if limit > 0 && count >= limit {
					return
				}
			}
		}
	}
}
