package main

import (
	"flag"
	"fmt"
	"iter"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/gomlx/go-huggingface/datasets"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/janpfeifer/treerank/pkg/kalmgemma3"
	"github.com/janpfeifer/treerank/pkg/msmarco"
	"k8s.io/klog/v2"
)

var (
	flagRepository = flag.String("repo", kalmgemma3.Repository, "Path to the repository")

	flagMSMarco      = flag.Bool("msmarco", false, "Pull sentences from MS MARCO dataset")
	flagMSMarcoSplit = flag.String("msmarco_split", msmarco.ValidationSplit, "Split to read from MS MARCO dataset (e.g. 'train', 'validation', 'test')")
	flagNumSentences = flag.Int("num_sentences", 10, "Number of sentences to pull from MS MARCO dataset. Set to any value <= 0 and it will use all of them.")
	flagPassages     = flag.Bool("passages", false, "Pull passages instead of queries from MS MARCO dataset")
)

func main() {
	flag.Parse()

	fmt.Println("Initializing repository:", *flagRepository)
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

	// Create context with model variables.
	/*
		ctx := context.New()
		mustRunWithElapsedTime("Loading variables into context", func() (any, error) {
			return nil, model.LoadContext(ctx)
		})
	*/

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
		tokenIDs := tokenizer.EncodeWithOptions(sentence, true)
		style := lipgloss.NewStyle().Underline(true)
		tokens := xslices.Map(tokenIDs, func(tokenID int) string {
			text := tokenizer.Decode([]int{tokenID})
			text = strings.ReplaceAll(text, " ", style.Render(" "))
			text = strings.ReplaceAll(text, " ", style.Render(" "))
			return text
		})
		fmt.Printf("\t- Tokens: \"%s\"\n", strings.Join(tokens, "\", \""))
		fmt.Printf("\t- Token IDs: %v\n", tokenIDs)
		count++
		if *flagNumSentences > 0 && count >= *flagNumSentences {
			break
		}
	}

	fmt.Println("\nDone.")
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
				fmt.Printf("- Record: %+v\n", record)
				fmt.Printf("\t- %d Answers, %d WellFormedAnswers, %d PassageText, %d URLs\n",
					len(record.Answers), len(record.WellFormedAnswers), len(record.Passages.PassageText), len(record.Passages.URL))
				count++
				if limit > 0 && count >= limit {
					return
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
