package main

import (
	"flag"
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/janpfeifer/treerank/pkg/kalmgemma3"
	"github.com/sugarme/tokenizer"
	"k8s.io/klog/v2"
)

var (
	flagRepository = flag.String("repo", kalmgemma3.Repository, "Path to the repository")
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
	tokenizer := mustRunWithElapsedTime("Loading tokenizer", func() (tokenizer.Tokenizer, error) {
		return model.GetTokenizer()
	})

	// Create context with model variables.
	ctx := context.New()
	mustRunWithElapsedTime("Loading variables into context", func() (any, error) {
		return nil, model.LoadContext(ctx)
	})

	// Loop over sentences:
	for ii, sentence := range flag.Args() {
		fmt.Printf("Sentence %d: %s\n", ii+1, sentence)
		tokenIDs := tokenizer.EncodeWithOptions(sentence, true)
		style := lipgloss.NewStyle().Underline(true)
		tokens := xslices.Map(tokenIDs, func(tokenID int) string {
			text := tokenizer.Decode([]int{tokenID})
			text = strings.ReplaceAll(text, " ", style.Render(" "))
			text = strings.ReplaceAll(text, " ", style.Render(" "))
			return text
		})
		fmt.Printf("Tokens: \"%s\"\n", strings.Join(tokens, "\", \""))
		fmt.Printf("Token IDs: %v\n", tokenIDs)
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
