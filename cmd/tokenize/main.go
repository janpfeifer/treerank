package main

import (
	"flag"
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/janpfeifer/treerank/pkg/kalmgemma3"
	"k8s.io/klog/v2"
)

var (
	flagRepository = flag.String("repo", kalmgemma3.Repository, "Path to the repository")
)

func main() {
	flag.Parse()

	repo := hub.New(*flagRepository).WithProgressBar(true)
	if err := repo.DownloadInfo(false); err != nil {
		klog.Fatalf("Failed to get repo info: %+v", err)
	}

	// Load tokenizer.
	tokenizer, err := tokenizers.New(repo)
	if err != nil {
		klog.Fatalf("Failed to create tokenizer: %+v", err)
	}

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
}
