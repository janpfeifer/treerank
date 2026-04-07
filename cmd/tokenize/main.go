package main

import (
	"flag"
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/gomlx/go-huggingface/examples/kalmgemma3"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"k8s.io/klog/v2"
)

var (
	flagRepository = flag.String("repo", kalmgemma3.Repository, "Path to the repository")
	flagTask       = flag.String("task", "", "Task selection (for queries), it adds a prompt accordingly. "+
		"If empty no prompt is prepended. "+
		"Set to '?' or 'list' to list supported values.")
)

func main() {
	flag.Parse()

	repo := hub.New(*flagRepository).WithProgressBar(true)
	if err := repo.DownloadInfo(false); err != nil {
		klog.Fatalf("Failed to get repo info: %+v", err)
	}

	var taskPrompts kalmgemma3.TaskPrompts
	if *flagTask != "" {
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
			return
		}
		if _, ok := taskPrompts[*flagTask]; !ok {
			klog.Fatalf("Unknown task prompt key: %s", *flagTask)
		}
	}

	// Load tokenizer.
	tokenizer, err := tokenizers.New(repo)
	if err != nil {
		klog.Fatalf("Failed to create tokenizer: %+v", err)
	}

	// Loop over sentences:
	for ii, sentence := range flag.Args() {
		if *flagTask != "" {
			sentence = taskPrompts.BuildQueryPrompt(sentence, *flagTask)
		}
		fmt.Printf("Sentence %d: %s\n", ii+1, sentence)
		tokenIDs := tokenizer.Encode(sentence)
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
