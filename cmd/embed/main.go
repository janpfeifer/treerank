package main

import (
	"flag"
	"fmt"

	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/janpfeifer/treerank/pkg/kalmgemma3"
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

	fmt.Println("Loading tokenizer...")
	tokenizer, err := kalmgemma3.LoadTokenizer(repo)
	if err != nil {
		klog.Fatalf("Failed to load tokenizer: %v", err)
	}
	_ = tokenizer // mark as used
	fmt.Println("Tokenizer loaded successfully.")

	fmt.Println("Loading model configurations...")
	model, err := kalmgemma3.LoadModel(repo)
	if err != nil {
		klog.Fatalf("Failed to load model configs: %v", err)
	}

	fmt.Println("\n=== Configurations ===")
	fmt.Println("--- Config.json ---")
	fmt.Printf("%+v\n", model.Config)

	fmt.Println("\n--- SentenceTransformerConfig ---")
	fmt.Printf("%+v\n", model.SentenceTransformerConfig)

	fmt.Println("\n--- Modules ---")
	for _, mod := range model.Modules {
		fmt.Printf("%+v\n", mod)
	}

	fmt.Println("\n--- TaskPrompts ---")
	fmt.Printf("%+v\n", model.TaskPrompts)

	fmt.Println("\n--- PoolingConfig ---")
	fmt.Printf("%+v\n", model.PoolingConfig)

	fmt.Println("\n=== Loading Variables into Context ===")
	ctx := context.New()
	model.LoadContext(ctx)

	fmt.Println("\nVariables loaded into context:")
	count := 0
	ctx.EnumerateVariables(func(v *context.Variable) {
		fmt.Printf("  %s: %s\n", v.Name(), v.Shape())
		count++
	})
	fmt.Printf("\nTotal variables loaded: %d\n", count)

	fmt.Println("\nDone.")
}
