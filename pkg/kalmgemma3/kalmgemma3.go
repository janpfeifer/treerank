// kalmgemma3 package loads the KaLM-Gemma3 model configuration and weights.
//
// See KaLM embedding site [1] and HuggingFace model repo [2]. You'll find
// links to the papers in either these locations.
//
// [1] https://kalm-embedding.github.io/
// [2] https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511
package kalmgemma3

import (
	"fmt"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"
)

const (
	// Repository for KaLM-Embedding-Gemma3-12B-2511.
	//
	// This is an embedding model, not a language model. It is trained to produce
	// embeddings for sentences, not to generate text.
	//
	// Remember to set Model.WithCausalMask(false) when using this model.
	// (The default is set to use causal mask, but the model shouldn't use it)
	Repository = "tencent/KaLM-Embedding-Gemma3-12B-2511"
)

// LoadRepo creates a hub.Repo that can be used to download tokenizer, configuration files and model files.
func LoadRepo() (*hub.Repo, error) {
	repo := hub.New(Repository)
	if err := repo.DownloadInfo(false); err != nil {
		return nil, fmt.Errorf("failed to get repo info: %w", err)
	}
	return repo, nil
}

// LoadTokenizer creates a Tokenizer from the given repository.
func LoadTokenizer(repo *hub.Repo) (tokenizers.Tokenizer, error) {
	tokenizer, err := tokenizers.New(repo)
	if err != nil {
		return nil, fmt.Errorf("failed to create tokenizer: %w", err)
	}
	return tokenizer, nil
}
