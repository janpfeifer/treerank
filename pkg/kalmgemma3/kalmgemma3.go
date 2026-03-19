// kalmgemma3 package loads the KaLM-Gemma3 model configuration and weights.
//
// See KaLM embedding site [1] and HuggingFace model repo [2]. You'll find
// links to the papers in either these locations.
//
// [1] https://kalm-embedding.github.io/
// [2] https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511
package kalmgemma3

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/safetensors"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

const (
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

// Config represents config.json
type Config struct {
	Architectures         []string       `json:"architectures"`
	HiddenSize            int            `json:"hidden_size"`
	NumHiddenLayers       int            `json:"num_hidden_layers"`
	NumAttentionHeads     int            `json:"num_attention_heads"`
	HeadDim               int            `json:"head_dim"`
	IntermediateSize      int            `json:"intermediate_size"`
	NumKeyValueHeads      int            `json:"num_key_value_heads"`
	RMSNormEps            float64        `json:"rms_norm_eps"`
	HiddenActivation      string         `json:"hidden_activation"`
	MaxPositionEmbeddings int            `json:"max_position_embeddings"`
	ModelType             string         `json:"model_type"`
	VocabSize             int            `json:"vocab_size"`
	Extra                 map[string]any `json:"-"`
}

func (c *Config) UnmarshalJSON(data []byte) error {
	type wrapper Config
	if err := json.Unmarshal(data, (*wrapper)(c)); err != nil {
		return err
	}
	return json.Unmarshal(data, &c.Extra)
}

// SentenceTransformerConfig represents config_sentence_transformer.json
type SentenceTransformerConfig struct {
	Extra map[string]any `json:"-"`
}

func (c *SentenceTransformerConfig) UnmarshalJSON(data []byte) error {
	type wrapper SentenceTransformerConfig
	if err := json.Unmarshal(data, (*wrapper)(c)); err != nil {
		return err
	}
	return json.Unmarshal(data, &c.Extra)
}

// ModuleConfig represents an entry in modules.json
type ModuleConfig struct {
	Idx   int            `json:"idx"`
	Name  string         `json:"name"`
	Path  string         `json:"path"`
	Type  string         `json:"type"`
	Extra map[string]any `json:"-"`
}

func (m *ModuleConfig) UnmarshalJSON(data []byte) error {
	type wrapper ModuleConfig
	if err := json.Unmarshal(data, (*wrapper)(m)); err != nil {
		return err
	}
	return json.Unmarshal(data, &m.Extra)
}

// TaskPromptsConfig represents task_prompts.json
type TaskPromptsConfig struct {
	Extra map[string]any `json:"-"`
}

func (c *TaskPromptsConfig) UnmarshalJSON(data []byte) error {
	type wrapper TaskPromptsConfig
	if err := json.Unmarshal(data, (*wrapper)(c)); err != nil {
		return err
	}
	return json.Unmarshal(data, &c.Extra)
}

// PoolingConfig represents 1_Pooling/config.json
type PoolingConfig struct {
	PoolingModeMeanTokens bool           `json:"pooling_mode_mean_tokens"`
	PoolingModeClsToken   bool           `json:"pooling_mode_cls_token"`
	PoolingModeMaxTokens  bool           `json:"pooling_mode_max_tokens"`
	Extra                 map[string]any `json:"-"`
}

func (c *PoolingConfig) UnmarshalJSON(data []byte) error {
	type wrapper PoolingConfig
	if err := json.Unmarshal(data, (*wrapper)(c)); err != nil {
		return err
	}
	return json.Unmarshal(data, &c.Extra)
}

// Model holds all the configuration loaded about the model.
type Model struct {
	Repo                      *hub.Repo
	Config                    Config
	SentenceTransformerConfig SentenceTransformerConfig
	Modules                   []ModuleConfig
	TaskPrompts               TaskPromptsConfig
	PoolingConfig             PoolingConfig
	// useCausalMask: The KaLM paper says that the model is trained without a causal mask, but HuggingFace transformer
	// leaves that on by default. We default to off, but we make it configuraable.
	useCausalMask bool
}

// LoadModel loads the configurations into the Model struct.
func LoadModel(repo *hub.Repo) (*Model, error) {
	m := &Model{
		Repo: repo,
	}

	loadFile := func(filename string, v any) error {
		path, err := repo.DownloadFile(filename)
		if err != nil {
			return fmt.Errorf("failed to download %s: %w", filename, err)
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("failed to read %s: %w", filename, err)
		}
		if err := json.Unmarshal(b, v); err != nil {
			return fmt.Errorf("failed to unmarshal %s: %w", filename, err)
		}
		return nil
	}

	if err := loadFile("config.json", &m.Config); err != nil {
		return nil, err
	}
	if err := loadFile("config_sentence_transformers.json", &m.SentenceTransformerConfig); err != nil {
		return nil, err
	}
	if err := loadFile("modules.json", &m.Modules); err != nil {
		return nil, err
	}
	if err := loadFile("task_prompts.json", &m.TaskPrompts); err != nil {
		return nil, err
	}
	if err := loadFile("1_Pooling/config.json", &m.PoolingConfig); err != nil {
		return nil, err
	}

	return m, nil
}

// LoadContext uses models/safetensors to load the variables of the model to a context.
func (m *Model) LoadContext(ctx *context.Context) {
	for tensorAndName, err := range safetensors.IterTensorsFromRepo(m.Repo) {
		if err != nil {
			panic(fmt.Errorf("failed to iterate tensors: %w", err))
		}

		scopePath, varName, ok := mapTensorName(tensorAndName.Name)
		if !ok {
			fmt.Printf("Skipping unmapped tensor: %s\n", tensorAndName.Name)
			continue
		}

		tensorToLoad := tensorAndName.Tensor

		scopeCtx := ctx
		for _, scope := range scopePath {
			scopeCtx = scopeCtx.In(scope)
		}

		scopeCtx.VariableWithValue(varName, tensorToLoad)
	}
}

// mapTensorName maps safetensors tensor names to GoMLX context variable names
// KaLM-Gemma3 format: layers.{N}.{component}.weight
// GoMLX format: layer_{N}/{component}/rms_norm/scale, layer_{N}/self_attn/..., etc.
func mapTensorName(safetensorsName string) (scopePath []string, varName string, ok bool) {
	safetensorsName = strings.TrimPrefix(safetensorsName, "model.")

	if safetensorsName == "embed_tokens.weight" {
		return []string{"token_embed"}, "embeddings", true
	}
	if safetensorsName == "norm.weight" {
		return []string{"final_norm", "rms_norm"}, "scale", true
	}

	var layerNum int
	var component, sub1, sub2 string

	// Parse layer-specific components
	if n, err := fmt.Sscanf(safetensorsName, "layers.%d.%s", &layerNum, &component); n == 2 && err == nil {
		layerScope := fmt.Sprintf("layer_%d", layerNum)

		switch component {
		case "input_layernorm.weight":
			return []string{layerScope, "input_norm", "rms_norm"}, "scale", true
		case "post_attention_layernorm.weight":
			return []string{layerScope, "post_attention_norm", "rms_norm"}, "scale", true
		case "pre_feedforward_layernorm.weight":
			return []string{layerScope, "pre_feedforward_norm", "rms_norm"}, "scale", true
		case "post_feedforward_layernorm.weight":
			return []string{layerScope, "post_feedforward_norm", "rms_norm"}, "scale", true
		}
	}

	// Parse attention blocks
	if n, err := fmt.Sscanf(safetensorsName, "layers.%d.self_attn.%s", &layerNum, &sub1); n == 2 && err == nil {
		layerScope := fmt.Sprintf("layer_%d", layerNum)
		switch sub1 {
		case "q_proj.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "query", "dense"}, "weights", true
		case "k_proj.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "key", "dense"}, "weights", true
		case "v_proj.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "value", "dense"}, "weights", true
		case "o_proj.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "output", "dense"}, "weights", true
		case "q_norm.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "query", "rms_norm"}, "scale", true
		case "k_norm.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "key", "rms_norm"}, "scale", true
		}
	}

	// Parse MLP blocks
	if n, err := fmt.Sscanf(safetensorsName, "layers.%d.mlp.%s", &layerNum, &sub2); n == 2 && err == nil {
		layerScope := fmt.Sprintf("layer_%d", layerNum)
		switch sub2 {
		case "gate_proj.weight":
			return []string{layerScope, "mlp", "gate_proj", "dense"}, "weights", true
		case "up_proj.weight":
			return []string{layerScope, "mlp", "up_proj", "dense"}, "weights", true
		case "down_proj.weight":
			return []string{layerScope, "mlp", "down_proj", "dense"}, "weights", true
		}
	}

	return nil, "", false
}
