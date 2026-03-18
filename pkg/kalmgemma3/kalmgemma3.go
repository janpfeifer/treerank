package kalmgemma3

import (
	"encoding/json"
	"fmt"
	"os"

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
	Architectures []string       `json:"architectures"`
	HiddenSize    int            `json:"hidden_size"`
	ModelType     string         `json:"model_type"`
	VocabSize     int            `json:"vocab_size"`
	Extra         map[string]any `json:"-"`
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
}

// LoadModel loads the configurations into the Model struct.
func LoadModel(repo *hub.Repo) (*Model, error) {
	m := &Model{Repo: repo}

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

		varName := context.EscapeScopeName(tensorAndName.Name)
		ctx.VariableWithValue(varName, tensorAndName.Tensor)
	}
}
