package kalmgemma3

import (
	"fmt"
	"math"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/model/transformer"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
)

// WithCausalMask sets whether to use a causal mask in the attention layers.
//
// The KaLM paper says that the model is trained without a causal mask, but HuggingFace transformer
// leaves that on by default. We default to off, but we make it configurable.
func (m *Model) WithCausalMask(useCausalMask bool) *Model {
	m.useCausalMask = useCausalMask
	return m
}

// BuildAllLayersGraph takes the input tokens and creates the GoMLX graph for the embedding layer.
// It returns an array of []*graph.Node containing [embeddings, layer0, (layer1, etc...)].
func (m *Model) BuildAllLayersGraph(ctx *context.Context, tokens *graph.Node) (outputs []*graph.Node) {
	g := tokens.Graph()

	// The embedding weight name mapped from the HuggingFace safetensors
	varName := "embeddings"

	// Check if the variable exists in the context
	weightVar := ctx.In("token_embed").InspectVariableInScope(varName)
	if weightVar == nil {
		exceptions.Panicf("failed to find embedding weights in context: token_embed/%s. did you call LoadContext?", varName)
	}

	weightNode := weightVar.ValueGraph(g)

	// Gather embeddings for the given tokens.
	// tokens shape: [batch, seq_len] -> expanded to [batch, seq_len, 1] for Gather
	// weightNode shape: [vocab_size, hidden_size]
	// Gather output shape: [batch, seq_len, hidden_size]
	embeddings := graph.Gather(weightNode, graph.ExpandAxes(tokens, -1))

	// Gemma3 scales the embeddings by sqrt(hidden_size).
	scale := math.Sqrt(float64(m.Config.HiddenSize))

	// Cast scale to the exact dtype of the embeddings
	embeddings = graph.MulScalar(embeddings, scale)

	// Output index 0 is token embeddings
	outputs = append(outputs, embeddings)

	// Initialize the base transformer.Model configuration using the loaded fields.
	headDim := m.Config.HeadDim
	if headDim == 0 {
		headDim = m.Config.HiddenSize / m.Config.NumAttentionHeads
	}
	tm := transformer.New(
		m.Config.VocabSize,
		m.Config.HiddenSize,
		m.Config.NumHiddenLayers,
		m.Config.NumAttentionHeads,
		headDim,
	).
		WithFFNDim(m.Config.IntermediateSize).
		WithMaxPosEmbed(m.Config.MaxPositionEmbeddings).
		WithArchitecture(transformer.ArchitectureGemma3).
		WithTransposedWeights(true).
		WithNormalization(layers.NormalizationRMSNorm).
		WithNormEpsilon(m.Config.RMSNormEps).
		WithActivation(activations.FromName(m.Config.HiddenActivation)).
		WithNumKVHeads(m.Config.NumKeyValueHeads).
		WithBias(false).
		WithCausalMask(m.useCausalMask)

	x := embeddings
	for layerIdx := 0; layerIdx < m.Config.NumHiddenLayers; layerIdx++ {
		layerCtx := ctx.In(fmt.Sprintf("layer_%d", layerIdx))
		x = tm.ForwardLayer(layerCtx, x, layerIdx, false, 0)

		// For the very last layer, huggingface returns the normalized hidden state in hidden_states[-1]
		if layerIdx == m.Config.NumHiddenLayers-1 {
			x = layers.RMSNorm(ctx.In("final_norm"), x).WithEpsilon(m.Config.RMSNormEps).WithScaleOffset(1.0).Done()
		}

		outputs = append(outputs, x)
	}

	return outputs
}

// BuildGraph takes the input tokens and creates the GoMLX graph for the model.
// It returns the final normalized output layer.
func (m *Model) BuildGraph(ctx *context.Context, tokens *graph.Node) *graph.Node {
	outputs := m.BuildAllLayersGraph(ctx, tokens)
	return outputs[len(outputs)-1]
}
