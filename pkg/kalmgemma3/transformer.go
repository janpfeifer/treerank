package kalmgemma3

import (
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

// BuildGraph takes the input tokens and creates the GoMLX graph for the embedding layer.
// It returns an array of []*graph.Node containing [embeddings, layer0, (layer1, etc...)].
func (m *Model) BuildGraph(ctx *context.Context, tokens *graph.Node) (outputs []*graph.Node) {
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
	scaleNode := graph.Scalar(g, embeddings.DType(), scale)
	embeddings = graph.Mul(embeddings, scaleNode)

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

	// Execute layer 0
	layer0 := tm.ForwardLayer(ctx.In("layer_0"), embeddings, 0, false, 0)
	outputs = append(outputs, layer0)

	// Execute layer 1
	layer1 := tm.ForwardLayer(ctx.In("layer_1"), layer0, 1, false, 0)
	outputs = append(outputs, layer1)

	return outputs
}
