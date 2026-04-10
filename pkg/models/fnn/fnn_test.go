package fnn

import (
	"strings"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestCopyQueriesToPassagesVariables(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()

	embedDim := 10
	ensembleSize := 3
	numClasses := 5
	model := NewFNN(ctx, ensembleSize, numClasses).
		NumHiddenLayers(2, 4)

	batchSize := 2
	queriesShape := shapes.Make(dtypes.Float32, batchSize, embedDim)
	queriesTensor := tensors.FromShape(queriesShape)

	// Execute model.QueriesLogits to initialize queries variables
	logitsTensor, err := context.ExecOnce(backend, ctx, func(ctx *context.Context, queries *Node) *Node {
		return model.QueriesLogits(queries)
	}, queriesTensor)
	require.NoError(t, err)
	require.NoErrorf(t, logitsTensor.Shape().CheckDims(batchSize, ensembleSize, numClasses),
		"expected logits shape [%d, %d, %d] to match, got %s", batchSize, ensembleSize, numClasses, logitsTensor.Shape())

	// Queries variables must have been created and initialized
	queriesCtx := ctx.In("queries")
	numQueriesVars := 0
	for range queriesCtx.IterVariablesInScope() {
		numQueriesVars++
	}
	require.Greater(t, numQueriesVars, 0)

	// Execute CopyQueriesToPassagesVariables
	err = model.CopyQueriesToPassagesVariables()
	require.NoError(t, err)

	// Check that the variables in the "passages" sub-context are correctly copied / extended
	passagesCtx := ctx.In("passages")

	for vQ := range queriesCtx.IterVariablesInScope() {
		scopeP := passagesCtx.Scope() + vQ.Scope()[len(queriesCtx.Scope()):]
		name := vQ.Name()

		vP := passagesCtx.GetVariableByScopeAndName(scopeP, name)
		require.NotNilf(t, vP, "variable %s/%s not found in passages", scopeP, name)

		shapeQ := vQ.Shape()
		shapeP := vP.Shape()

		scopeAndNameP := scopeP + context.ScopeSeparator + name
		isLastLayerWeights := strings.HasSuffix(scopeAndNameP, "fnn_output_layer/weights")
		isLastLayerBiases := strings.HasSuffix(scopeAndNameP, "fnn_output_layer/biases")

		if isLastLayerWeights || isLastLayerBiases {
			expectedShapeP := shapeQ.Clone()
			expectedShapeP.Dimensions[len(expectedShapeP.Dimensions)-1]++
			assert.True(t, expectedShapeP.Equal(shapeP), "shape mismatch for %s", scopeAndNameP)
		} else {
			assert.True(t, shapeQ.Equal(shapeP), "shape mismatch for %s", scopeAndNameP)
		}
	}
}
