// Package fnn implements a "treerank" model that is a feed-forward neural network (and not a tree).
//
// The model is actually a model-pair: one for queries, one for passages. Each of them is an ensemble of models
// that is grown in a boosting fashion.
package fnn

import (
	"fmt"
	"strings"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/simplego"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	mlfnn "github.com/gomlx/gomlx/pkg/ml/layers/fnn"
	"github.com/gomlx/gomlx/pkg/ml/layers/regularizers"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
)

// FNN is a feed-forward neural network model for retrieval.
//
// It classifies queries/passages into N classes (construction parameter),
// with passages being also allowed to classify to a special "out-of-retrieval" class (N+1)
//
// It is composed of two independent models: one for queries and one for passages.
// Each model is an ensemble of models that is grown in a boosting fashion.
type FNN struct {
	ctx *context.Context

	// EnsembleSize is the number of models in the ensemble.
	EnsembleSize int

	// Classes is the number of classes to map the queries/passages to:
	NumClasses int

	// FNN parameters, copied from mlfnn.Config.
	numHiddenLayers, numHiddenNodes int
	activation                      activations.Type
	normalization                   string
	dropoutRatio                    float64
	useBias, useResidual            bool
	regularizer                     regularizers.Regularizer

	// Optional configuration.
	NumLayers, NumHiddenNodes int
}

// NewFNN creates a new FNN model.
func NewFNN(ctx *context.Context, ensembleSize, numClasses int) *FNN {
	f := &FNN{
		ctx:          ctx,
		EnsembleSize: ensembleSize,
		NumClasses:   numClasses,

		numHiddenLayers: context.GetParamOr(ctx, mlfnn.ParamNumHiddenLayers, 0),
		numHiddenNodes:  context.GetParamOr(ctx, mlfnn.ParamNumHiddenNodes, 10),
		activation:      activations.FromName(context.GetParamOr(ctx, activations.ParamActivation, "relu")),
		normalization:   context.GetParamOr(ctx, mlfnn.ParamNormalization, ""),
		regularizer:     regularizers.FromContext(ctx),
		dropoutRatio:    context.GetParamOr(ctx, mlfnn.ParamDropoutRate, -1.0),
		useResidual:     context.GetParamOr(ctx, mlfnn.ParamResidual, false),
		useBias:         true,
	}
	return f
}

// NumHiddenLayers configure the number of hidden layers between the input and the output.
// Each layer will have numHiddenNodes nodes.
//
// The default is 0 (no hidden layers), but it will be overridden if the hyperparameter
// ParamNumHiddenLayers is set in the context (ctx).
// The value for numHiddenNodes can also be configured with the hyperparameter ParamNumHiddenNodes.
func (f *FNN) NumHiddenLayers(numLayers, numHiddenNodes int) *FNN {
	if numLayers < 0 || (numLayers > 0 && numHiddenNodes < 1) {
		exceptions.Panicf("fnn: numHiddenLayers (%d) must be greater or equal to 0 and numHiddenNodes (%d) must be greater or equal to 1",
			numLayers, numHiddenNodes)
	}
	f.numHiddenLayers = numLayers
	f.numHiddenNodes = numHiddenNodes
	return f
}

// UseBias configures whether to add a bias term to each node.
// Almost always you want this to be true, and that is the default.
func (f *FNN) UseBias(useBias bool) *FNN {
	f.useBias = useBias
	return f
}

// Activation sets the activation for the FNN, in between each layer.
// The input and output layers don't get an activation layer.
//
// The default is "relu", but it can be overridden by setting the hyperparameter layers.ParamActivation (="activation")
// in the context.
func (f *FNN) Activation(activation activations.Type) *FNN {
	f.activation = activation
	return f
}

// Residual configures if residual connections in between layers with the same number of nodes should be used.
// They are very useful for deep models.
//
// The default is false, and it may be configured with the hyperparameter ParamResidual.
func (f *FNN) Residual(useResidual bool) *FNN {
	f.useResidual = useResidual
	return f
}

// Normalization sets the normalization type to use in between layers.
// The input and output layers don't get a normalization layer.
//
// The default is "none", but it can be overridden by setting the hyperparameter ParamNormalization (="fnn_normalization")
// in the context.
func (f *FNN) Normalization(normalization string) *FNN {
	_, found := layers.KnownNormalizers[normalization]
	if normalization != "" && !found {
		exceptions.Panicf("fnn: unknown normalization %q given: valid values are %v or \"\"",
			normalization, xslices.SortedKeys(layers.KnownNormalizers))
	}
	f.normalization = normalization
	return f
}

// Regularizer to be applied to the learned weights (but not the biases).
// Default is none.
//
// To use more than one type of Regularizer, use regularizers.Combine, and set the returned combined regularizer here.
//
// The default is regularizers.FromContext, which is configured by regularizers.ParamL1 and regularizers.ParamL2.
func (f *FNN) Regularizer(regularizer regularizers.Regularizer) *FNN {
	f.regularizer = regularizer
	return f
}

// Dropout sets the dropout ratio for the FNN, in between each layer. The output layer doesn't get dropout.
// It uses the normalized form of dropout (see layers.DropoutNormalize).
//
// If set to 0.0, no dropout is used.
//
// The default is 0.0, but it can be overridden by setting the hyperparameter layers.ParamDropoutRate (="dropout_rate")
// in the context.
func (f *FNN) Dropout(ratio float64) *FNN {
	if ratio >= 1.0 {
		exceptions.Panicf("fnn: invalid dropout ratio %f -- set to <= 0.0 to disable it, and it must be < 1.0 otherwise everything is dropped out",
			ratio)
	}
	f.dropoutRatio = ratio
	return f
}

// QueriesLogits returns the logits for the queries.
//
// - queries: shaped [NumQueries, embeddingDim]
//
// It returns the logits for each model in the ensemble, shaped [NumQueries, EnsembleSize, NumClasses].
func (f *FNN) QueriesLogits(queries *Node) *Node {
	ctxQuery := f.ctx.In("queries")
	return mlfnn.New(ctxQuery, queries, f.EnsembleSize, f.NumClasses).
		NumHiddenLayers(f.NumLayers, f.NumHiddenNodes).
		Activation(f.activation).
		Normalization(f.normalization).
		Dropout(f.dropoutRatio).
		Residual(f.useResidual).
		UseBias(f.useBias).
		Regularizer(f.regularizer).
		Done()
}

// CopyQueriesToPassagesVariables initialize the Passages classifier weights to match those of the queries
// weights, with a shape adjustment to the last layer to account for the extra "no-retrieval" class.
//
// This can be called from either outside a model graph building, or during graph building.
// But it will fail if executed during graph building, and the queries variables it is copying from hasn't been
// initialized yet.
//
// It takes a backend which is used to shape adjust the last layer weights.
//
// TODO: it also needs to handle any normalization with learned weights ... currently
// if the shape is not converted it will likely fail.
func (f *FNN) CopyQueriesToPassagesVariables() error {
	queriesCtx := f.ctx.In("queries")
	queriesBaseScope := queriesCtx.Scope()
	passagesCtx := f.ctx.In("passages")
	passagesBaseScope := passagesCtx.Scope()
	queriesToPassageScopeFn := func(v *context.Variable) string {
		queriesScope := v.Scope()
		if !strings.HasPrefix(queriesScope, queriesBaseScope) {
			exceptions.Panicf("Queries variable %s is not in scope %q!?", v.Scope(), queriesBaseScope)
		}
		return fmt.Sprintf("%s%s", passagesBaseScope, queriesScope[len(queriesBaseScope):])
	}

	// Go backend used to extend variable values.
	var goBackend backends.Backend

	// Loop and copy over variables in "queries"
	for vQ := range queriesCtx.IterVariablesInScope() {
		scopeP := queriesToPassageScopeFn(vQ)
		name := vQ.Name()
		vP := passagesCtx.GetVariableByScopeAndName(scopeP, name)
		if vP != nil {
			// Variable already exists in passages, nothing to do.
			continue
		}
		if !vQ.HasValue() {
			exceptions.Panicf("queries variable %s has not been initialized yet, it can't be copied to a passages variable",
				vQ.ScopeAndName())
		}
		valueQ, err := vQ.Value()
		if err != nil {
			return err
		}

		scopeAndNameP := fmt.Sprintf("%s%s%s", scopeP, context.ScopeSeparator, name)
		isLastLayerWeights := strings.HasSuffix(scopeAndNameP, "fnn_output_layer/weights")
		isLastLayerBiases := strings.HasSuffix(scopeAndNameP, "fnn_output_layer/biases")
		var valueP *tensors.Tensor
		if isLastLayerWeights || isLastLayerBiases {
			// Output layer weights/biases must grow an extra "no-retrieval" class:
			// Original shape is [embeddingDim, EnsembleSize, NumClasses] for weights or [EnsemebleSize, NumClasses] for biases.
			// It needs to be extended to [embeddingDim, EnsemebleSize, NumClasses+1] or [EnsembleSize, NumClasses+1] respectively.
			shape := vQ.Shape()
			if isLastLayerWeights {
				shape.AssertDims(-1, f.EnsembleSize, f.NumClasses)
			} else {
				shape.AssertDims(f.EnsembleSize, f.NumClasses)
			}
			shapeP := shape.Clone()
			shapeP.Dimensions[len(shapeP.Dimensions)-1]++
			extendValueFn := func(varQ *Node) *Node {
				sliceShape := varQ.Shape().Clone()
				sliceShape.Dimensions[len(sliceShape.Dimensions)-1] = 1
				slice := Zeros(varQ.Graph(), sliceShape)
				return Concatenate([]*Node{varQ, slice}, -1)
			}
			if valueQ.IsOnAnyDevice() {
				// Execute on the backend already used by the queries variable.
				backend, err := valueQ.Backend()
				if err != nil {
					return err
				}
				valueP, err = ExecOnce(backend, extendValueFn, valueQ)
				if err != nil {
					return errors.WithMessagef(err, "while extending variable %s to new shape %s", vQ.ScopeAndName(), shapeP)
				}
			} else {
				// Execute on the Go backend.
				if goBackend == nil {
					goBackend, err = simplego.New("")
					if err != nil {
						return err
					}
				}
				valueP, err = ExecOnce(goBackend, extendValueFn, valueQ)
				if err != nil {
					return errors.WithMessagef(err, "while extending variable %s to new shape %s", vQ.ScopeAndName(), shapeP)
				}
				err = valueP.ToLocal()
				if err != nil {
					return errors.WithMessagef(err, "failed to move tensor locally for variable %s", vQ.ScopeAndName())
				}
			}

		} else {
			// Simply copy over the value of the queries variable.
			valueP, err = valueQ.Clone()
			if err != nil {
				return err
			}
		}

		// With the value for the passages variable, we can create the variable in the passages context.
		absCtx := passagesCtx.InAbsPath(scopeP)
		_ = absCtx.VariableWithValue(name, valueP)
	}
	return nil
}

// PassagesLogits returns the logits for the passages.
//
// - passages: shaped [NumPassages, embeddingDim]
//
// It returns the logits for each model in the ensemble, shaped [NumPassages, EnsembleSize, NumClasses+1].
// The last class is a special "no-retrieval" class.
func (f *FNN) PassagesLogits(passages *Node) *Node {
	// CopyQueriesToPassagesVariables: it's a no-op if the passages variables already exist
	// or if the queries variables doesn't yet exist.
	if err := f.CopyQueriesToPassagesVariables(); err != nil {
		panic(err)
	}
	ctxPassage := f.ctx.In("passages")
	return mlfnn.New(ctxPassage, passages, f.EnsembleSize, f.NumClasses+1).
		NumHiddenLayers(f.NumLayers, f.NumHiddenNodes).
		Activation(f.activation).
		Normalization(f.normalization).
		Dropout(f.dropoutRatio).
		Residual(f.useResidual).
		UseBias(f.useBias).
		Regularizer(f.regularizer).
		Done()
}

// Train trains the FNN model.
func (f *FNN) Train(queries, passages, queriesIsSelected, queriesPassageIDs *tensors.Tensor) error {
	return nil
}

// Predict predicts the relevance of queries and passages.
func (f *FNN) Predict(queries, passages *tensors.Tensor) (*tensors.Tensor, error) {
	return nil, nil
}
