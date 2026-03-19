package kalmgemma3

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"k8s.io/klog/v2"
)

func init() {
	klog.InitFlags(nil)
}

// readPythonEmbeddings reads the embeddings and first layer output dumped by the python script.
func readPythonEmbeddings(path string) ([]float32, []float32, []float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, err
	}
	defer file.Close()

	var embeddings []float32
	var layer1 []float32
	var layer2 []float32
	scanner := bufio.NewScanner(file)
	currentBlock := ""
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "#") {
			if strings.HasPrefix(line, "# Token Embeddings") {
				currentBlock = "embeddings"
			} else if strings.HasPrefix(line, "# Layer 1 Output") {
				currentBlock = "layer1"
			} else if strings.HasPrefix(line, "# Layer 2 Output") {
				currentBlock = "layer2"
			} else if strings.HasPrefix(line, "# Layer") {
				currentBlock = "skip" // We only care about embeddings and layer 1 and 2
			}
			continue
		}
		if currentBlock == "skip" || currentBlock == "" {
			continue
		}

		val, err := strconv.ParseFloat(line, 32)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to parse float: %v", err)
		}

		if currentBlock == "embeddings" {
			embeddings = append(embeddings, float32(val))
		} else if currentBlock == "layer1" {
			layer1 = append(layer1, float32(val))
		} else if currentBlock == "layer2" {
			layer2 = append(layer2, float32(val))
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, nil, nil, err
	}
	return embeddings, layer1, layer2, nil
}

func TestTransformerEmbeddings(t *testing.T) {
	// Require Python embeddings to exist
	pythonPath := "hello_world_embeddings.txt"
	expectedEmbeddings, expectedLayer1, expectedLayer2, err := readPythonEmbeddings(pythonPath)
	if err != nil {
		t.Skipf("Skipping test because %s is not available: %v", pythonPath, err)
	}

	backend, err := backends.New()
	if err != nil {
		t.Fatalf("Failed to initialize backend: %v", err)
	}
	defer backend.Finalize()

	ctx := context.New()

	repo, err := LoadRepo()
	if err != nil {
		t.Fatalf("Failed to LoadRepo: %v", err)
	}

	model, err := LoadModel(repo)
	if err != nil {
		t.Fatalf("Failed to LoadModel: %v", err)
	}

	// Set causal mask to true for testing: it makes the result closer to Python's using HF transformer library.
	model = model.WithCausalMask(true)
	// Loading weights to memory.
	fmt.Printf("Loading model ...")
	start := time.Now()
	model.LoadContext(ctx)
	fmt.Printf("done (%v)\n", time.Since(start))

	// Ensure the weights were loaded.
	varName := "embeddings"
	if ctx.In("token_embed").InspectVariableInScope(varName) == nil {
		t.Fatalf("Variable token_embed/%s not loaded in context", varName)
	}

	// We use the tokens from "Hello World" which are usually known, but let's hardcode the ones we saw in Python.
	// In the python output it was: [2, 9259, 4109]
	inputTokens := []int32{2, 9259, 4109}
	inputTensor := tensors.FromValue([][]int32{inputTokens})

	exec, err := context.NewExec(backend, ctx.Reuse(), func(ctx *context.Context, tokens *graph.Node) []*graph.Node {
		outputs := model.BuildGraph(ctx, tokens)
		var converted []*graph.Node
		for _, o := range outputs {
			converted = append(converted, graph.ConvertDType(o, dtypes.Float32))
		}
		return converted
	})
	if err != nil {
		t.Fatalf("Failed to create exec: %v", err)
	}

	results, err := exec.Exec(inputTensor)
	if err != nil {
		t.Fatalf("Failed to execute graph: %v", err)
	}

	if len(results) < 3 {
		t.Fatalf("Expected at least 3 outputs from graph, got %d", len(results))
	}

	// Validate Token Embeddings
	outTensorEmbeddings := results[0]
	outTensorEmbeddings.ConstFlatData(func(flatAny any) {
		flat := flatAny.([]float32)
		validateTensor(t, flat, outTensorEmbeddings.Shape(), inputTokens, model.Config.HiddenSize, expectedEmbeddings, "Token Embeddings")
	})

	// Validate Layer 1 Output
	outTensorLayer1 := results[1]
	outTensorLayer1.ConstFlatData(func(flatAny any) {
		flat := flatAny.([]float32)
		validateTensor(t, flat, outTensorLayer1.Shape(), inputTokens, model.Config.HiddenSize, expectedLayer1, "Layer 1 Output")
	})

	// Validate Layer 2 Output
	outTensorLayer2 := results[2]
	outTensorLayer2.ConstFlatData(func(flatAny any) {
		flat := flatAny.([]float32)
		validateTensor(t, flat, outTensorLayer2.Shape(), inputTokens, model.Config.HiddenSize, expectedLayer2, "Layer 2 Output")
	})
}

func validateTensor(t *testing.T, outData []float32, outShape shapes.Shape, inputTokens []int32, hiddenSize int, expected []float32, name string) {
	batchSize := outShape.Dimensions[1]
	if batchSize != len(inputTokens) {
		t.Fatalf("[%s] Expected %d tokens in output, got %d -- output shape is %s", name, len(inputTokens), batchSize, outShape)
	}
	if outShape.Dimensions[2] != hiddenSize {
		t.Fatalf("[%s] Expected hidden size %d, got %d -- output shape is %s", name, hiddenSize, outShape.Dimensions[2], outShape)
	}

	if len(outData) != len(expected) {
		t.Fatalf("[%s] Shape mismatch: expected %d flat floats, got %d -- output shape is %s", name, len(expected), len(outData), outShape)
	}

	var maxRelDiff float32
	var maxRelDiffIdx int
	var sumAbsDiff float64
	const minRelDenominator = 0.2
	for i, gotValue := range outData {
		expectValue := float64(expected[i])
		gotValueF64 := float64(gotValue)

		absDiff := math.Abs(gotValueF64 - expectValue)
		sumAbsDiff += absDiff

		// Avoid division by zero
		maxMag := max(math.Abs(gotValueF64), math.Abs(expectValue))
		if maxMag == 0 {
			continue
		}
		maxMag = max(maxMag, minRelDenominator)

		relDiff := float32(absDiff / maxMag)
		if relDiff > maxRelDiff {
			maxRelDiff = relDiff
			maxRelDiffIdx = i
		}
	}

	meanAbsDiff := float32(sumAbsDiff / float64(len(outData)))

	relTolerance := float32(3.0)
	meanAbsTolerance := float32(1.0)

	if maxRelDiff > relTolerance || meanAbsDiff > meanAbsTolerance {
		t.Errorf("[%s] Mismatch in values. Max rel diff: %.3g at idx %d (ex %f, got %f) -- tol %.3g. Mean abs diff: %.3g -- tol %.3g",
			name, maxRelDiff, maxRelDiffIdx, expected[maxRelDiffIdx], outData[maxRelDiffIdx], relTolerance, meanAbsDiff, meanAbsTolerance)
		for i, gotValue := range outData {
			expectValue := float64(expected[i])
			gotValueF64 := float64(gotValue)
			if i > 10 && i < len(outData)-10 {
				continue
			}
			fmt.Printf("\t- Value #%d:\tgot %.3g,\t expected %.3g\n", i, gotValueF64, expectValue)
		}
	} else {
		t.Logf("[%s] Match! Max rel diff: %.3g, Mean abs diff: %.3g", name, maxRelDiff, meanAbsDiff)
	}
}
