package kalmgemma3

import (
	"bufio"
	"flag"
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

var flagUseCausalMask = flag.Bool("use_causal_mask", true, "Use causal mask in the transformer: the paper suggests one shouldn't, "+
	"but for testing it makes the result closer to Python's using HF transformer library, which seems to use it.")

var (
	testBackend backends.Backend
	testCtx     *context.Context
	testModel   *Model
)

func TestMain(m *testing.M) {
	klog.InitFlags(nil)
	flag.Parse() // Ensure flags are parsed before we use them

	var err error
	testBackend, err = backends.New()
	if err != nil {
		fmt.Printf("Failed to initialize backend: %v\n", err)
		os.Exit(1)
	}

	testCtx = context.New()

	repo, err := LoadRepo()
	if err != nil {
		fmt.Printf("Failed to LoadRepo: %v\n", err)
		os.Exit(1)
	}

	testModel, err = LoadModel(repo)
	if err != nil {
		fmt.Printf("Failed to LoadModel: %v\n", err)
		os.Exit(1)
	}
	testModel = testModel.WithCausalMask(*flagUseCausalMask)

	fmt.Printf("- Loading model weights ...")
	start := time.Now()
	testModel.LoadContext(testCtx)
	fmt.Printf("done (%v)\n", time.Since(start))

	code := m.Run()

	testBackend.Finalize()
	os.Exit(code)
}

// readPythonEmbeddings reads the embeddings and layers dumped by the python script.
func readPythonEmbeddings(path string, layersToRead []int) (map[int][]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	wantLayers := make(map[int]bool)
	for _, l := range layersToRead {
		wantLayers[l] = true
	}

	results := make(map[int][]float32)
	scanner := bufio.NewScanner(file)
	currentLayer := -1
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "#") {
			if strings.HasPrefix(line, "# Token Embeddings") {
				currentLayer = 0
			} else if strings.HasPrefix(line, "# Layer ") {
				var l int
				if _, err := fmt.Sscanf(line, "# Layer %d Output", &l); err == nil {
					currentLayer = l
				} else {
					currentLayer = -1
				}
			} else {
				currentLayer = -1
			}
			continue
		}
		if currentLayer == -1 || !wantLayers[currentLayer] {
			continue
		}

		val, err := strconv.ParseFloat(line, 32)
		if err != nil {
			return nil, fmt.Errorf("failed to parse float: %v", err)
		}

		results[currentLayer] = append(results[currentLayer], float32(val))
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return results, nil
}

func TestTransformerEmbeddings(t *testing.T) {

	// Ensure the weights were loaded.
	varName := "embeddings"
	if testCtx.In("token_embed").InspectVariableInScope(varName) == nil {
		t.Fatalf("Variable token_embed/%s not loaded in context", varName)
	}

	// We use the tokens from "Hello World" which are usually known, but let's hardcode the ones we saw in Python.
	// In the python output it was: [2, 9259, 4109]
	inputTokens := []int32{2, 9259, 4109}
	inputTensor := tensors.FromValue([][]int32{inputTokens})

	layersToCheck := []int{0, 1, 2, 10, 20, 30, 40, testModel.Config.NumHiddenLayers}

	uniqueLayers := make(map[int]bool)
	var finalLayersToCheck []int
	for _, l := range layersToCheck {
		if l <= testModel.Config.NumHiddenLayers && !uniqueLayers[l] {
			uniqueLayers[l] = true
			finalLayersToCheck = append(finalLayersToCheck, l)
		}
	}

	// Require Python embeddings to exist
	pythonPath := "hello_world_embeddings.txt"
	expectedLayers, err := readPythonEmbeddings(pythonPath, finalLayersToCheck)
	if err != nil {
		t.Skipf("Skipping test because %s is not available: %v", pythonPath, err)
	}

	// Ensure that we successfully parsed the required layers.
	for _, l := range finalLayersToCheck {
		if len(expectedLayers[l]) == 0 {
			t.Fatalf("Failed to read expected data for layer %d from %s", l, pythonPath)
		}
	}

	exec, err := context.NewExec(testBackend, testCtx.Reuse(), func(ctx *context.Context, tokens *graph.Node) []*graph.Node {
		outputs := testModel.BuildAllLayersGraph(ctx, tokens)
		var converted []*graph.Node
		for _, o := range outputs {
			converted = append(converted, graph.ConvertDType(o, dtypes.Float32))
		}
		return converted
	})
	if err != nil {
		t.Fatalf("Failed to create exec: %v", err)
	}

	fmt.Printf("- Pre-compiling model ...")
	start := time.Now()
	_, err = exec.Exec(inputTensor)
	if err != nil {
		t.Fatalf("Failed to compile graph: %v", err)
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	fmt.Printf("- Executing model ...")
	start = time.Now()
	results, err := exec.Exec(inputTensor)
	if err != nil {
		t.Fatalf("Failed to execute graph: %v", err)
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	if len(results) < testModel.Config.NumHiddenLayers+1 {
		t.Fatalf("Expected at least %d outputs from graph, got %d", testModel.Config.NumHiddenLayers+1, len(results))
	}

	for _, l := range finalLayersToCheck {
		outTensor := results[l]
		expectedData := expectedLayers[l]
		name := fmt.Sprintf("Layer %d Output", l)
		if l == 0 {
			name = "Token Embeddings"
		}
		outTensor.ConstFlatData(func(flatAny any) {
			flat := flatAny.([]float32)
			validateTensor(t, flat, outTensor.Shape(), inputTokens, testModel.Config.HiddenSize, expectedData, name)
		})
	}
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

	var sumAbsDiff, sumAbsExpected float64
	const minRelDenominator = 0.2
	for i, gotValue := range outData {
		expectValue := float64(expected[i])
		gotValueF64 := float64(gotValue)

		absDiff := math.Abs(gotValueF64 - expectValue)
		sumAbsDiff += absDiff
		sumAbsExpected += math.Abs(expectValue)
	}
	meanAbsDiff := sumAbsDiff / float64(len(outData))
	meanAbsExpected := sumAbsExpected / float64(len(outData))

	var maxRelDiff float64
	var maxRelDiffIdx int
	for i, gotValue := range outData {
		expectValue := float64(expected[i])
		gotValueF64 := float64(gotValue)
		absDiff := math.Abs(gotValueF64 - expectValue)
		relDenominator := math.Max(math.Abs(expectValue), math.Abs(gotValueF64))
		relDenominator = max(relDenominator, meanAbsExpected)
		relDiff := absDiff / relDenominator
		if relDiff > maxRelDiff {
			maxRelDiff = relDiff
			maxRelDiffIdx = i
		}
	}

	relTolerance := 5.0
	meanAbsTolerance := 10.0

	if maxRelDiff > relTolerance || meanAbsDiff > meanAbsTolerance {
		t.Errorf("[%s] Mismatch in values. Max rel diff: %.3g at idx %d (ex %f, got %f) -- tol %.3g. Mean abs diff: %.3g (%.3g is the mean absolute) -- tol %.3g",
			name, maxRelDiff, maxRelDiffIdx, expected[maxRelDiffIdx], outData[maxRelDiffIdx], relTolerance, meanAbsDiff, meanAbsExpected, meanAbsTolerance)
		for i, gotValue := range outData {
			expectValue := float64(expected[i])
			gotValueF64 := float64(gotValue)
			if i > 10 && i < len(outData)-10 {
				continue
			}
			fmt.Printf("\t- Value #%d:\tgot %.3g,\t expected %.3g\n", i, gotValueF64, expectValue)
		}
	} else {
		t.Logf("[%s] Match! Max rel diff: %.3g, Mean abs diff: %.3g (%.3g is the mean absolute)", name, maxRelDiff, meanAbsDiff, meanAbsExpected)
	}
}

func readPythonEmbeddingsList(path string) ([]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var results []float32
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		val, err := strconv.ParseFloat(line, 32)
		if err != nil {
			return nil, fmt.Errorf("failed to parse float: %v", err)
		}
		results = append(results, float32(val))
	}
	return results, scanner.Err()
}

func TestSentenceEmbedding(t *testing.T) {
	inputTokens := []int32{2, 9259, 4109}
	inputTensor := tensors.FromValue([][]int32{inputTokens})

	pythonPath := "hello_world_sentence_embed.txt"
	expectedData, err := readPythonEmbeddingsList(pythonPath)
	if err != nil {
		t.Skipf("Skipping test because %s is not available: %v", pythonPath, err)
	}

	exec, err := context.NewExec(testBackend, testCtx.Reuse(), func(ctx *context.Context, tokens *graph.Node) *graph.Node {
		x := testModel.SentenceEmbeddingGraph(ctx, tokens)
		return graph.ConvertDType(x, dtypes.Float32)
	})
	if err != nil {
		t.Fatalf("Failed to create exec: %v", err)
	}

	fmt.Printf("- Pre-compiling model ...")
	start := time.Now()
	_, err = exec.Exec(inputTensor)
	if err != nil {
		t.Fatalf("Failed to compile graph: %v", err)
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	fmt.Printf("- Executing model ...")
	start = time.Now()
	results, err := exec.Exec(inputTensor)
	if err != nil {
		t.Fatalf("Failed to execute graph: %v", err)
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	outTensor := results[0]
	outShape := outTensor.Shape()
	if outShape.Rank() != 2 || outShape.Dimensions[1] != testModel.Config.HiddenSize {
		t.Fatalf("Expected shape [batch, hidden_size] ([1, %d]), got %s", testModel.Config.HiddenSize, outShape)
	}

	outTensor.ConstFlatData(func(flatAny any) {
		flat := flatAny.([]float32)
		validateFlatTensor(t, flat, expectedData, "Sentence Embedding")
	})
}

func validateFlatTensor(t *testing.T, gotData, expectedData []float32, name string) {
	if len(gotData) != len(expectedData) {
		t.Fatalf("[%s] Shape mismatch: expected %d flat floats, got %d", name, len(expectedData), len(gotData))
	}

	var sumAbsDiff, sumAbsExpected float64
	var maxRelDiff float64
	var maxRelDiffIdx int
	for i, gotValue := range gotData {
		expectValue := float64(expectedData[i])
		gotValueF64 := float64(gotValue)
		absDiff := math.Abs(gotValueF64 - expectValue)
		sumAbsDiff += absDiff
		sumAbsExpected += math.Abs(expectValue)
	}
	meanAbsDiff := sumAbsDiff / float64(len(gotData))
	meanAbsExpected := sumAbsExpected / float64(len(gotData))

	for i, gotValue := range gotData {
		expectValue := float64(expectedData[i])
		gotValueF64 := float64(gotValue)
		absDiff := math.Abs(gotValueF64 - expectValue)
		relDenominator := math.Max(math.Abs(expectValue), math.Abs(gotValueF64))
		relDenominator = max(relDenominator, meanAbsExpected)
		relDiff := absDiff / relDenominator
		if relDiff > maxRelDiff {
			maxRelDiff = relDiff
			maxRelDiffIdx = i
		}
	}

	relTolerance := 5.0
	meanAbsTolerance := 10.0

	if maxRelDiff > relTolerance || meanAbsDiff > meanAbsTolerance {
		t.Errorf("[%s] Mismatch in values. Max rel diff: %.3g at idx %d (ex %f, got %f) -- tol %.3g. Mean abs diff: %.3g (%.3g is the mean absolute) -- tol %.3g",
			name, maxRelDiff, maxRelDiffIdx, expectedData[maxRelDiffIdx], gotData[maxRelDiffIdx], relTolerance, meanAbsDiff, meanAbsExpected, meanAbsTolerance)
	} else {
		t.Logf("[%s] Match! Max rel diff: %.3g, Mean abs diff: %.3g (%.3g is the mean absolute)", name, maxRelDiff, meanAbsDiff, meanAbsExpected)
	}
}
