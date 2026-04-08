package datamanager

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/edsrzf/mmap-go"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

const (
	EmbeddingSize     = 3840
	PassagesPerQuery  = 10
	EmbeddingByteSize = EmbeddingSize * 4 // float32 is 4 bytes
)

// DataManager manages memory-mapped data files for the dataset.
type DataManager struct {
	QueriesMMap         mmap.MMap
	PassagesMMap        mmap.MMap
	QueryIndicesMMap    mmap.MMap
	QueryIsSelectedMMap mmap.MMap

	NumQueries  int
	NumPassages int
}

// New opens a memory map of each of the data files and returns a DataManager.
func New(dataPath string) (*DataManager, error) {
	openMMap := func(filename string) (mmap.MMap, error) {
		f, err := os.Open(filepath.Join(dataPath, filename))
		if err != nil {
			return nil, err
		}
		defer f.Close() // mmap.Map keeps a reference to the file mapping

		// Open the file as read-only memory map
		mapped, err := mmap.Map(f, mmap.RDONLY, 0)
		if err != nil {
			return nil, err
		}
		return mapped, nil
	}

	queriesMMap, err := openMMap("queries.bin")
	if err != nil {
		return nil, fmt.Errorf("failed to mmap queries.bin: %w", err)
	}

	passagesMMap, err := openMMap("passages.bin")
	if err != nil {
		return nil, fmt.Errorf("failed to mmap passages.bin: %w", err)
	}

	queryIndicesMMap, err := openMMap("query_indices.bin")
	if err != nil {
		return nil, fmt.Errorf("failed to mmap query_indices.bin: %w", err)
	}

	queryIsSelectedMMap, err := openMMap("query_is_selected.bin")
	if err != nil {
		return nil, fmt.Errorf("failed to mmap query_is_selected.bin: %w", err)
	}

	numQueries := len(queriesMMap) / EmbeddingByteSize
	numPassages := len(passagesMMap) / EmbeddingByteSize

	return &DataManager{
		QueriesMMap:         queriesMMap,
		PassagesMMap:        passagesMMap,
		QueryIndicesMMap:    queryIndicesMMap,
		QueryIsSelectedMMap: queryIsSelectedMMap,
		NumQueries:          numQueries,
		NumPassages:         numPassages,
	}, nil
}

// Close unmaps all the underlying memory maps.
func (d *DataManager) Close() error {
	var firstErr error
	if err := d.QueriesMMap.Unmap(); err != nil && firstErr == nil {
		firstErr = err
	}
	if err := d.PassagesMMap.Unmap(); err != nil && firstErr == nil {
		firstErr = err
	}
	if err := d.QueryIndicesMMap.Unmap(); err != nil && firstErr == nil {
		firstErr = err
	}
	if err := d.QueryIsSelectedMMap.Unmap(); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}

// LoadQueryIndices loads the indices of the passages for each query.
func (d *DataManager) LoadQueryIndices(backend backends.Backend) (*tensors.Tensor, error) {
	shape := shapes.Make(dtypes.Int32, d.NumQueries, PassagesPerQuery)
	return tensors.FromRaw(backend, 0, shape, []byte(d.QueryIndicesMMap))
}

// LoadQueryIsSelected loads the is_selected flags for each passage of a query.
func (d *DataManager) LoadQueryIsSelected(backend backends.Backend) (*tensors.Tensor, error) {
	shape := shapes.Make(dtypes.Int8, d.NumQueries, PassagesPerQuery)
	return tensors.FromRaw(backend, 0, shape, []byte(d.QueryIsSelectedMMap))
}

// LoadQuery loads one query identified by queryID into a tensor.
// The returned tensor will have shape [EmbeddingSize].
func (d *DataManager) LoadQuery(backend backends.Backend, queryID int) (*tensors.Tensor, error) {
	if queryID < 0 || queryID >= d.NumQueries {
		return nil, fmt.Errorf("queryID %d out of bounds [0, %d)", queryID, d.NumQueries)
	}
	start := queryID * EmbeddingByteSize
	end := start + EmbeddingByteSize
	shape := shapes.Make(dtypes.Float32, EmbeddingSize)
	return tensors.FromRaw(backend, 0, shape, []byte(d.QueriesMMap)[start:end])
}

// LoadPassage loads one passage identified by passageID.
// The returned tensor will have shape [EmbeddingSize].
func (d *DataManager) LoadPassage(backend backends.Backend, passageID int) (*tensors.Tensor, error) {
	if passageID < 0 || passageID >= d.NumPassages {
		return nil, fmt.Errorf("passageID %d out of bounds [0, %d)", passageID, d.NumPassages)
	}
	start := passageID * EmbeddingByteSize
	end := start + EmbeddingByteSize
	shape := shapes.Make(dtypes.Float32, EmbeddingSize)
	return tensors.FromRaw(backend, 0, shape, []byte(d.PassagesMMap)[start:end])
}

// LoadAllQueries loads all the queries into one large tensor shaped [NumQueries, EmbeddingSize].
func (d *DataManager) LoadAllQueries(backend backends.Backend) (*tensors.Tensor, error) {
	shape := shapes.Make(dtypes.Float32, d.NumQueries, EmbeddingSize)
	return tensors.FromRaw(backend, 0, shape, []byte(d.QueriesMMap))
}

// LoadQueriesByIDs loads queries identified by queryIDs into a tensor.
// The returned tensor will have shape [len(queryIDs), EmbeddingSize].
//
// For efficiency, sort the queryIDs before calling this.
func (d *DataManager) LoadQueriesByIDs(backend backends.Backend, queryIDs ...int) (*tensors.Tensor, error) {
	shape := shapes.Make(dtypes.Float32, len(queryIDs), EmbeddingSize)
	t, err := tensors.FromShapeForBackend(backend, 0, shape)
	if err != nil {
		return nil, err
	}
	t.MutableBytes(func(tensorBytes []byte) {
		for i, queryID := range queryIDs {
			if queryID < 0 || queryID >= d.NumQueries {
				err = errors.Errorf("queryID %d (at index %d) out of bounds [0, %d)", queryID, i, d.NumQueries)
				return
			}
			dsStart := queryID * EmbeddingByteSize
			dsEnd := dsStart + EmbeddingByteSize
			tStart := i * EmbeddingByteSize
			tEnd := tStart + EmbeddingByteSize
			copy(tensorBytes[tStart:tEnd], []byte(d.QueriesMMap)[dsStart:dsEnd])
		}
	})
	if err != nil {
		t.FinalizeAll()
		return nil, err
	}
	return t, nil
}

// LoadPassagesByIDs loads passages identified by passageIDs into a tensor.
// The returned tensor will have shape [len(passageIDs), EmbeddingSize].
//
// For efficiency, sort the passageIDs before calling this.
func (d *DataManager) LoadPassagesByIDs(backend backends.Backend, passageIDs ...int) (*tensors.Tensor, error) {
	shape := shapes.Make(dtypes.Float32, len(passageIDs), EmbeddingSize)
	t, err := tensors.FromShapeForBackend(backend, 0, shape)
	if err != nil {
		return nil, err
	}
	t.MutableBytes(func(tensorBytes []byte) {
		for i, passageID := range passageIDs {
			if passageID < 0 || passageID >= d.NumPassages {
				err = errors.Errorf("passageID %d (at index %d) out of bounds [0, %d)", passageID, i, d.NumPassages)
				return
			}
			dsStart := passageID * EmbeddingByteSize
			dsEnd := dsStart + EmbeddingByteSize
			tStart := i * EmbeddingByteSize
			tEnd := tStart + EmbeddingByteSize
			copy(tensorBytes[tStart:tEnd], []byte(d.PassagesMMap)[dsStart:dsEnd])
		}
	})
	if err != nil {
		t.FinalizeAll()
		return nil, err
	}
	return t, nil
}

// LoadAllPassages loads all the passages into one large tensor shaped [NumPassages, EmbeddingSize].
func (d *DataManager) LoadAllPassages(backend backends.Backend) (*tensors.Tensor, error) {
	shape := shapes.Make(dtypes.Float32, d.NumPassages, EmbeddingSize)
	return tensors.FromRaw(backend, 0, shape, []byte(d.PassagesMMap))
}

// LoadPassagesBatch loads a batch of passages int a tensor shaped [batchSize, EmbeddingSize].
func (d *DataManager) LoadPassagesBatch(backend backends.Backend, firstPassageID, batchSize int) (*tensors.Tensor, error) {
	if firstPassageID < 0 || firstPassageID+batchSize > d.NumPassages {
		return nil, fmt.Errorf("passage batch [%d, %d) out of bounds [0, %d)", firstPassageID, firstPassageID+batchSize, d.NumPassages)
	}
	start := firstPassageID * EmbeddingByteSize
	end := (firstPassageID + batchSize) * EmbeddingByteSize
	shape := shapes.Make(dtypes.Float32, batchSize, EmbeddingSize)
	return tensors.FromRaw(backend, 0, shape, []byte(d.PassagesMMap)[start:end])
}
