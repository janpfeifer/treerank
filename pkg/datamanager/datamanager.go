package datamanager

import (
	"fmt"
	"os"
	"path/filepath"
	"unsafe"

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
	QueriesPassageIDs   mmap.MMap
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

	queryPassageIDsMMap, err := openMMap("queries_passage_ids.bin")
	if err != nil {
		return nil, fmt.Errorf("failed to mmap query_indices.bin: %w", err)
	}

	queryIsSelectedMMap, err := openMMap("queries_is_selected.bin")
	if err != nil {
		return nil, fmt.Errorf("failed to mmap query_is_selected.bin: %w", err)
	}

	numQueries := len(queriesMMap) / EmbeddingByteSize
	numPassages := len(passagesMMap) / EmbeddingByteSize

	return &DataManager{
		QueriesMMap:         queriesMMap,
		PassagesMMap:        passagesMMap,
		QueriesPassageIDs:   queryPassageIDsMMap,
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
	if err := d.QueriesPassageIDs.Unmap(); err != nil && firstErr == nil {
		firstErr = err
	}
	if err := d.QueryIsSelectedMMap.Unmap(); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}

// LoadQueriesPassageIDs loads the IDs (indices) of the passages for each query.
// The returned tensor has shape [NumQueries, 10].
// It returns -1 for missing passages, when a query doesn't have the 10 passages.
func (d *DataManager) LoadQueriesPassageIDs(backend backends.Backend) (*tensors.Tensor, error) {
	shape := shapes.Make(dtypes.Int32, d.NumQueries, PassagesPerQuery)
	return tensors.FromRaw(backend, 0, shape, []byte(d.QueriesPassageIDs))
}

// LoadQueriesIsSelected loads the is_selected flags for each passage of a query.
func (d *DataManager) LoadQueriesIsSelected(backend backends.Backend) (*tensors.Tensor, error) {
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

// LoadPassagesForQueries loads the passages for the queries indexed by queryIDs (indices in the full dataset).
//
// It returns two tensors:
//
//   - passageIDs: global passage IDs (indices to the whole dataset of passages) for the given queries.
//     You can feed these to LoadPassagesByIDs to get their embeddings.
//   - queriesPassageIDs: passage indices, re-mapped to indices into the returned passageIDs,
//     with -1 indicating no passage for the position, shaped [len(queryIDs), 10].
//     Note: these IDs are **not global**, but indices relative to the returned passages.
func (d *DataManager) LoadPassagesForQueries(backend backends.Backend, queryIDs ...int) (passageIDs []int, queriesPassageIDs *tensors.Tensor, err error) {
	queriesPassageIDsShape := shapes.Make(dtypes.Int32, len(queryIDs), PassagesPerQuery)
	queriesPassageIDs, err = tensors.FromShapeForBackend(backend, 0, queriesPassageIDsShape)
	passageIDs = make([]int, 0, len(queryIDs)*PassagesPerQuery) // Reserve 10 (PassagesPerQuery) per query.
	tensors.MutableFlatData(queriesPassageIDs, func(flatQueriesPassageIDs []int32) {
		flatAllQueryPassageIDs := dtypes.UnsafeSliceFromBytes[int32](unsafe.Pointer(unsafe.SliceData([]byte(d.QueriesPassageIDs))), d.NumPassages)
		for queryIdx, queryID := range queryIDs {
			globalQueryPassageIDs := flatAllQueryPassageIDs[queryID*PassagesPerQuery : (queryID+1)*PassagesPerQuery]
			for passageIdx, passageID := range globalQueryPassageIDs {
				if passageID >= 0 {
					flatQueriesPassageIDs[queryIdx*PassagesPerQuery+passageIdx] = int32(len(passageIDs)) // Index to list of passageIDs.
					passageIDs = append(passageIDs, int(passageID))
				} else {
					flatQueriesPassageIDs[queryIdx*PassagesPerQuery+passageIdx] = -1 // No passage for this position.
				}
			}
		}
	})
	return
}

// LoadIsSelectedForQueries loads the is_selected flags for the queries indexed by queryIDs (indices in the full dataset).
// It's a sub-set of what you get from LoadQueryIsSelected, but gathering only the rows for the given queryIDs.
func (d *DataManager) LoadIsSelectedForQueries(backend backends.Backend, queryIDs ...int) (queriesIsSelected *tensors.Tensor, err error) {
	queriesIsSelectedShape := shapes.Make(dtypes.Int8, len(queryIDs), PassagesPerQuery)
	queriesIsSelected, err = tensors.FromShapeForBackend(backend, 0, queriesIsSelectedShape)
	tensors.MutableFlatData(queriesIsSelected, func(flatQueriesIsSelected []int8) {
		flatAllQueryIsSelected := dtypes.UnsafeSliceFromBytes[int8](unsafe.Pointer(unsafe.SliceData([]byte(d.QueryIsSelectedMMap))), d.NumQueries)
		for queryIdx, queryID := range queryIDs {
			globalQueryIsSelected := flatAllQueryIsSelected[queryID*PassagesPerQuery : (queryID+1)*PassagesPerQuery]
			subsetQueryIsSelected := flatQueriesIsSelected[queryIdx*PassagesPerQuery : (queryIdx+1)*PassagesPerQuery]
			copy(subsetQueryIsSelected, globalQueryIsSelected)
		}
	})
	return
}
