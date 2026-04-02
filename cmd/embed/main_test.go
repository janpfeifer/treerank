package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestEmbed100Queries(t *testing.T) {
	tempDir := t.TempDir()

	// Append "&& exit" for the shell to exit, since the tool wants that as requested by user?
	// The user requested: "Also, always when executing any command during the process, append a `&& exit`, so Antigravity will recognize when the command finishes."
	// This usually applies to run_command tool during the agent process, not necessarily inside the Go test itself.
	// We'll write the Go test normally using exec.Command which handles termination.
	
	cmd := exec.Command("go", "run", "main.go", "-data", tempDir, "-limit", "100", "-msmarco_split", "validation")
	
	// Set the current directory to where main.go is
	cmd.Dir = "."
	
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err := cmd.Run()
	if err != nil {
		t.Fatalf("go run failed: %v", err)
	}

	// Verify the 4 exported files
	splitDir := filepath.Join(tempDir, "validation")
	
	verifySize := func(filename string, exactSize int64, minSize int64, modSize int64) {
		info, err := os.Stat(filepath.Join(splitDir, filename))
		if err != nil {
			t.Fatalf("Failed to stat %s: %v", filename, err)
		}
		size := info.Size()
		if exactSize > 0 && size != exactSize {
			t.Errorf("Expected %s to have size %d, got %d", filename, exactSize, size)
		}
		if minSize > 0 && size < minSize {
			t.Errorf("Expected %s to have min size %d, got %d", filename, minSize, size)
		}
		if modSize > 0 && size % modSize != 0 {
			t.Errorf("Expected %s size to be a multiple of %d, got %d", filename, modSize, size)
		}
	}

	// 100 queries
	// query_indices: 100 * 10 * 4 = 4000
	verifySize("query_indices.bin", 4000, 0, 0)
	// query_is_selected: 100 * 10 * 1 = 1000
	verifySize("query_is_selected.bin", 1000, 0, 0)
	
	// queries: 100 * 3840 * 4 = 1536000
	verifySize("queries.bin", 1536000, 0, 0)
	
	// passages: >0, mod 15360 (3840*4)
	verifySize("passages.bin", 0, 15360, 15360)
}
