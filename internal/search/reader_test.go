package search

import (
	"os"
	"testing"
)

func TestOpen_MissingFile(t *testing.T) {
	_, err := Open("/nonexistent/path.bin")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestOpen_RealIndex(t *testing.T) {
	path := "../../index/index_p0.bin"
	if _, err := os.Stat(path); err != nil {
		t.Skip("index not built yet")
	}
	ix, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ix.Close()
	if ix.NClusters < 1 {
		t.Fatal("zero clusters")
	}
	if ix.NVectors < 1 {
		t.Fatal("zero vectors")
	}
}
