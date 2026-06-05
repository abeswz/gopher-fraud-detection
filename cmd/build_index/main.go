// cmd/build_index/main.go
// Builds 12 flat IVF partition index files from references.json.gz.
// Usage: build_index <input.json[.gz]> <output_dir>
// Output: output_dir/index_p0.bin .. output_dir/index_p15.bin (skips empty partitions)
package main

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"strconv"
	"time"

	"gopher-fraud-detection/internal/search"
)

const (
	maxRefs     = 3_500_000
	kmeansIters = 20
)

func readInput(path string) ([]byte, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) >= 2 && raw[0] == 0x1f && raw[1] == 0x8b {
		zr, err := gzip.NewReader(bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		defer zr.Close()
		return io.ReadAll(zr)
	}
	return raw, nil
}

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: build_index <input.json[.gz]> <output_dir>")
		os.Exit(1)
	}
	inPath, outDir := os.Args[1], os.Args[2]

	log := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[build_index] "+format+"\n", a...)
	}
	t0 := time.Now()

	buf, err := readInput(inPath)
	if err != nil {
		log("read input: %v", err)
		os.Exit(1)
	}
	log("read %d MB", len(buf)>>20)

	refs, err := search.ParseCorpusRefs(buf, maxRefs)
	if err != nil {
		log("parse: %v", err)
		os.Exit(1)
	}
	log("parsed %d refs (%.1fs)", len(refs), time.Since(t0).Seconds())
	buf = nil // allow GC

	built := 0
	for tag := 0; tag < search.NPartitions; tag++ {
		part := search.FilterByTag(refs, tag)
		if len(part) == 0 {
			continue
		}

		// Adaptive K: n/300, clamp [64, MaxK]
		k := len(part) / 300
		if k < 64 {
			k = 64
		}
		if k > search.MaxK {
			k = search.MaxK
		}
		log("tag %d: %d refs, K=%d", tag, len(part), k)

		tk := time.Now()
		cent, assignments := search.KMeans(part, k, kmeansIters)
		log("tag %d: k-means done (%.1fs)", tag, time.Since(tk).Seconds())

		offsets, order := search.CountingSortByCluster(assignments, k)
		search.SortWithinClusters(part, cent, assignments, offsets, order)
		bboxMin, bboxMax, pairArr, labels := search.BBoxPack(part, order, offsets, k)

		outPath := outDir + "/index_p" + strconv.Itoa(tag) + ".bin"
		if err := search.WriteIndexBin(outPath, len(part), offsets, bboxMin, bboxMax, pairArr, labels); err != nil {
			log("write %s: %v", outPath, err)
			os.Exit(1)
		}
		log("tag %d: wrote %s", tag, outPath)
		built++
	}

	log("done: %d partitions, total %.1fs", built, time.Since(t0).Seconds())
}
