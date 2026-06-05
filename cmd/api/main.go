package main

import (
	"gopher-fraud-detection/internal/router"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/service"
	"gopher-fraud-detection/internal/vectorizer"
	"log"
	"net"
	"net/http"
	"os"
)

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func loadIdx(path string, nProbe int) *search.IVFHIndex {
	idx, err := search.LoadIVFHIndex(path, nProbe)
	if err != nil {
		log.Fatalf("load index %s: %v", path, err)
	}
	return idx
}

func main() {
	firstKnownPath    := envOr("FIRST_KNOWN_INDEX_PATH",    "index/first_known.ivfh")
	firstUnknownPath  := envOr("FIRST_UNKNOWN_INDEX_PATH",  "index/first_unknown.ivfh")
	subseqKnownPath   := envOr("SUBSEQ_KNOWN_INDEX_PATH",   "index/subseq_known.ivfh")
	subseqUnknownPath := envOr("SUBSEQ_UNKNOWN_INDEX_PATH", "index/subseq_unknown.ivfh")
	normPath          := envOr("NORM_PATH", "resources/normalization.json")
	mccPath           := envOr("MCC_PATH",  "resources/mcc_risk.json")

	vec, err := vectorizer.Load(normPath, mccPath)
	if err != nil {
		log.Fatalf("load vectorizer: %v", err)
	}

	firstKnownIdx    := loadIdx(firstKnownPath,    search.NCoarseProbeFirstKnown)
	firstUnknownIdx  := loadIdx(firstUnknownPath,  search.NCoarseProbeFirstUnknown)
	subseqKnownIdx   := loadIdx(subseqKnownPath,   search.NCoarseProbeSubseqKnown)
	subseqUnknownIdx := loadIdx(subseqUnknownPath, search.NCoarseProbeSubseqUnknown)
	defer firstKnownIdx.Close()
	defer firstUnknownIdx.Close()
	defer subseqKnownIdx.Close()
	defer subseqUnknownIdx.Close()

	service.Vec = vec
	service.FirstKnownIdx = firstKnownIdx
	service.FirstUnknownIdx = firstUnknownIdx
	service.SubseqKnownIdx = subseqKnownIdx
	service.SubseqUnknownIdx = subseqUnknownIdx

	log.Printf("loaded first_known=%d first_unknown=%d subseq_known=%d subseq_unknown=%d",
		firstKnownIdx.N, firstUnknownIdx.N, subseqKnownIdx.N, subseqUnknownIdx.N)

	sock := envOr("SOCK", "")
	if sock == "" {
		log.Fatal("SOCK environment variable is required")
	}

	_ = os.Remove(sock)

	listener, err := net.Listen("unix", sock)
	if err != nil {
		log.Fatal(err)
	}
	if err := os.Chmod(sock, 0666); err != nil {
		log.Fatal(err)
	}

	server := &http.Server{Handler: router.New()}
	log.Fatal(server.Serve(listener))
}
