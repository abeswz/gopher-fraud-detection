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

func main() {
	firstIdxPath := envOr("FIRST_TX_INDEX_PATH", "index/first_tx.ivfh")
	subseqIdxPath := envOr("SUBSEQ_INDEX_PATH", "index/subsequent_tx.ivfh")
	normPath := envOr("NORM_PATH", "resources/normalization.json")
	mccPath := envOr("MCC_PATH", "resources/mcc_risk.json")

	vec, err := vectorizer.Load(normPath, mccPath)
	if err != nil {
		log.Fatalf("load vectorizer: %v", err)
	}

	firstIdx, err := search.LoadIVFHIndex(firstIdxPath, search.NCoarseProbeFirst)
	if err != nil {
		log.Fatalf("load first_tx index: %v", err)
	}
	defer firstIdx.Close()

	subseqIdx, err := search.LoadIVFHIndex(subseqIdxPath, search.NCoarseProbeSubseq)
	if err != nil {
		log.Fatalf("load subsequent_tx index: %v", err)
	}
	defer subseqIdx.Close()

	service.Vec = vec
	service.FirstTxIdx = firstIdx
	service.SubseqIdx = subseqIdx

	log.Printf("loaded first_tx: %d vectors, subsequent_tx: %d vectors", firstIdx.N, subseqIdx.N)

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
