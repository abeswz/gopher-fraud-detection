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
	indexPath := envOr("INDEX_PATH", "index/references.bin")
	normPath := envOr("NORM_PATH", "resources/normalization.json")
	mccPath := envOr("MCC_PATH", "resources/mcc_risk.json")

	vec, err := vectorizer.Load(normPath, mccPath)
	if err != nil {
		log.Fatalf("load vectorizer: %v", err)
	}

	idx, err := search.LoadIVFIndex(indexPath)
	if err != nil {
		log.Fatalf("load index: %v", err)
	}

	service.Vec = vec
	service.Idx = idx

	log.Printf("loaded %d vectors", idx.N)

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
