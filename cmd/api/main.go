package main

import (
	"gopher-fraud-detection/internal/router"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/service"
	"gopher-fraud-detection/internal/vectorizer"
	"io"
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

func readMagic(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	var buf [4]byte
	_, err = io.ReadFull(f, buf[:])
	return string(buf[:]), err
}

func main() {
	indexPath := envOr("INDEX_PATH", "index/references.bin")
	normPath := envOr("NORM_PATH", "resources/normalization.json")
	mccPath := envOr("MCC_PATH", "resources/mcc_risk.json")

	vec, err := vectorizer.Load(normPath, mccPath)
	if err != nil {
		log.Fatalf("load vectorizer: %v", err)
	}

	magic, err := readMagic(indexPath)
	if err != nil {
		log.Fatalf("read index magic: %v", err)
	}

	var idx search.Index
	var n int

	switch magic {
	case "IVF1":
		ivf, err := search.LoadIVFIndex(indexPath)
		if err != nil {
			log.Fatalf("load IVF index: %v", err)
		}
		idx = ivf
		n = ivf.N
	case "VPT1":
		vp, err := search.LoadVPIndex(indexPath)
		if err != nil {
			log.Fatalf("load VP index: %v", err)
		}
		idx = vp
		n = len(vp.Labels)
	default:
		log.Fatalf("unknown index format: %q (want IVF1 or VPT1)", magic)
	}

	service.Vec = vec
	service.Idx = idx

	log.Printf("loaded %d vectors (format: %s)", n, magic)

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
