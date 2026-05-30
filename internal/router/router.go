package router

import (
	"gopher-fraud-detection/internal/handler"
	"net/http"
)

func New() *http.ServeMux {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /ready", handler.Ready)
	mux.HandleFunc("POST /fraud-score", handler.FraudScore)

	return mux
}
