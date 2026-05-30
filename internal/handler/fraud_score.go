package handler

import (
	"encoding/json"
	"gopher-fraud-detection/internal/dto"
	"gopher-fraud-detection/internal/service"
	"net/http"
)

func FraudScore(w http.ResponseWriter, r *http.Request) {
	var req dto.FraudRequest

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	resp := service.CalculateFraudScore(req)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	if err := json.NewEncoder(w).Encode(resp); err != nil {
		http.Error(w, "failed to enconde response", http.StatusInternalServerError)
		return
	}
}
