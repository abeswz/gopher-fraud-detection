package handler

import (
	"encoding/json"
	"gopher-fraud-detection/internal/dto"
	"gopher-fraud-detection/internal/service"
	"net/http"
)

// Pre-computed responses for all 6 possible fraudCount values (0–5).
// Verified against encoding/json output: json.NewEncoder.Encode(FraudResponse{...}).
var fraudResponses = [6][]byte{
	[]byte("{\"approved\":true,\"fraud_score\":0}\n"),
	[]byte("{\"approved\":true,\"fraud_score\":0.2}\n"),
	[]byte("{\"approved\":true,\"fraud_score\":0.4}\n"),
	[]byte("{\"approved\":false,\"fraud_score\":0.6}\n"),
	[]byte("{\"approved\":false,\"fraud_score\":0.8}\n"),
	[]byte("{\"approved\":false,\"fraud_score\":1}\n"),
}

func FraudScore(w http.ResponseWriter, r *http.Request) {
	var req dto.FraudRequest

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	fraudCount := service.CalculateFraudScore(req)

	w.Header().Set("Content-Type", "application/json")
	w.Write(fraudResponses[fraudCount]) //nolint:errcheck
}
