package service

import (
	"encoding/json"

	"gopher-fraud-detection/internal/dto"
)

// CalculateFraudScoreRaw parses the JSON body and calls CalculateFraudScore.
// Returns -1 on parse error (caller sends 400).
func CalculateFraudScoreRaw(body []byte) int {
	var req dto.FraudRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return -1
	}
	return CalculateFraudScore(req)
}
