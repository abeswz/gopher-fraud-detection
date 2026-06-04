package service

import (
	"gopher-fraud-detection/internal/dto"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/vectorizer"
)

var (
	FirstTxIdx search.Index
	SubseqIdx  search.Index
	Vec        *vectorizer.Vectorizer
)

// CalculateFraudScore returns fraudCount (0–5): fraud neighbors among k=5.
// Routes by LastTx: nil → firstTx index, non-nil → subsequent index.
func CalculateFraudScore(req dto.FraudRequest) int {
	if count, ok := fastPath(req); ok {
		return count
	}
	vec := Vec.Vectorize(req)
	if count, ok := RawTreePredict(vec); ok {
		return count
	}
	if req.LastTx == nil {
		return FirstTxIdx.KNN(vec, 5)
	}
	return SubseqIdx.KNN(vec, 5)
}
