// internal/service/fraud_detection.go
package service

import (
	"gopher-fraud-detection/internal/dto"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/vectorizer"
)

var (
	Idx *search.IVFIndex
	Vec *vectorizer.Vectorizer
)

// CalculateFraudScore returns fraudCount (0–5): number of fraud neighbors among k=5.
// k=5 and threshold=0.6 are fixed by spec — do not change.
// Pipeline: fast_path → vectorize → raw_tree → IVF k-NN (AVX2).
// RawTreePredict short-circuits ~96.5% of requests before the expensive IVF call.
func CalculateFraudScore(req dto.FraudRequest) int {
	if count, ok := fastPath(req); ok {
		return count
	}
	vec := Vec.Vectorize(req)
	if count, ok := RawTreePredict(vec); ok {
		return count
	}
	return Idx.KNN(vec, 5)
}
