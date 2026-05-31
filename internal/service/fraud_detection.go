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
func CalculateFraudScore(req dto.FraudRequest) int {
	vec := Vec.Vectorize(req)
	return Idx.KNN(vec, 5)
}
