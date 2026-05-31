// internal/service/fraud_detection.go
package service

import (
	"gopher-fraud-detection/internal/dto"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/vectorizer"
)

var (
	Idx search.Index
	Vec *vectorizer.Vectorizer
)

// CalculateFraudScore returns fraudCount (0–5): number of fraud neighbors among k=5.
// k=5 and threshold=0.6 are fixed by spec — do not change.
// Pipeline: fast_path → decision_tree → k-NN (each step only runs if previous returns ok=false).
func CalculateFraudScore(req dto.FraudRequest) int {
	if count, ok := fastPath(req); ok {
		return count
	}
	vec := Vec.Vectorize(req)
	if count, ok := search.DecisionTree(vec); ok {
		return count
	}
	return Idx.KNN(vec, 5)
}
