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

func CalculateFraudScore(req dto.FraudRequest) dto.FraudResponse {
	vec := Vec.Vectorize(req)
	fraudCount := Idx.KNN(vec, 5)
	fraudScore := float64(fraudCount) / 5.0
	return dto.FraudResponse{
		Approved:   fraudScore < 0.6,
		FraudScore: fraudScore,
	}
}
