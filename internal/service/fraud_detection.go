package service

import (
	"gopher-fraud-detection/internal/dto"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/vectorizer"
)

var (
	FirstKnownIdx    search.Index
	FirstUnknownIdx  search.Index
	SubseqKnownIdx   search.Index
	SubseqUnknownIdx search.Index
	Vec              *vectorizer.Vectorizer
)

func isUnknownMerchant(req dto.FraudRequest) bool {
	for _, m := range req.Customer.KnownMerchants {
		if m == req.Merchant.ID {
			return false
		}
	}
	return true
}

func CalculateFraudScore(req dto.FraudRequest) int {
	if count, ok := fastPath(req); ok {
		return count
	}
	vec := Vec.Vectorize(req)
	if count, ok := RawTreePredict(vec); ok {
		return count
	}
	unknown := isUnknownMerchant(req)
	if req.LastTx == nil {
		if unknown {
			return FirstUnknownIdx.KNN(vec, 5)
		}
		return FirstKnownIdx.KNN(vec, 5)
	}
	if unknown {
		return SubseqUnknownIdx.KNN(vec, 5)
	}
	return SubseqKnownIdx.KNN(vec, 5)
}
