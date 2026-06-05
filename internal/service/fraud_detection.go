package service

import (
	"gopher-fraud-detection/internal/dto"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/vectorizer"
)

var (
	// Indices holds the 12 partition indexes indexed by 4-bit tag.
	// Tags 12-15 (online & card_present) are nil — router falls back.
	Indices [search.NPartitions]*search.IvfIndex
	Vec     *vectorizer.Vectorizer
)

// CalculateFraudScore returns the fraud count (0..5) for a request.
func CalculateFraudScore(req dto.FraudRequest) int {
	if count, ok := fastPath(req); ok {
		return count
	}
	tag := vectorizer.TagFromRequest(req)
	idx := Indices[tag]
	if idx == nil {
		tag &^= 8 // clear card_present; try online-only
		idx = Indices[tag]
	}
	if idx == nil {
		tag &^= 4 // clear is_online; try base tag
		idx = Indices[tag]
	}
	if idx == nil {
		return 0 // fallback: approve
	}
	q := Vec.Vectorize(req)
	return int(idx.Search(&q))
}
