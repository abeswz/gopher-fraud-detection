package service

import (
	"testing"

	"gopher-fraud-detection/internal/dto"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/vectorizer"
)

// stubIndex returns a fixed fraudCount regardless of query.
type stubIndex struct{ count int }

func (s *stubIndex) KNN(_ [16]float32, _ int) int { return s.count }

func TestRouting_NilLastTx_UsesFirstIdx(t *testing.T) {
	FirstTxIdx = &stubIndex{count: 4} // fraud
	SubseqIdx = &stubIndex{count: 0}  // legit
	Vec = &vectorizer.Vectorizer{MccRisk: map[string]float32{}}
	_ = search.Index(FirstTxIdx)

	req := dto.FraudRequest{}
	// LastTx is nil (zero value for *LastTransaction) → routes to FirstTxIdx → fraudCount=4
	got := CalculateFraudScore(req)
	if got != 4 {
		t.Errorf("nil LastTx routing: got %d, want 4", got)
	}
}

func TestRouting_NonNilLastTx_UsesSubseqIdx(t *testing.T) {
	FirstTxIdx = &stubIndex{count: 0}  // legit
	SubseqIdx = &stubIndex{count: 3}   // fraud
	Vec = &vectorizer.Vectorizer{MccRisk: map[string]float32{}}

	lastTx := &dto.LastTransaction{}
	req := dto.FraudRequest{LastTx: lastTx}
	// LastTx is non-nil → should route to SubseqIdx → fraudCount=3
	got := CalculateFraudScore(req)
	if got != 3 {
		t.Errorf("nonNil LastTx routing: got %d, want 3", got)
	}
}
