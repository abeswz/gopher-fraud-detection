package service

import (
	"testing"

	"gopher-fraud-detection/internal/dto"
	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/vectorizer"
)

var _ search.Index = (*stubIndex)(nil)

// stubIndex returns a fixed fraudCount regardless of query.
type stubIndex struct{ count int }

func (s *stubIndex) KNN(_ [16]float32, _ int) int { return s.count }

func setupStubs() {
	FirstKnownIdx = &stubIndex{count: 0}
	FirstUnknownIdx = &stubIndex{count: 1}
	SubseqKnownIdx = &stubIndex{count: 2}
	SubseqUnknownIdx = &stubIndex{count: 3}
	Vec = &vectorizer.Vectorizer{MccRisk: map[string]float32{}}
}

func TestRouting_NilLastTx_KnownMerchant(t *testing.T) {
	setupStubs()
	FirstKnownIdx = &stubIndex{count: 4}

	req := dto.FraudRequest{
		Customer: dto.Customer{KnownMerchants: []string{"MERC-01"}},
		Merchant: dto.Merchant{ID: "MERC-01"},
	}
	got := CalculateFraudScore(req)
	if got != 4 {
		t.Errorf("first+known routing: got %d, want 4", got)
	}
}

func TestRouting_NilLastTx_UnknownMerchant(t *testing.T) {
	setupStubs()
	FirstUnknownIdx = &stubIndex{count: 5}

	req := dto.FraudRequest{
		Customer: dto.Customer{KnownMerchants: []string{"MERC-01"}},
		Merchant: dto.Merchant{ID: "MERC-99"},
	}
	got := CalculateFraudScore(req)
	if got != 5 {
		t.Errorf("first+unknown routing: got %d, want 5", got)
	}
}

func TestRouting_SubseqKnown(t *testing.T) {
	setupStubs()

	lastTx := &dto.LastTransaction{}
	req := dto.FraudRequest{
		LastTx:   lastTx,
		Customer: dto.Customer{KnownMerchants: []string{"MERC-01"}},
		Merchant: dto.Merchant{ID: "MERC-01"},
	}
	got := CalculateFraudScore(req)
	if got != 2 {
		t.Errorf("subseq+known routing: got %d, want 2", got)
	}
}

func TestRouting_SubseqUnknown(t *testing.T) {
	setupStubs()

	lastTx := &dto.LastTransaction{}
	req := dto.FraudRequest{
		LastTx:   lastTx,
		Customer: dto.Customer{KnownMerchants: []string{"MERC-01"}},
		Merchant: dto.Merchant{ID: "MERC-99"},
	}
	got := CalculateFraudScore(req)
	if got != 3 {
		t.Errorf("subseq+unknown routing: got %d, want 3", got)
	}
}
