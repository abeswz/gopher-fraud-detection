package vectorizer

import (
	"math"
	"testing"

	"gopher-fraud-detection/internal/dto"
)

var testNorm = Normalization{
	MaxAmount:            10000,
	MaxInstallments:      12,
	AmountVsAvgRatio:     10,
	MaxMinutes:           1440,
	MaxKm:                1000,
	MaxTxCount24h:        20,
	MaxMerchantAvgAmount: 10000,
}

var testMcc = map[string]float32{
	"5411": 0.15,
	"7802": 0.75,
}

func approxEqual(a, b float32, tol float64) bool {
	return math.Abs(float64(a-b)) <= tol
}

func checkVec(t *testing.T, got [16]float32, want [14]float32) {
	t.Helper()
	for i := range want {
		if !approxEqual(got[i], want[i], 1e-3) {
			t.Errorf("dim[%d]: got %.4f, want %.4f", i, got[i], want[i])
		}
	}
}

// Example 1 from DETECTION_RULES.md — legit, last_transaction null
// Expected vector: [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
func TestVectorize_LegitNullLastTx(t *testing.T) {
	v := &Vectorizer{Norm: testNorm, MccRisk: testMcc}
	req := dto.FraudRequest{
		Transaction: dto.Transaction{
			Amount:       41.12,
			Installments: 2,
			RequestedAt:  "2026-03-11T18:45:53Z",
		},
		Customer: dto.Customer{
			AvgAmount:      82.24,
			TxCount24h:     3,
			KnownMerchants: []string{"MERC-003", "MERC-016"},
		},
		Merchant: dto.Merchant{ID: "MERC-016", MCC: "5411", AvgAmount: 60.25},
		Terminal: dto.Terminal{IsOnline: false, CardPresent: true, KmFromHome: 29.2331036248},
		LastTx:   nil,
	}
	want := [14]float32{0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006}
	checkVec(t, v.Vectorize(req), want)
}

// Example 2 from DETECTION_RULES.md — fraud, last_transaction null
// Expected vector: [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]
func TestVectorize_FraudNullLastTx(t *testing.T) {
	v := &Vectorizer{Norm: testNorm, MccRisk: testMcc}
	req := dto.FraudRequest{
		Transaction: dto.Transaction{
			Amount:       9505.97,
			Installments: 10,
			RequestedAt:  "2026-03-14T05:15:12Z",
		},
		Customer: dto.Customer{
			AvgAmount:      81.28,
			TxCount24h:     20,
			KnownMerchants: []string{"MERC-008", "MERC-007", "MERC-005"},
		},
		Merchant: dto.Merchant{ID: "MERC-068", MCC: "7802", AvgAmount: 54.86},
		Terminal: dto.Terminal{IsOnline: false, CardPresent: true, KmFromHome: 952.2745933273},
		LastTx:   nil,
	}
	want := [14]float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}
	checkVec(t, v.Vectorize(req), want)
}

// Known merchant → unknown_merchant = 0
func TestVectorize_KnownMerchant(t *testing.T) {
	v := &Vectorizer{Norm: testNorm, MccRisk: testMcc}
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 100, Installments: 1, RequestedAt: "2026-01-01T12:00:00Z"},
		Customer:    dto.Customer{AvgAmount: 100, TxCount24h: 1, KnownMerchants: []string{"MERC-A", "MERC-B"}},
		Merchant:    dto.Merchant{ID: "MERC-A", MCC: "5411", AvgAmount: 100},
		Terminal:    dto.Terminal{},
		LastTx:      nil,
	}
	got := v.Vectorize(req)
	if got[11] != 0 {
		t.Errorf("known merchant: dim[11] got %.1f, want 0", got[11])
	}
}

// Unknown merchant → unknown_merchant = 1
func TestVectorize_UnknownMerchant(t *testing.T) {
	v := &Vectorizer{Norm: testNorm, MccRisk: testMcc}
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 100, Installments: 1, RequestedAt: "2026-01-01T12:00:00Z"},
		Customer:    dto.Customer{AvgAmount: 100, TxCount24h: 1, KnownMerchants: []string{"MERC-A", "MERC-B"}},
		Merchant:    dto.Merchant{ID: "MERC-Z", MCC: "9999", AvgAmount: 100},
		Terminal:    dto.Terminal{},
		LastTx:      nil,
	}
	got := v.Vectorize(req)
	if got[11] != 1 {
		t.Errorf("unknown merchant: dim[11] got %.1f, want 1", got[11])
	}
	if got[12] != 0.5 {
		t.Errorf("unknown mcc default: dim[12] got %.1f, want 0.5", got[12])
	}
}

// last_transaction present → dims 5,6 computed
func TestVectorize_WithLastTx(t *testing.T) {
	v := &Vectorizer{Norm: testNorm, MccRisk: testMcc}
	req := dto.FraudRequest{
		Transaction: dto.Transaction{
			Amount:       384.88,
			Installments: 3,
			RequestedAt:  "2026-03-11T20:23:35Z",
		},
		Customer: dto.Customer{
			AvgAmount:      769.76,
			TxCount24h:     3,
			KnownMerchants: []string{"MERC-009", "MERC-009", "MERC-001", "MERC-001"},
		},
		Merchant: dto.Merchant{ID: "MERC-001", MCC: "5912", AvgAmount: 298.95},
		Terminal: dto.Terminal{IsOnline: false, CardPresent: true, KmFromHome: 13.7090520965},
		LastTx: &dto.LastTransaction{
			Timestamp:     "2026-03-11T14:58:35Z",
			KmFromCurrent: 18.8626479774,
		},
	}
	got := v.Vectorize(req)
	// minutes between 14:58:35 and 20:23:35 = 5h25m = 325 minutes
	// clamp(325/1440) = 0.2257
	if got[5] < 0.224 || got[5] > 0.228 {
		t.Errorf("minutes_since_last_tx: got %.4f, want ~0.2257", got[5])
	}
	// clamp(18.8626/1000) = 0.0189
	if got[6] < 0.018 || got[6] > 0.020 {
		t.Errorf("km_from_last_tx: got %.4f, want ~0.0189", got[6])
	}
	// dims 5,6 must not be -1
	if got[5] < 0 || got[6] < 0 {
		t.Errorf("dims 5,6 should not be -1 when last_tx present")
	}
}
