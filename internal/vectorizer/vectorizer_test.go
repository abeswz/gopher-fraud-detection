package vectorizer

import (
	"testing"

	"gopher-fraud-detection/internal/dto"
)

// tagFromVec mirrors the 4-bit tag logic for test verification.
func tagFromVec(v [16]int16) int {
	tag := 0
	if v[5] >= 0 {
		tag |= 1
	} // has_last_tx
	if v[11] > 0 {
		tag |= 2
	} // unknown_merchant
	if v[9] > 0 {
		tag |= 4
	} // is_online
	if v[10] > 0 {
		tag |= 8
	} // card_present
	return tag
}

type lastTxArgs struct {
	Timestamp     string
	KmFromCurrent float64
}

func makeReq(amount float64, installments int, custAvg float64, txCount24h int,
	kmFromHome float64, knownMerchant bool, isOnline bool, cardPresent bool,
	mcc string, merchantAvg float64, lastTx *lastTxArgs) dto.FraudRequest {

	var merchants []string
	merchantID := "merchant-abc"
	if knownMerchant {
		merchants = []string{merchantID}
	}
	req := dto.FraudRequest{
		Transaction: dto.Transaction{
			Amount:       amount,
			Installments: installments,
			RequestedAt:  "2024-01-15T14:30:00Z",
		},
		Customer: dto.Customer{
			AvgAmount:      custAvg,
			TxCount24h:     txCount24h,
			KnownMerchants: merchants,
		},
		Merchant: dto.Merchant{
			ID:        merchantID,
			MCC:       mcc,
			AvgAmount: merchantAvg,
		},
		Terminal: dto.Terminal{
			KmFromHome:  kmFromHome,
			IsOnline:    isOnline,
			CardPresent: cardPresent,
		},
	}
	if lastTx != nil {
		req.LastTx = &dto.LastTransaction{
			Timestamp:     lastTx.Timestamp,
			KmFromCurrent: lastTx.KmFromCurrent,
		}
	}
	return req
}

func TestVectorize_Sentinel(t *testing.T) {
	v := &Vectorizer{
		Norm: Normalization{
			MaxAmount: 10000, MaxInstallments: 12,
			AmountVsAvgRatio: 10, MaxMinutes: 1440, MaxKm: 1000,
			MaxTxCount24h: 20, MaxMerchantAvgAmount: 10000,
		},
		MccRisk: map[string]float32{"5411": 0.1},
	}
	req := makeReq(500, 1, 1000, 5, 50, true, false, false, "5411", 1000, nil)
	vec := v.Vectorize(req)
	// last_tx nil → sentinel -10000
	if vec[5] != -10000 {
		t.Errorf("sentinel v[5] = %d, want -10000", vec[5])
	}
	if vec[6] != -10000 {
		t.Errorf("sentinel v[6] = %d, want -10000", vec[6])
	}
	tag := tagFromVec(vec)
	if tag&1 != 0 {
		t.Errorf("has_last_tx bit should be 0, tag=%d", tag)
	}
}

func TestVectorize_IsOnline(t *testing.T) {
	v := &Vectorizer{
		Norm: Normalization{MaxAmount: 10000, MaxInstallments: 12,
			AmountVsAvgRatio: 10, MaxMinutes: 1440, MaxKm: 1000,
			MaxTxCount24h: 20, MaxMerchantAvgAmount: 10000},
		MccRisk: map[string]float32{},
	}
	req := makeReq(100, 1, 500, 2, 10, false, true, false, "1234", 500, nil)
	vec := v.Vectorize(req)
	if vec[9] != 10000 {
		t.Errorf("is_online v[9] = %d, want 10000", vec[9])
	}
	if vec[10] != 0 {
		t.Errorf("card_present v[10] = %d, want 0", vec[10])
	}
}

func TestVectorize_Clamp(t *testing.T) {
	v := &Vectorizer{
		Norm: Normalization{MaxAmount: 10000, MaxInstallments: 12,
			AmountVsAvgRatio: 10, MaxMinutes: 1440, MaxKm: 1000,
			MaxTxCount24h: 20, MaxMerchantAvgAmount: 10000},
		MccRisk: map[string]float32{},
	}
	// amount = 99999 >> MaxAmount: clamped to 1.0 → 10000
	req := makeReq(99999, 1, 100, 1, 10, false, false, false, "1234", 100, nil)
	vec := v.Vectorize(req)
	if vec[0] != 10000 {
		t.Errorf("clamped v[0] = %d, want 10000", vec[0])
	}
}
