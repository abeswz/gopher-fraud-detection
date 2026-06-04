package service

import (
	"testing"
	"time"

	"gopher-fraud-detection/internal/dto"
)

func makeTestRequest(amount, custAvg, merchantAvg float64, installments, txCount24h int,
	kmFromHome float64, mcc string, isOnline, cardPresent bool, known bool,
	lastKm float64, lastTimeDeltaSec float64) dto.FraudRequest {

	merchantID := "merch-A"
	knownMerchants := []string{}
	if known {
		knownMerchants = []string{merchantID}
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
			KnownMerchants: knownMerchants,
		},
		Merchant: dto.Merchant{
			ID:        merchantID,
			MCC:       mcc,
			AvgAmount: merchantAvg,
		},
		Terminal: dto.Terminal{
			IsOnline:    isOnline,
			CardPresent: cardPresent,
			KmFromHome:  kmFromHome,
		},
	}

	if lastTimeDeltaSec >= 0 {
		lastTime := time.Date(2024, 1, 15, 14, 30, 0, 0, time.UTC).
			Add(-time.Duration(lastTimeDeltaSec) * time.Second)
		req.LastTx = &dto.LastTransaction{
			Timestamp:     lastTime.Format(time.RFC3339),
			KmFromCurrent: lastKm,
		}
	}
	return req
}

var testMccRisk = map[string]float32{
	"5411": 0.1,
	"7995": 0.9,
}

func TestExtractRawFeatures_NoLastTx(t *testing.T) {
	req := makeTestRequest(
		200.0, 400.0, 100.0,
		2, 3,
		10.0, "5411", false, true, true,
		-1, -1,
	)
	f := ExtractRawFeatures(req, testMccRisk)

	if f.Amount != float32(200.0) {
		t.Errorf("Amount: got %v, want 200", f.Amount)
	}
	if f.AmountRatio != float32(200.0/400.0) {
		t.Errorf("AmountRatio: got %v, want 0.5", f.AmountRatio)
	}
	if f.Installments != float32(2) {
		t.Errorf("Installments: got %v, want 2", f.Installments)
	}
	if f.TxCount24h != float32(3) {
		t.Errorf("TxCount24h: got %v, want 3", f.TxCount24h)
	}
	if f.KmFromHome != float32(10.0) {
		t.Errorf("KmFromHome: got %v, want 10", f.KmFromHome)
	}
	if f.IsKnownMerchant != 1.0 {
		t.Errorf("IsKnownMerchant: got %v, want 1", f.IsKnownMerchant)
	}
	if f.MccRiskScore != 0.1 {
		t.Errorf("MccRiskScore: got %v, want 0.1", f.MccRiskScore)
	}
	if f.LastKmFromCurrent != -1.0 {
		t.Errorf("LastKmFromCurrent: got %v, want -1 (sentinel)", f.LastKmFromCurrent)
	}
	if f.LastTimeDeltaSec != -1.0 {
		t.Errorf("LastTimeDeltaSec: got %v, want -1 (sentinel)", f.LastTimeDeltaSec)
	}
	if f.AmountOverMax != 0.0 {
		t.Errorf("AmountOverMax: got %v, want 0", f.AmountOverMax)
	}
	if f.IsSafeMCC != 1.0 {
		t.Errorf("IsSafeMCC: got %v, want 1 (mcc 5411 is safe)", f.IsSafeMCC)
	}
	if f.IsRiskyMCC != 0.0 {
		t.Errorf("IsRiskyMCC: got %v, want 0", f.IsRiskyMCC)
	}
	wantAmtNorm := float32(200.0 / 10000.0)
	if f.AmountNormalized != wantAmtNorm {
		t.Errorf("AmountNormalized: got %v, want %v", f.AmountNormalized, wantAmtNorm)
	}
}

func TestExtractRawFeatures_ZeroAvgAmount(t *testing.T) {
	req := makeTestRequest(500.0, 0.0, 0.0, 1, 1, 5.0, "9999", false, true, false, -1, -1)
	f := ExtractRawFeatures(req, testMccRisk)

	if f.AmountRatio != 0.0 {
		t.Errorf("AmountRatio with zero AvgAmount: got %v, want 0.0", f.AmountRatio)
	}
	if f.MerchantAmountRatio != 0.0 {
		t.Errorf("MerchantAmountRatio with zero MerchantAvgAmount: got %v, want 0.0", f.MerchantAmountRatio)
	}
	if f.MccRiskScore != 0.5 {
		t.Errorf("MccRiskScore unknown MCC: got %v, want 0.5", f.MccRiskScore)
	}
}

func TestExtractRawFeatures_AmountOverMax(t *testing.T) {
	req := makeTestRequest(15000.0, 1000.0, 500.0, 1, 1, 5.0, "5411", false, true, false, -1, -1)
	f := ExtractRawFeatures(req, testMccRisk)
	if f.AmountOverMax != 1.0 {
		t.Errorf("AmountOverMax: got %v, want 1 for amount>10000", f.AmountOverMax)
	}
	if f.AmountNormalized != 1.0 {
		t.Errorf("AmountNormalized: got %v, want 1.0 (clamped)", f.AmountNormalized)
	}
}

func TestExtractRawFeatures_WithLastTx(t *testing.T) {
	req := makeTestRequest(200.0, 400.0, 100.0, 1, 1, 5.0, "5411", false, true, false, 3.5, 120.0)
	f := ExtractRawFeatures(req, testMccRisk)
	if f.LastKmFromCurrent != float32(3.5) {
		t.Errorf("LastKmFromCurrent: got %v, want 3.5", f.LastKmFromCurrent)
	}
	if f.LastTimeDeltaSec != float32(120.0) {
		t.Errorf("LastTimeDeltaSec: got %v, want 120", f.LastTimeDeltaSec)
	}
}

func TestExtractRawFeatures_HourOfDay(t *testing.T) {
	req := makeTestRequest(100.0, 200.0, 50.0, 1, 1, 5.0, "5411", false, true, false, -1, -1)
	f := ExtractRawFeatures(req, testMccRisk)
	if f.HourOfDay != float32(14) {
		t.Errorf("HourOfDay: got %v, want 14", f.HourOfDay)
	}
}

func TestRawTreePredict_ReturnsValidFraudCount(t *testing.T) {
	tests := []struct {
		name string
		vec  [16]float32
	}{
		{"zero vector", [16]float32{}},
		{"high amount norm", [16]float32{0: 0.9}},
		{"risky mcc", [16]float32{12: 0.9, 0: 0.5}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			count, ok := RawTreePredict(tc.vec)
			if ok {
				if count != 0 && count != 5 {
					t.Errorf("confident leaf returned %d, want 0 or 5", count)
				}
			} else {
				if count != 0 {
					t.Errorf("uncertain leaf returned %d, want 0", count)
				}
			}
		})
	}
}

func TestCalculateFraudScore_PipelineSmoke(t *testing.T) {
	if FirstTxIdx == nil || Vec == nil {
		t.Skip("service globals not initialized")
	}

	req := makeTestRequest(100.0, 200.0, 50.0, 1, 1, 5.0, "5411", false, true, true, -1, -1)
	count := CalculateFraudScore(req)
	if count < 0 || count > 5 {
		t.Errorf("CalculateFraudScore: got %d, want 0–5", count)
	}
}
