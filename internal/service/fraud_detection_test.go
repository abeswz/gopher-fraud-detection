package service

import (
	"testing"

	"gopher-fraud-detection/internal/dto"
)

func TestCalculateFraudScore_FastPath(t *testing.T) {
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 100, Installments: 1, RequestedAt: "2024-01-15T14:00:00Z"},
		Customer:    dto.Customer{AvgAmount: 500, TxCount24h: 2, KnownMerchants: []string{"m1"}},
		Merchant:    dto.Merchant{ID: "m1", MCC: "5411", AvgAmount: 200},
		Terminal:    dto.Terminal{KmFromHome: 5, IsOnline: false, CardPresent: false},
	}
	score := CalculateFraudScore(req)
	if score != 0 {
		t.Errorf("fast path should return 0, got %d", score)
	}
}
