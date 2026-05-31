package service

import (
	"testing"

	"gopher-fraud-detection/internal/dto"
)

func TestFastPath_SafeSpend(t *testing.T) {
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 80, Installments: 2},
		Customer:    dto.Customer{AvgAmount: 200, TxCount24h: 3, KnownMerchants: []string{"MERC-01"}},
		Merchant:    dto.Merchant{ID: "MERC-01", MCC: "5411"},
		Terminal:    dto.Terminal{KmFromHome: 10},
	}
	count, ok := fastPath(req)
	if !ok {
		t.Fatal("expected fast path hit")
	}
	if count != 0 {
		t.Errorf("safe spend: got count=%d, want 0", count)
	}
}

func TestFastPath_RiskySpend(t *testing.T) {
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 8000, Installments: 7},
		Customer:    dto.Customer{AvgAmount: 100, TxCount24h: 10, KnownMerchants: []string{"MERC-01"}},
		Merchant:    dto.Merchant{ID: "MERC-99", MCC: "7995"},
		Terminal:    dto.Terminal{KmFromHome: 300},
	}
	count, ok := fastPath(req)
	if !ok {
		t.Fatal("expected fast path hit")
	}
	if count != 5 {
		t.Errorf("risky spend: got count=%d, want 5", count)
	}
}

func TestFastPath_SafeMissOneCondition(t *testing.T) {
	// safe in all ways except amount is 600 (> 500)
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 600, Installments: 2},
		Customer:    dto.Customer{AvgAmount: 2000, TxCount24h: 3, KnownMerchants: []string{"MERC-01"}},
		Merchant:    dto.Merchant{ID: "MERC-01", MCC: "5411"},
		Terminal:    dto.Terminal{KmFromHome: 10},
	}
	_, ok := fastPath(req)
	if ok {
		t.Fatal("should not hit fast path when amount > 500")
	}
}

func TestFastPath_RiskyMissOneCondition(t *testing.T) {
	// risky in all ways except installments is 4 (< 5)
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 8000, Installments: 4},
		Customer:    dto.Customer{AvgAmount: 100, TxCount24h: 10, KnownMerchants: []string{"MERC-01"}},
		Merchant:    dto.Merchant{ID: "MERC-99", MCC: "7995"},
		Terminal:    dto.Terminal{KmFromHome: 300},
	}
	_, ok := fastPath(req)
	if ok {
		t.Fatal("should not hit fast path when installments < 5")
	}
}

func TestFastPath_SafeHighAmountVsAvg(t *testing.T) {
	// amount=400 but > 50% of avg_amount=600 (400 > 300)
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 400, Installments: 2},
		Customer:    dto.Customer{AvgAmount: 600, TxCount24h: 3, KnownMerchants: []string{"MERC-01"}},
		Merchant:    dto.Merchant{ID: "MERC-01", MCC: "5411"},
		Terminal:    dto.Terminal{KmFromHome: 10},
	}
	_, ok := fastPath(req)
	if ok {
		t.Fatal("should not hit safe path when amount > 50%% of avg_amount")
	}
}

func TestFastPath_SafeUnknownMerchant(t *testing.T) {
	req := dto.FraudRequest{
		Transaction: dto.Transaction{Amount: 80, Installments: 2},
		Customer:    dto.Customer{AvgAmount: 200, TxCount24h: 3, KnownMerchants: []string{"MERC-01"}},
		Merchant:    dto.Merchant{ID: "MERC-99", MCC: "5411"}, // unknown merchant
		Terminal:    dto.Terminal{KmFromHome: 10},
	}
	_, ok := fastPath(req)
	if ok {
		t.Fatal("should not hit safe path for unknown merchant")
	}
}
